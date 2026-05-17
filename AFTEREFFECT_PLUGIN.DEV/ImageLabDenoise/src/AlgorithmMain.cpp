#include <cstdint>
#include <cstring>
#include <cfloat>
#include <iostream>
#include <algorithm>
#include <immintrin.h>
#include "AlgoMemHandler.hpp"
#include "AlgoControls.hpp"
#include "AlgorithmMain.hpp"

// NC Algorithm headers
#include "AVX2_AlgoPyramidBuilder.hpp"
#include "AVX2_AlgoNoiseOracle.hpp"
#include "AVX2_Smpl_AlgoBayesFilter.hpp"

inline void Pad_Edges_YUV
(
    float* RESTRICT plane,
    int32_t sizeX,
    int32_t sizeY,
    int32_t padW,
    int32_t padH
) noexcept
{
    if (padW > sizeX)
    {
        for (int32_t y = sizeY - 1; y > 0; --y)
        {
            for (int32_t x = sizeX - 1; x >= 0; --x)
            {
                plane[y * padW + x] = plane[y * sizeX + x];
            }
        }
    }

    for (int32_t y = 0; y < sizeY; ++y)
    {
        const float last_val = plane[y * padW + (sizeX - 1)];
        for (int32_t x = sizeX; x < padW; ++x)
        {
            plane[y * padW + x] = last_val;
        }
    }
    
    for (int32_t y = sizeY; y < padH; ++y)
    {
        for (int32_t x = 0; x < padW; ++x)
        {
            plane[y * padW + x] = plane[(sizeY - 1) * padW + x];
        }
    }
}

inline void Unpack_Edges_YUV
(
    float* RESTRICT plane,
    int32_t sizeX,
    int32_t sizeY,
    int32_t padW
) noexcept
{
    if (padW > sizeX)
    {
        for (int32_t y = 1; y < sizeY; ++y)
        {
            for (int32_t x = 0; x < sizeX; ++x)
            {
                plane[y * sizeX + x] = plane[y * padW + x];
            }
        }
    }
}

// =========================================================
// NOISE MAP HELPERS (Option A: dedicated buffers in MemHandler)
// =========================================================
//
// Snapshot the noisy YUV into MemHandler::*_noisy_orig BEFORE pyramid build
// mutates the planar buffers via Laplacian reconstruction. Only invoked when
// the caller requested OutputType::NoiseMap; zero overhead otherwise.
//
// The snapshot covers the full padded plane (padW x padH) -- copying the
// padded edges is harmless since we ignore them at noise-map compute time.
//
inline void Snapshot_Noisy_YUV
(
    const MemHandler& mem
) noexcept
{
    const size_t bytes = static_cast<size_t>(mem.padW) * mem.padH * sizeof(float);
    std::memcpy(mem.Y_noisy_orig, mem.Y_planar, bytes);
    std::memcpy(mem.U_noisy_orig, mem.U_planar, bytes);
    std::memcpy(mem.V_noisy_orig, mem.V_planar, bytes);
}


// =========================================================
// AVX2 MAX-SCAN over the noisy Y plane.
//
// Used to detect the engine's CURRENT value scale at runtime. The engine
// internally operates in different absolute ranges depending on the input
// pixel format (BGRA_8u -> [0, 442], BGRA_16u -> [0, 56754], BGRA_32f ->
// [0, 1.732], etc.). All ranges share one invariant:
//
//   max_Y_white = (3 * white_RGB) * (1/sqrt(3)) = white_RGB * sqrt(3)
//   mid_gray_Y  = max_Y_white / 2     (exactly half of white)
//
// So a single max scan over the noisy snapshot gives us the engine scale,
// from which the mid-gray Y offset follows trivially.
//
// Scans only the active sizeX x sizeY region (not the padded edges).
// =========================================================
inline float AVX2_Find_Max_Y_Noisy
(
    const float* RESTRICT noisy_Y,
    const int32_t         padW,
    const int32_t         sizeX,
    const int32_t         sizeY
) noexcept
{
    __m256 vMax = _mm256_set1_ps(-FLT_MAX);

    for (int32_t y = 0; y < sizeY; ++y)
    {
        const float* RESTRICT row = noisy_Y + y * padW;

        int32_t x = 0;
        for (; x <= sizeX - 8; x += 8)
        {
            __m256 v = _mm256_loadu_ps(row + x);
            vMax = _mm256_max_ps(vMax, v);
        }
        // Scalar tail
        for (; x < sizeX; ++x)
        {
            // Reduce vMax to scalar lazily only if needed -- but cheaper to
            // just merge per-tail using a single broadcast comparison.
            __m256 vTail = _mm256_set1_ps(row[x]);
            vMax = _mm256_max_ps(vMax, vTail);
        }
    }

    // Horizontal max across the 8 lanes of vMax.
    __m128 lo = _mm256_castps256_ps128(vMax);
    __m128 hi = _mm256_extractf128_ps(vMax, 1);
    __m128 m4 = _mm_max_ps(lo, hi);
    __m128 m2 = _mm_max_ps(m4, _mm_movehl_ps(m4, m4));
    __m128 m1 = _mm_max_ss(m2, _mm_shuffle_ps(m2, m2, _MM_SHUFFLE(1, 1, 1, 1)));
    return _mm_cvtss_f32(m1);
}


// =========================================================
// AVX2 NOISE MAP COMPUTATION (per-channel, in-place on Accum_*)
//
// For each pixel in the active sizeX x sizeY region, replaces the denoised
// YUV value in Accum_* with the noise map YUV value:
//
//   accum[i] = mid_gray_offset + AMP * (noisy[i] - accum[i])
//
// where mid_gray_offset is the YUV value at the engine's CURRENT scale
// (auto-adapted via the noisy max-Y scan) that maps to mid-gray RGB
// through the orthonormal output transform. Because the YUV<->RGB
// transform is linear, the same downstream converter handles both
// denoised image and noise map outputs identically -- the converter
// API is NOT modified.
//
// The noise map output values match the dynamic range of the denoised
// output for the active pixel format:
//   - BGRA_8u    : values in [0, ~442],     "mid-gray Y" ~ 221
//   - BGRA_16u   : values in [0, ~56754],   "mid-gray Y" ~ 28377
//   - RGB_10u    : values in [0, ~1772],    "mid-gray Y" ~ 886
//   - BGRA_32f   : values in [0, ~1.732],   "mid-gray Y" ~ 0.866
//
// CALLED BEFORE Unpack_Edges_YUV: both accum and noisy_orig are at stride
// = padW (padded layout). The unpack step that follows compacts the
// result to host's tight pitch identically for denoised and noise-map
// modes.
//
// AMP = 8.0 (constexpr) matches the GPU output kernel.
// =========================================================
inline void AVX2_Compute_NoiseMap_Channel
(
    float* RESTRICT       accum,        // in/out: denoised -> noise map
    const float* RESTRICT noisy_orig,
    const float           mid_offset,   // engine-scale mid-gray for this channel
    const int32_t         padW,
    const int32_t         sizeX,
    const int32_t         sizeY
) noexcept
{
    constexpr float NOISE_MAP_AMP = 8.0f;

    const __m256 vAmp    = _mm256_set1_ps(NOISE_MAP_AMP);
    const __m256 vOffset = _mm256_set1_ps(mid_offset);

    for (int32_t y = 0; y < sizeY; ++y)
    {
        float* RESTRICT       row_accum = accum      + y * padW;
        const float* RESTRICT row_noisy = noisy_orig + y * padW;

        int32_t x = 0;
        for (; x <= sizeX - 8; x += 8)
        {
            __m256 vNoisy = _mm256_loadu_ps(row_noisy + x);
            __m256 vDen   = _mm256_loadu_ps(row_accum + x);
            __m256 vDiff  = _mm256_sub_ps(vNoisy, vDen);                      // noisy - denoised
            __m256 vRes   = _mm256_fmadd_ps(vAmp, vDiff, vOffset);             // offset + AMP*diff
            _mm256_storeu_ps(row_accum + x, vRes);
        }
        for (; x < sizeX; ++x)
        {
            row_accum[x] = mid_offset + NOISE_MAP_AMP * (row_noisy[x] - row_accum[x]);
        }
    }
}


inline void AVX2_Build_NoiseMap_YUV
(
    const MemHandler& mem,
    const int32_t     sizeX,
    const int32_t     sizeY
) noexcept
{
    // -----------------------------------------------------------------------
    // ADAPTIVE SCALE DETECTION
    //
    // The engine internally operates in different absolute ranges depending
    // on the input pixel format. Rather than tracking the format through
    // many abstraction layers, we DETECT the current scale empirically by
    // scanning the noisy Y plane for its maximum.
    //
    // Math: for any orthonormal YUV in any scale,
    //   max_Y_white = white_RGB * sqrt(3)
    //   mid_gray_Y  = max_Y_white / 2
    //
    // For U and V: any neutral gray (R = G = B) yields U = 0 and V = 0
    // exactly, regardless of scale. So those offsets are always 0.
    //
    // SAFETY: a fully-dark scene would yield max_Y == 0, producing
    // mid_gray_Y == 0 (i.e., black-centered noise map). We guard against
    // this degenerate case with a small floor -- if no bright content
    // exists, we fall back to the FP32-style normalized constant so the
    // visualization remains stable rather than collapsing to black.
    // -----------------------------------------------------------------------
    const float max_Y_noisy = AVX2_Find_Max_Y_Noisy(mem.Y_noisy_orig, mem.padW, sizeX, sizeY);

    // For non-degenerate images (max_Y > a small threshold) use max/2 as
    // the engine-scale mid-gray. The threshold uses the smallest expected
    // scale (BGRA_32f range, white_Y ~= 1.732) so even very dim float
    // content still trips it.
    constexpr float MIN_VALID_MAX_Y = 0.5f;
    constexpr float FALLBACK_MID_Y  = 0.8660254037844386f;   // sqrt(3) / 2, BGRA_32f case

    const float MID_GRAY_Y = (max_Y_noisy > MIN_VALID_MAX_Y)
                             ? (max_Y_noisy * 0.5f)
                             :  FALLBACK_MID_Y;
    constexpr float MID_GRAY_U = 0.0f;
    constexpr float MID_GRAY_V = 0.0f;

    AVX2_Compute_NoiseMap_Channel(mem.Accum_Y, mem.Y_noisy_orig, MID_GRAY_Y,
                                  mem.padW, sizeX, sizeY);
    AVX2_Compute_NoiseMap_Channel(mem.Accum_U, mem.U_noisy_orig, MID_GRAY_U,
                                  mem.padW, sizeX, sizeY);
    AVX2_Compute_NoiseMap_Channel(mem.Accum_V, mem.V_noisy_orig, MID_GRAY_V,
                                  mem.padW, sizeX, sizeY);
}


// =========================================================
// ALGORITHM MAIN (MULTI-SCALE PYRAMID)
// =========================================================
void Algorithm_Main
(
    const MemHandler& mem,
    const int32_t sizeX,
    const int32_t sizeY,
    const AlgoControls& algoCtrl
)
{
    if (mem_handler_valid(mem))
    {
        // 0. DUPLICATE EDGES INTO PADDING
        Pad_Edges_YUV(mem.Y_planar, sizeX, sizeY, mem.padW, mem.padH);
        Pad_Edges_YUV(mem.U_planar, sizeX, sizeY, mem.padW, mem.padH);
        Pad_Edges_YUV(mem.V_planar, sizeX, sizeY, mem.padW, mem.padH);

        // 0a. NOISE MAP MODE: snapshot the noisy YUV NOW, before pyramid
        //     construction. The LEVEL 0 reconstruction below OVERWRITES
        //     Y/U/V_planar via AVX2_Reconstruct_Laplacian_Level, so the
        //     noisy originals would be lost without this copy. The snapshot
        //     is skipped entirely when the user wants the denoised image
        //     (zero overhead in normal mode).
        const bool want_noise_map = (algoCtrl.out == OutputType::NoiseMap);
        if (want_noise_map)
        {
            Snapshot_Noisy_YUV(mem);
        }

        // 1. CONSTRUCT PYRAMID
        AVX2_Build_Laplacian_Pyramid (mem, mem.padW, mem.padH);
        
        // 2. BLIND NOISE ORACLE
        AVX2_Estimate_Noise_Covariances (mem, mem.padW, mem.padH, algoCtrl);

        // 3. MULTI-SCALE DENOISING
        const int32_t qW = mem.padW / 4, qH = mem.padH / 4;
        const int32_t hW = mem.padW / 2, hH = mem.padH / 2;
        const int32_t fW = mem.padW,     fH = mem.padH;
        
        // --- LEVEL 2: QUARTER RESOLUTION (Coarse Noise) ---
        float mult_Q = algoCtrl.master_denoise_amount * algoCtrl.coarse_noise_reduction * 0.0625f;
        AVX2_Smpl_Process_Scale_NL_Bayes (mem, mem.Y_quart, mem.U_quart, mem.V_quart, qW, qH, mult_Q, algoCtrl);

        AVX2_Reconstruct_Laplacian_Level (mem.Accum_Y, mem.Y_diff_half, mem.Y_half, hW, hH);
        AVX2_Reconstruct_Laplacian_Level (mem.Accum_U, mem.U_diff_half, mem.U_half, hW, hH);
        AVX2_Reconstruct_Laplacian_Level (mem.Accum_V, mem.V_diff_half, mem.V_half, hW, hH);

        // --- LEVEL 1: HALF RESOLUTION (Standard Noise) ---
        float mult_H = algoCtrl.master_denoise_amount * 0.25f;
        AVX2_Smpl_Process_Scale_NL_Bayes (mem, mem.Y_half, mem.U_half, mem.V_half, hW, hH, mult_H, algoCtrl);

        AVX2_Reconstruct_Laplacian_Level (mem.Accum_Y, mem.Y_diff_full, mem.Y_planar, fW, fH);
        AVX2_Reconstruct_Laplacian_Level (mem.Accum_U, mem.U_diff_full, mem.U_planar, fW, fH);
        AVX2_Reconstruct_Laplacian_Level (mem.Accum_V, mem.V_diff_full, mem.V_planar, fW, fH);

        // --- LEVEL 0: FULL RESOLUTION (Fine Detail) ---
        float mult_F = algoCtrl.master_denoise_amount * algoCtrl.fine_detail_preservation * 1.0f;
        AVX2_Smpl_Process_Scale_NL_Bayes (mem, mem.Y_planar, mem.U_planar, mem.V_planar, fW, fH, mult_F, algoCtrl);
        
        // 4. NOISE MAP MODE: replace denoised values in Accum_* with the
        //    noise map (in YUV space). The downstream YUV->RGB conversion
        //    is UNCHANGED -- the orthonormal transform is linear, so the
        //    same conversion API processes both denoised image and noise
        //    map values identically. After this step, Accum_* holds:
        //
        //      Accum_Y = sqrt(3)/2 + 8 * (noisy_Y - denoised_Y)
        //      Accum_U = 0         + 8 * (noisy_U - denoised_U)
        //      Accum_V = 0         + 8 * (noisy_V - denoised_V)
        //
        //    which yields RGB = (0.5, 0.5, 0.5) + 8 * RGB_noise after the
        //    standard YUV->RGB step downstream.
        //
        //    IMPORTANT: this runs BEFORE Unpack_Edges_YUV. At this point
        //    both Accum_* and *_noisy_orig use stride = padW, so the
        //    helper's row addressing (row * padW + x) is correct for both.
        //    The Unpack step that follows will compact Accum_* to stride
        //    = sizeX exactly as in denoised mode.
        if (want_noise_map)
        {
            AVX2_Build_NoiseMap_YUV(mem, sizeX, sizeY);
        }

        // 5. RESTORE TIGHT HOST PITCH (same as denoised mode -- runs after
        //    the noise map transform so the host sees tightly packed rows
        //    regardless of which output type was selected).
        Unpack_Edges_YUV(mem.Accum_Y, sizeX, sizeY, mem.padW);
        Unpack_Edges_YUV(mem.Accum_U, sizeX, sizeY, mem.padW);
        Unpack_Edges_YUV(mem.Accum_V, sizeX, sizeY, mem.padW);
    }
}