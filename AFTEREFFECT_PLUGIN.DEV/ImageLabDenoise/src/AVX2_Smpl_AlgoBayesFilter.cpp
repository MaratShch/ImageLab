#include <cmath>
#include <algorithm>
#include <cstring>
#include <immintrin.h>
#include "AVX2_Smpl_AlgoBayesFilter.hpp"
#include "AVX2_Smpl_AlgoMatrixMath.hpp"
#include "AVX2_Smpl_AlgoBlockMatch.hpp"

// =========================================================
// FAST HORIZONTAL SUM HELPER (SINGLE PRECISION)
// =========================================================
inline float AVX2_Smpl_HSum_PS(__m256 v) noexcept {
    __m128 vlow = _mm256_castps256_ps128(v);
    __m128 vhigh = _mm256_extractf128_ps(v, 1);
    vlow = _mm_add_ps(vlow, vhigh);
    __m128 shuf = _mm_movehl_ps(vlow, vlow);
    vlow = _mm_add_ps(vlow, shuf);
    shuf = _mm_shuffle_ps(vlow, vlow, _MM_SHUFFLE(1, 1, 1, 1));
    vlow = _mm_add_ss(vlow, shuf);
    return _mm_cvtss_f32(vlow);
}

// =========================================================
// HELPER: STEP 1 - SINGLE CHANNEL SHRINKAGE
// =========================================================
inline void AVX2_Smpl_Apply_Bayes_Shrinkage_Step1
(
    float* RESTRICT vol_noisy, 
    const int32_t num_patches, 
    const float* RESTRICT covLUT, 
    const float noise_mult
) noexcept
{
    constexpr int32_t K = 16; 
    float C_empirical[256];
    float C_noise[256];
    float C_diff[256];
    float Shrinkage[256];
    float mean_P[16];

    const float inv_N = 1.0f / static_cast<float>(num_patches);
    const float inv_Nm1 = 1.0f / static_cast<float>(num_patches - 1);

    __m256 vMean[2] = { _mm256_setzero_ps(), _mm256_setzero_ps() };
    for (int32_t p = 0; p < num_patches; ++p) {
        vMean[0] = _mm256_add_ps(vMean[0], _mm256_loadu_ps(&vol_noisy[p * K]));
        vMean[1] = _mm256_add_ps(vMean[1], _mm256_loadu_ps(&vol_noisy[p * K + 8]));
    }
    __m256 vInvN = _mm256_set1_ps(inv_N);
    _mm256_storeu_ps(&mean_P[0], _mm256_mul_ps(vMean[0], vInvN));
    _mm256_storeu_ps(&mean_P[8], _mm256_mul_ps(vMean[1], vInvN));

    std::memset(C_empirical, 0, sizeof(C_empirical));
    for (int32_t p = 0; p < num_patches; ++p) {
        float temp[16];
        for (int32_t k = 0; k < K; ++k) temp[k] = vol_noisy[p * K + k] - mean_P[k];
        
        for (int32_t i = 0; i < K; ++i) {
            __m256 vA = _mm256_set1_ps(temp[i]);
            int32_t j = 0;
            for (; j <= i - 7; j += 8) {
                __m256 vB = _mm256_loadu_ps(&temp[j]);
                __m256 vC = _mm256_loadu_ps(&C_empirical[i * K + j]);
                _mm256_storeu_ps(&C_empirical[i * K + j], _mm256_fmadd_ps(vA, vB, vC));
            }
            for (; j <= i; ++j) {
                C_empirical[i * K + j] += temp[i] * temp[j];
            }
        }
    }
    for (int32_t i = 0; i < K; ++i) {
        for (int32_t j = 0; j <= i; ++j) {
            C_empirical[j * K + i] = (C_empirical[i * K + j] *= inv_Nm1); 
        }
    }

    int32_t intensity = std::max(0, std::min(255, static_cast<int32_t>(mean_P[0]))); 
    for (int32_t i = 0; i < 256; ++i) {
        C_noise[i] = covLUT[intensity * 256 + i] * noise_mult;
    }

    float trace_diff = 0.0f;
    for (int32_t i = 0; i < K; ++i) {
        float val = C_empirical[i * K + i] - C_noise[i * K + i];
        C_diff[i * K + i] = std::max(0.0f, val);
        trace_diff += val;
    }
    for (int32_t i = 0; i < K; ++i) {
        for (int32_t j = 0; j < K; ++j) {
            if (i != j) C_diff[i * K + j] = C_empirical[i * K + j] - C_noise[i * K + j];
        }
    }

    if (trace_diff <= 0.0f) {
        for (int32_t p = 0; p < num_patches; ++p) {
            for (int32_t i = 0; i < K; ++i) vol_noisy[p * K + i] = mean_P[i];
        }
        return;
    }

    // Tikhonov Regularization for 32-bit float stability
    for (int32_t i = 0; i < K; ++i) C_empirical[i * K + i] += 0.001f;

    if (AVX2_Smpl_Inverse_Matrix(C_empirical, K)) {
        AVX2_Smpl_Product_Matrix(Shrinkage, C_diff, C_empirical, K, K, K);
        
        for (int32_t p = 0; p < num_patches; ++p) {
            float out_patch[16];
            for (int32_t i = 0; i < K; ++i) {
                __m256 vSum = _mm256_setzero_ps();
                for (int32_t j = 0; j < K; j += 8) {
                    __m256 w = _mm256_loadu_ps(&Shrinkage[i * K + j]);
                    __m256 f = _mm256_loadu_ps(&vol_noisy[p * K + j]);
                    __m256 b = _mm256_loadu_ps(&mean_P[j]);
                    __m256 c = _mm256_sub_ps(f, b);
                    vSum = _mm256_fmadd_ps(w, c, vSum);
                }
                out_patch[i] = mean_P[i] + AVX2_Smpl_HSum_PS(vSum);
            }
            for (int32_t i = 0; i < K; ++i) vol_noisy[p * K + i] = out_patch[i];
        }
    }
}

// =========================================================
// HELPER: STEP 2 - JOINT EMPIRICAL WIENER
// =========================================================
// =========================================================
// HELPER: STEP 2 - DECOUPLED EMPIRICAL WIENER (PHASE 2)
// =========================================================
inline void AVX2_Smpl_Apply_Bayes_Shrinkage_Step2_Decoupled
(
    float* RESTRICT vol_noisy_joint, 
    float* RESTRICT vol_basic_joint, 
    const int32_t num_patches,
    const float* RESTRICT covLUT_Y,
    const float* RESTRICT covLUT_U,
    const float* RESTRICT covLUT_V,
    const float noise_mult
) noexcept
{
    constexpr int32_t K = 16; 
    const float inv_N = 1.0f / static_cast<float>(num_patches);
    const float inv_Nm1 = 1.0f / static_cast<float>(num_patches - 1);
    
    // Array of pointers to easily loop through the 3 channels
    const float* covLUTs[3] = { covLUT_Y, covLUT_U, covLUT_V };

    // Process Y, U, and V as completely independent 16x16 matrices
    for (int32_t c = 0; c < 3; ++c) 
    {
        CACHE_ALIGN float C_basic[256];
        CACHE_ALIGN float C_sum[256];
        CACHE_ALIGN float Shrinkage[256];
        CACHE_ALIGN float barycenter[16];
        const int32_t ch_offset = c * 16;
        
        // 1. Barycenter of Noisy Data (for this specific channel)
        __m256 vBary[2] = { _mm256_setzero_ps(), _mm256_setzero_ps() };
        for (int32_t p = 0; p < num_patches; ++p)
        {
            vBary[0] = _mm256_add_ps(vBary[0], _mm256_loadu_ps(&vol_noisy_joint[p * 48 + ch_offset]));
            vBary[1] = _mm256_add_ps(vBary[1], _mm256_loadu_ps(&vol_noisy_joint[p * 48 + ch_offset + 8]));
        }
        __m256 vInvN = _mm256_set1_ps(inv_N);
        _mm256_storeu_ps(&barycenter[0], _mm256_mul_ps(vBary[0], vInvN));
        _mm256_storeu_ps(&barycenter[8], _mm256_mul_ps(vBary[1], vInvN));

        // 2. Covariance of Basic Data
        std::memset(C_basic, 0, sizeof(C_basic));
        for (int32_t p = 0; p < num_patches; ++p)
        {
            CACHE_ALIGN float temp[16];
            for (int32_t k = 0; k < K; ++k) temp[k] = vol_basic_joint[p * 48 + ch_offset + k] - barycenter[k];
            
            for (int32_t i = 0; i < K; ++i)
            {
                __m256 vA = _mm256_set1_ps(temp[i]);
                int32_t j = 0;
                for (; j <= i - 7; j += 8)
                {
                    __m256 vB = _mm256_loadu_ps(&temp[j]);
                    __m256 vC = _mm256_loadu_ps(&C_basic[i * K + j]);
                    _mm256_storeu_ps(&C_basic[i * K + j], _mm256_fmadd_ps(vA, vB, vC));
                }
                for (; j <= i; ++j)
                {
                    C_basic[i * K + j] += temp[i] * temp[j];
                }
            }
        }
        for (int32_t i = 0; i < K; ++i)
        {
            for (int32_t j = 0; j <= i; ++j)
            {
                C_basic[j * K + i] = (C_basic[i * K + j] *= inv_Nm1); 
            }
        }

        std::memcpy(C_sum, C_basic, K * K * sizeof(float));

        // 3. Add Noise Covariance
        int32_t intensity = std::max(0, std::min(255, static_cast<int32_t>(barycenter[0])));
        for (int32_t i = 0; i < K; ++i)
        {
            for (int32_t j = 0; j < K; ++j)
            {
                C_sum[i * K + j] += covLUTs[c][intensity * 256 + i * 16 + j] * noise_mult;
            }
        }

        // Tikhonov Regularization for 32-bit float stability
        for (int32_t i = 0; i < K; ++i) C_sum[i * K + i] += 0.001f;

        // 4. Shrinkage Math
        if (AVX2_Smpl_Inverse_Matrix(C_sum, K))
        {
            AVX2_Smpl_Product_Matrix(Shrinkage, C_basic, C_sum, K, K, K);

            for (int32_t p = 0; p < num_patches; ++p)
            {
                CACHE_ALIGN float out_patch[16];
                for (int32_t i = 0; i < K; ++i)
                {
                    __m256 vSum = _mm256_setzero_ps();
                    for (int32_t j = 0; j < K; j += 8)
                    {
                        __m256 w = _mm256_loadu_ps(&Shrinkage[i * K + j]);
                        __m256 f = _mm256_loadu_ps(&vol_noisy_joint[p * 48 + ch_offset + j]);
                        __m256 b = _mm256_loadu_ps(&barycenter[j]);
                        __m256 c_diff = _mm256_sub_ps(f, b);
                        vSum = _mm256_fmadd_ps(w, c_diff, vSum);
                    }
                    out_patch[i] = barycenter[i] + AVX2_Smpl_HSum_PS(vSum);
                }
                // Write back to the basic joint payload area directly
                for (int32_t i = 0; i < K; ++i)
                {
                    vol_basic_joint[p * 48 + ch_offset + i] = out_patch[i];
                }
            }
        }
    }
}

// =========================================================
// VECTORIZED AGGREGATION HELPERS
// =========================================================
inline void AVX2_Smpl_Aggregate_3D_Group
(
    const float* RESTRICT vol, 
    const int32_t num_patches, 
    const PatchDistance* RESTRICT pool,
    float* RESTRICT accum, 
    float* RESTRICT weight,
    const int32_t width
) noexcept
{
    __m128 vOne = _mm_set1_ps(1.0f);
    
    for (int32_t p = 0; p < num_patches; ++p) {
        const int32_t px = pool[p].x;
        const int32_t py = pool[p].y;
        
        for (int32_t i = 0; i < 4; ++i) {
            if (p == 0 && (i == 1 || i == 2)) {
                for (int32_t j = 0; j < 4; ++j) {
                    if (j == 1 || j == 2) continue;
                    const int32_t idx = (py + i) * width + (px + j);
                    accum[idx] += vol[p * 16 + (i * 4 + j)];
                    weight[idx] += 1.0f;
                }
            } else {
                const int32_t idx = (py + i) * width + px;
                __m128 vVol = _mm_loadu_ps(&vol[p * 16 + i * 4]);
                __m128 vAccum = _mm_loadu_ps(&accum[idx]);
                __m128 vWeight = _mm_loadu_ps(&weight[idx]);
                
                _mm_storeu_ps(&accum[idx], _mm_add_ps(vAccum, vVol));
                _mm_storeu_ps(&weight[idx], _mm_add_ps(vWeight, vOne));
            }
        }
    }
}

inline void AVX2_Smpl_Aggregate_3D_Group_Payload
(
    const float* RESTRICT vol, 
    const int32_t num_patches, 
    const PatchDistance* RESTRICT pool,
    float* RESTRICT accum, 
    const int32_t width
) noexcept
{
    for (int32_t p = 0; p < num_patches; ++p) {
        const int32_t px = pool[p].x;
        const int32_t py = pool[p].y;
        
        for (int32_t i = 0; i < 4; ++i) {
            if (p == 0 && (i == 1 || i == 2)) {
                for (int32_t j = 0; j < 4; ++j) {
                    if (j == 1 || j == 2) continue;
                    accum[(py + i) * width + (px + j)] += vol[p * 16 + (i * 4 + j)];
                }
            } else {
                const int32_t idx = (py + i) * width + px;
                __m128 vVol = _mm_loadu_ps(&vol[p * 16 + i * 4]);
                __m128 vAccum = _mm_loadu_ps(&accum[idx]);
                _mm_storeu_ps(&accum[idx], _mm_add_ps(vAccum, vVol));
            }
        }
    }
}

inline void AVX2_Smpl_Aggregate_3D_Group_Joint
(
    const float* RESTRICT vol_joint, 
    const int32_t num_patches, 
    const PatchDistance* RESTRICT pool,
    float* RESTRICT accum_Y, float* RESTRICT accum_U, float* RESTRICT accum_V,
    float* RESTRICT weight, const int32_t width
) noexcept
{
    __m128 vOne = _mm_set1_ps(1.0f);
    
    for (int32_t p = 0; p < num_patches; ++p) {
        const int32_t px = pool[p].x;
        const int32_t py = pool[p].y;
        
        for (int32_t i = 0; i < 4; ++i) {
            if (p == 0 && (i == 1 || i == 2)) {
                for (int32_t j = 0; j < 4; ++j) {
                    if (j == 1 || j == 2) continue;
                    const int32_t idx = (py + i) * width + (px + j);
                    const int32_t offset = p * 48 + (i * 4 + j);
                    accum_Y[idx] += vol_joint[offset];
                    accum_U[idx] += vol_joint[offset + 16];
                    accum_V[idx] += vol_joint[offset + 32];
                    weight[idx] += 1.0f;
                }
            } else {
                const int32_t idx = (py + i) * width + px;
                const int32_t offset = p * 48 + i * 4;
                
                __m128 vY = _mm_loadu_ps(&vol_joint[offset]);
                __m128 vU = _mm_loadu_ps(&vol_joint[offset + 16]);
                __m128 vV = _mm_loadu_ps(&vol_joint[offset + 32]);
                
                _mm_storeu_ps(&accum_Y[idx], _mm_add_ps(_mm_loadu_ps(&accum_Y[idx]), vY));
                _mm_storeu_ps(&accum_U[idx], _mm_add_ps(_mm_loadu_ps(&accum_U[idx]), vU));
                _mm_storeu_ps(&accum_V[idx], _mm_add_ps(_mm_loadu_ps(&accum_V[idx]), vV));
                _mm_storeu_ps(&weight[idx], _mm_add_ps(_mm_loadu_ps(&weight[idx]), vOne));
            }
        }
    }
}


// =========================================================
// MAIN API: DUAL-PASS BAYES FILTER
// =========================================================
void AVX2_Smpl_Process_Scale_NL_Bayes
(
    const MemHandler& mem,
    float* RESTRICT Y_scale, 
    float* RESTRICT U_scale, 
    float* RESTRICT V_scale,
    const int32_t width, 
    const int32_t height,
    const float noise_variance_multiplier
)
{
    const int32_t frameSize = width * height;
    
    float* RESTRICT Pilot_Y = mem.OracleWorkspace;
    float* RESTRICT Pilot_U = mem.OracleWorkspace + frameSize;
    float* RESTRICT Pilot_V = mem.OracleWorkspace + (2 * frameSize);
    float* RESTRICT Pilot_W = mem.OracleWorkspace + (3 * frameSize);

    std::memset(Pilot_Y, 0, frameSize * sizeof(float));
    std::memset(Pilot_U, 0, frameSize * sizeof(float));
    std::memset(Pilot_V, 0, frameSize * sizeof(float));
    std::memset(Pilot_W, 0, frameSize * sizeof(float));

    std::memset(mem.Accum_Y, 0, frameSize * sizeof(float));
    std::memset(mem.Accum_U, 0, frameSize * sizeof(float));
    std::memset(mem.Accum_V, 0, frameSize * sizeof(float));
    std::memset(mem.Weight_Count, 0, frameSize * sizeof(float));

    const float tau_step1 = 48.0f * noise_variance_multiplier; 
    const float tau_step2 = 144.0f * noise_variance_multiplier; 
    
    float* RESTRICT vol_Y = mem.Scratch3D;
    float* RESTRICT vol_U = mem.Scratch3D + (512 * 16);
    float* RESTRICT vol_V = mem.Scratch3D + (512 * 32);

    // =========================================================
    // STEP 1: BASIC ESTIMATE (Generate Pilot)
    // =========================================================
    for (int32_t y = 0; y < height - 4; y += 2)
    {
        for (int32_t x = 0; x < width - 4; x += 2)
        {
            const int32_t N = AVX2_Smpl_Extract_Similar_Patches
            (
                Y_scale, U_scale, V_scale, width, height, x, y, 
                tau_step1, mem.SearchPool
            );
            
            if (N < 16) continue;

            for (int32_t p = 0; p < N; ++p)
            {
                int32_t px = mem.SearchPool[p].x;
                int32_t py = mem.SearchPool[p].y;
                for (int32_t i = 0; i < 4; ++i)
                {
                    for (int32_t j = 0; j < 4; ++j)
                    {
                        const int32_t src_idx = (py + i) * width + px + j;
                        const int32_t dst_idx = p * 16 + (i * 4 + j);
                        vol_Y[dst_idx] = Y_scale[src_idx];
                        vol_U[dst_idx] = U_scale[src_idx];
                        vol_V[dst_idx] = V_scale[src_idx];
                    }
                }
            }

            AVX2_Smpl_Apply_Bayes_Shrinkage_Step1(vol_Y, N, mem.NoiseCov_Y, noise_variance_multiplier);
            AVX2_Smpl_Apply_Bayes_Shrinkage_Step1(vol_U, N, mem.NoiseCov_U, noise_variance_multiplier);
            AVX2_Smpl_Apply_Bayes_Shrinkage_Step1(vol_V, N, mem.NoiseCov_V, noise_variance_multiplier);

            AVX2_Smpl_Aggregate_3D_Group(vol_Y, N, mem.SearchPool, Pilot_Y, Pilot_W, width);
            AVX2_Smpl_Aggregate_3D_Group_Payload(vol_U, N, mem.SearchPool, Pilot_U, width);
            AVX2_Smpl_Aggregate_3D_Group_Payload(vol_V, N, mem.SearchPool, Pilot_V, width);
        }
    }

    // Normalize Pilot
    for (int32_t i = 0; i < frameSize; ++i)
    {
        if (Pilot_W[i] > 0.0f)
        {
            Pilot_Y[i] /= Pilot_W[i];
            Pilot_U[i] /= Pilot_W[i];
            Pilot_V[i] /= Pilot_W[i];
        } else
        {
            Pilot_Y[i] = Y_scale[i];
            Pilot_U[i] = U_scale[i];
            Pilot_V[i] = V_scale[i];
        }
    }

    // =========================================================
    // STEP 2: FINAL ESTIMATE (Empirical Wiener using Pilot)
    // =========================================================
    float* RESTRICT vol_noisy_joint = mem.Scratch3D;
    float* RESTRICT vol_basic_joint = mem.Scratch3D + (512 * 48);

    for (int32_t y = 0; y < height - 4; y += 2)
    {
        for (int32_t x = 0; x < width - 4; x += 2)
        {
            
            int32_t N = AVX2_Smpl_Extract_Similar_Patches(
                Pilot_Y, Pilot_U, Pilot_V, width, height, x, y, 
                tau_step2, mem.SearchPool
            );
            
            if (N < 16) continue;

            for (int32_t p = 0; p < N; ++p)
            {
                int32_t px = mem.SearchPool[p].x;
                int32_t py = mem.SearchPool[p].y;
                for (int32_t i = 0; i < 4; ++i)
                {
                    for (int32_t j = 0; j < 4; ++j)
                    {
                        const int32_t src_idx = (py + i) * width + px + j;
                        const int32_t dst_idx = p * 48 + (i * 4 + j);
                        
                        vol_noisy_joint[dst_idx]      = Y_scale[src_idx];
                        vol_noisy_joint[dst_idx + 16] = U_scale[src_idx];
                        vol_noisy_joint[dst_idx + 32] = V_scale[src_idx];
                        
                        vol_basic_joint[dst_idx]      = Pilot_Y[src_idx];
                        vol_basic_joint[dst_idx + 16] = Pilot_U[src_idx];
                        vol_basic_joint[dst_idx + 32] = Pilot_V[src_idx];
                    }
                }
            }

            AVX2_Smpl_Apply_Bayes_Shrinkage_Step2_Decoupled
            (
                vol_noisy_joint, vol_basic_joint, N, 
                mem.NoiseCov_Y, mem.NoiseCov_U, mem.NoiseCov_V, 
                noise_variance_multiplier
            );

            AVX2_Smpl_Aggregate_3D_Group_Joint
            (
                vol_basic_joint, N, mem.SearchPool, 
                mem.Accum_Y, mem.Accum_U, mem.Accum_V, mem.Weight_Count, width
            );
        }
    }

    // Normalize Final Accumulators
    for (int32_t i = 0; i < frameSize; ++i)
    {
        if (mem.Weight_Count[i] > 0.0f) {
            mem.Accum_Y[i] /= mem.Weight_Count[i];
            mem.Accum_U[i] /= mem.Weight_Count[i];
            mem.Accum_V[i] /= mem.Weight_Count[i];
        } else {
            mem.Accum_Y[i] = Y_scale[i];
            mem.Accum_U[i] = U_scale[i];
            mem.Accum_V[i] = V_scale[i];
        }
    }

}