#include <algorithm>
#include <immintrin.h>
#include "AVX2_AlgoNoiseOracle.hpp"
#include "AlgoNoiseOracle.hpp" // Required for OracleCandidate, BinInfo, Compute_MAD, etc.

// =========================================================
// AVX2 ACCELERATED DCT PROJECTION
// =========================================================
inline void AVX2_Forward_DCT_4x4
(
    const float* RESTRICT src, 
    const int32_t pitch, 
    float* RESTRICT dct_out, 
    const float D[16][16]
) noexcept
{
    // 1. Load the 4x4 spatial patch into two 256-bit registers (16 floats total)
    __m128 r0 = _mm_loadu_ps(src);
    __m128 r1 = _mm_loadu_ps(src + pitch);
    __m128 r2 = _mm_loadu_ps(src + 2 * pitch);
    __m128 r3 = _mm_loadu_ps(src + 3 * pitch);

    __m256 p01 = _mm256_insertf128_ps(_mm256_castps128_ps256(r0), r1, 1);
    __m256 p23 = _mm256_insertf128_ps(_mm256_castps128_ps256(r2), r3, 1);

    // 2. Project against the 16 DCT basis vectors
    for (int32_t k = 0; k < 16; ++k)
    {
        // Load the DCT basis row (16 floats)
        __m256 d01 = _mm256_loadu_ps(&D[k][0]);
        __m256 d23 = _mm256_loadu_ps(&D[k][8]);

        // Multiply Spatial Patch * DCT Basis
        __m256 m01 = _mm256_mul_ps(p01, d01);
        __m256 m23 = _mm256_mul_ps(p23, d23);
        __m256 vSum = _mm256_add_ps(m01, m23);

        // Fast horizontal sum of the 8 floats in vSum
        __m128 sum128 = _mm_add_ps(_mm256_castps256_ps128(vSum), _mm256_extractf128_ps(vSum, 1));
        __m128 sum64  = _mm_add_ps(sum128, _mm_movehl_ps(sum128, sum128));
        __m128 sum32  = _mm_add_ss(sum64, _mm_shuffle_ps(sum64, sum64, _MM_SHUFFLE(1, 1, 1, 1)));
        
        dct_out[k] = _mm_cvtss_f32(sum32);
    }
}

// =========================================================
// AVX2 ACCELERATED FREQUENCY DISTANCE
// =========================================================
inline float AVX2_Calculate_Sparse_Distance
(
    const float* RESTRICT dct1, 
    const float* RESTRICT dct2
) noexcept
{
    // Load the first 8 frequencies
    __m256 a0 = _mm256_loadu_ps(dct1);
    __m256 b0 = _mm256_loadu_ps(dct2);
    __m256 diff0 = _mm256_sub_ps(a0, b0);
    __m256 vSum = _mm256_mul_ps(diff0, diff0);

    // Load the remaining 8 frequencies and accumulate (FMA)
    __m256 a1 = _mm256_loadu_ps(dct1 + 8);
    __m256 b1 = _mm256_loadu_ps(dct2 + 8);
    __m256 diff1 = _mm256_sub_ps(a1, b1);
    vSum = _mm256_fmadd_ps(diff1, diff1, vSum);

    // Fast horizontal sum
    __m128 sum128 = _mm_add_ps(_mm256_castps256_ps128(vSum), _mm256_extractf128_ps(vSum, 1));
    __m128 sum64  = _mm_add_ps(sum128, _mm_movehl_ps(sum128, sum128));
    __m128 sum32  = _mm_add_ss(sum64, _mm_shuffle_ps(sum64, sum64, _MM_SHUFFLE(1, 1, 1, 1)));
    
    return _mm_cvtss_f32(sum32);
}

// =========================================================
// THE HYBRID AVX2/SCALAR ORACLE PIPELINE
// =========================================================
void AVX2_Process_Channel_Oracle
(
    const float* RESTRICT plane, 
    float* RESTRICT covLUT, 
    const int32_t pitch, 
    const int32_t height, 
    float* RESTRICT workspace,
    const int32_t workspaceSizeFloats
)
{
    CACHE_ALIGN float D[16][16];
    Generate_DCT_Basis(D);
    
    OracleCandidate* candidates = reinterpret_cast<OracleCandidate*>(workspace);
    const int32_t max_candidates = workspaceSizeFloats / (sizeof(OracleCandidate) / sizeof(float));
    
    int32_t step = 2;
    while (((pitch / step) * (height / step)) > max_candidates) step++;
    
    // 1. EXTRACT BLOCKS AND SEARCH FOR SPARSITY (VECTORIZED)
    int32_t num_candidates = 0;
    for (int32_t y = 0; y < height - 15; y += step)
	{ 
        for (int32_t x = 0; x < pitch - 15; x += step)
		{
            
            if (!Is_Valid_Block(plane, x, y, pitch)) continue;
            
            OracleCandidate& cand = candidates[num_candidates];
            cand.x = x; 
            cand.y = y;
            
            float sum = 0.0f;
            for(int32_t i = 0; i < 4; ++i)
			{
                for(int32_t j = 0; j < 4; ++j) sum += plane[(y + i) * pitch + (x + j)];
            }
            cand.mean = sum / 16.0f;
            
            // Execute AVX2 Forward DCT
            AVX2_Forward_DCT_4x4(&plane[y * pitch + x], pitch, cand.dct, D);
            
            cand.sd = 1e15f;
            for(int32_t sy = y + 4; sy < y + 14 && sy < height - 3; ++sy)
			{
                for(int32_t sx = x + 4; sx < x + 14 && sx < pitch - 3; ++sx)
				{
                    CACHE_ALIGN float cand_dct[16];
                    AVX2_Forward_DCT_4x4(&plane[sy * pitch + sx], pitch, cand_dct, D);
                    
                    // Execute AVX2 Sparse Distance
                    float sd = AVX2_Calculate_Sparse_Distance(cand.dct, cand_dct);
                    if (sd < cand.sd) cand.sd = sd;
                }
            }
            num_candidates++;
        }
    }
    
    // 2. BUILD THE HISTOGRAM (SCALAR)
    int32_t numBins = std::max(1, (pitch * height) / 42000);
    if (numBins > 64) numBins = 64; 

    std::sort(candidates, candidates + num_candidates, [](const OracleCandidate& a, const OracleCandidate& b) {
        return a.mean < b.mean;
    });

    CACHE_ALIGN BinInfo bins[64] = {};
    const float min_mean = candidates[0].mean;
    const float max_mean = candidates[num_candidates - 1].mean;
    const float bin_width = (max_mean - min_mean + 0.001f) / numBins;
    
    int32_t current_bin = 0;
    float current_boundary = min_mean + bin_width;
    bins[0].start_idx = 0;
    
    for(int32_t i = 0; i < num_candidates; ++i)
	{
        if (candidates[i].mean > current_boundary && current_bin < numBins - 1)
		{
            current_bin++;
            current_boundary += bin_width;
            bins[current_bin].start_idx = i;
        }
        bins[current_bin].count++;
    }

    // 3. STATISTICAL EVALUATION PER BIN (SCALAR)
    for(int32_t b = 0; b < numBins; ++b)
	{
        if (bins[b].count == 0) continue;
        
        std::sort(candidates + bins[b].start_idx, candidates + bins[b].start_idx + bins[b].count, [](const OracleCandidate& c1, const OracleCandidate& c2) {
            return c1.sd < c2.sd;
        });
        
        int32_t k_blocks = std::max(1, static_cast<int32_t>(bins[b].count * 0.005f));
        if (k_blocks > 4000) k_blocks = 4000; 
        
        CACHE_ALIGN float mad_temp[4000];
        
        for (int32_t k = 0; k < 16; ++k) {
            for (int32_t i = 0; i < k_blocks; ++i) {
                mad_temp[i] = candidates[bins[b].start_idx + i].dct[k];
            }
            float hat_std = Compute_MAD(mad_temp, k_blocks);
            bins[b].stds[k] = hat_std * 1.791106f; 
        }
        
        float b_mean = 0.0f;
        for (int32_t i = 0; i < k_blocks; ++i) b_mean += candidates[bins[b].start_idx + i].mean;
        bins[b].mean_val = b_mean / k_blocks;
    }

    // 4. CURVE SMOOTHING (SCALAR)
    for(int32_t k = 0; k < 16; ++k)
	{
        CACHE_ALIGN float temp_stds[64] = {0};
        for(int32_t b = 0; b < numBins; ++b)
		{
            if (bins[b].count == 0) continue;
            float sum = 0.0f; int32_t cnt = 0;
            for(int32_t j = std::max(0, b - 1); j <= std::min(numBins - 1, b + 1); ++j) {
                if (bins[j].count > 0) { sum += bins[j].stds[k]; cnt++; }
            }
            temp_stds[b] = sum / cnt;
        }
        for(int32_t b = 0; b < numBins; ++b) {
            if (bins[b].count > 0) bins[b].stds[k] = temp_stds[b];
        }
    }

    // 5. FREQUENCY TO SPATIAL COVARIANCE (SCALAR)
    CACHE_ALIGN float bin_covs[64][256] = {0};
    for(int32_t b = 0; b < numBins; ++b)
	{
        if (bins[b].count == 0) continue;
        for (int32_t p = 0; p < 16; ++p)
		{
            for (int32_t q = 0; q < 16; ++q)
			{
                float sum = 0.0f;
                for (int32_t k = 0; k < 16; ++k) {
                    sum += D[k][p] * D[k][q] * (bins[b].stds[k] * bins[b].stds[k]);
                }
                bin_covs[b][p * 16 + q] = sum;
            }
        }
    }

    // 6. INTERPOLATE COVARIANCE MATRICES TO 256-LEVEL LUT (SCALAR)
    for(int32_t intensity = 0; intensity < 256; ++intensity) {
        int32_t b_left = -1, b_right = -1;
        for(int32_t b = 0; b < numBins; ++b) {
            if (bins[b].count > 0) {
                if (bins[b].mean_val <= intensity) b_left = b;
                if (bins[b].mean_val >= intensity && b_right == -1) b_right = b;
            }
        }
        
        if (b_left == -1) b_left = b_right;
        if (b_right == -1) b_right = b_left;
        if (b_left == -1) continue; 
        
        float weight_right = 0.0f;
        if (b_left != b_right) {
            weight_right = (intensity - bins[b_left].mean_val) / (bins[b_right].mean_val - bins[b_left].mean_val);
        }
        
        for(int32_t i = 0; i < 256; ++i) {
            covLUT[intensity * 256 + i] = bin_covs[b_left][i] * (1.0f - weight_right) + bin_covs[b_right][i] * weight_right;
        }
    }
}

// =========================================================
// MAIN ENTRY POINT
// =========================================================
void AVX2_Estimate_Noise_Covariances
(
    const MemHandler& mem,
    const int32_t width, // Pitch/stride
    const int32_t height,
    const AlgoControls& algoCtrl
)
{
    const int32_t workspaceSizeFloats = width * height * 4;

    AVX2_Process_Channel_Oracle(mem.Y_planar, mem.NoiseCov_Y, width, height, mem.OracleWorkspace, workspaceSizeFloats);
    AVX2_Process_Channel_Oracle(mem.U_planar, mem.NoiseCov_U, width, height, mem.OracleWorkspace, workspaceSizeFloats);
    AVX2_Process_Channel_Oracle(mem.V_planar, mem.NoiseCov_V, width, height, mem.OracleWorkspace, workspaceSizeFloats);
}