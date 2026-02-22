#include <cmath>
#include <algorithm>
#include <cstring>
#include <immintrin.h>
#include "AVX2_AlgoBayesFilter.hpp"
#include "AVX2_AlgoMatrixMath.hpp"
#include "AVX2_AlgoBlockMatch.hpp"

// =========================================================
// FAST HORIZONTAL SUM HELPER (DOUBLE PRECISION)
// =========================================================
inline double AVX2_HSum_PD(__m256d v) noexcept
{
    __m128d vlow  = _mm256_castpd256_pd128(v);
    __m128d vhigh = _mm256_extractf128_pd(v, 1);
    vlow = _mm_add_pd(vlow, vhigh);
    __m128d shuf = _mm_unpackhi_pd(vlow, vlow);
    vlow = _mm_add_pd(vlow, shuf);
    return _mm_cvtsd_f64(vlow);
}

// =========================================================
// HELPER: STEP 1 - SINGLE CHANNEL SHRINKAGE (AVX2)
// =========================================================
inline void AVX2_Apply_Bayes_Shrinkage_Step1
(
    float* RESTRICT vol_noisy, 
    const int32_t num_patches, 
    const float* RESTRICT covLUT, 
    const float noise_mult
) noexcept
{
    constexpr int32_t K = 16; 
    double C_empirical[256];
    double C_noise[256];
    double C_diff[256];
    double Shrinkage[256];
    double mean_P[16];

    const double inv_N = 1.0 / static_cast<double>(num_patches);
    const double inv_Nm1 = 1.0 / static_cast<double>(num_patches - 1);

    __m256d vMean[4] = { _mm256_setzero_pd(), _mm256_setzero_pd(), _mm256_setzero_pd(), _mm256_setzero_pd() };
    for (int32_t p = 0; p < num_patches; ++p) {
        for(int32_t i = 0; i < K; i += 4) {
            __m128 f = _mm_loadu_ps(&vol_noisy[p * K + i]);
            vMean[i/4] = _mm256_add_pd(vMean[i/4], _mm256_cvtps_pd(f));
        }
    }
    __m256d vInvN = _mm256_set1_pd(inv_N);
    for (int32_t i = 0; i < 4; ++i) {
        _mm256_storeu_pd(&mean_P[i * 4], _mm256_mul_pd(vMean[i], vInvN));
    }

    std::memset(C_empirical, 0, sizeof(C_empirical));
    for (int32_t p = 0; p < num_patches; ++p) {
        double temp[16];
        for (int32_t k = 0; k < K; ++k) temp[k] = static_cast<double>(vol_noisy[p * K + k]) - mean_P[k];
        
        for (int32_t i = 0; i < K; ++i) {
            __m256d vA = _mm256_set1_pd(temp[i]);
            int32_t j = 0;
            for (; j <= i - 3; j += 4) {
                __m256d vB = _mm256_loadu_pd(&temp[j]);
                __m256d vC = _mm256_loadu_pd(&C_empirical[i * K + j]);
                _mm256_storeu_pd(&C_empirical[i * K + j], _mm256_fmadd_pd(vA, vB, vC));
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
        C_noise[i] = static_cast<double>(covLUT[intensity * 256 + i]) * noise_mult;
    }

    double trace_diff = 0.0;
    for (int32_t i = 0; i < K; ++i) {
        double val = C_empirical[i * K + i] - C_noise[i * K + i];
        C_diff[i * K + i] = std::max(0.0, val);
        trace_diff += val;
    }
    for (int32_t i = 0; i < K; ++i) {
        for (int32_t j = 0; j < K; ++j) {
            if (i != j) C_diff[i * K + j] = C_empirical[i * K + j] - C_noise[i * K + j];
        }
    }

    if (trace_diff <= 0.0) {
        for (int32_t p = 0; p < num_patches; ++p) {
            for (int32_t i = 0; i < K; ++i) vol_noisy[p * K + i] = static_cast<float>(mean_P[i]);
        }
        return;
    }

    if (AVX2_Inverse_Matrix(C_empirical, K)) {
        AVX2_Product_Matrix(Shrinkage, C_diff, C_empirical, K, K, K);
        
        for (int32_t p = 0; p < num_patches; ++p) {
            double out_patch[16];
            for (int32_t i = 0; i < K; ++i) {
                __m256d vSum = _mm256_setzero_pd();
                for (int32_t j = 0; j < K; j += 4) {
                    __m256d w = _mm256_loadu_pd(&Shrinkage[i * K + j]);
                    __m128 f = _mm_loadu_ps(&vol_noisy[p * K + j]);
                    __m256d d = _mm256_cvtps_pd(f);
                    __m256d b = _mm256_loadu_pd(&mean_P[j]);
                    __m256d c = _mm256_sub_pd(d, b);
                    vSum = _mm256_fmadd_pd(w, c, vSum);
                }
                out_patch[i] = mean_P[i] + AVX2_HSum_PD(vSum);
            }
            for (int32_t i = 0; i < K; ++i) {
                vol_noisy[p * K + i] = static_cast<float>(out_patch[i]);
            }
        }
    }
}

// =========================================================
// HELPER: STEP 2 - JOINT EMPIRICAL WIENER (AVX2)
// =========================================================
inline void AVX2_Apply_Bayes_Shrinkage_Step2_Joint
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
    constexpr int32_t K = 48; 
    double C_basic[48 * 48];
    double C_sum[48 * 48];
    double Shrinkage[48 * 48];
    double barycenter[48];

    const double inv_N = 1.0 / static_cast<double>(num_patches);
    const double inv_Nm1 = 1.0 / static_cast<double>(num_patches - 1);

    __m256d vBary[12];
    for (int32_t i = 0; i < 12; ++i) vBary[i] = _mm256_setzero_pd();

    for (int32_t p = 0; p < num_patches; ++p) {
        for(int32_t k = 0; k < K; k += 4) {
            __m128 f = _mm_loadu_ps(&vol_noisy_joint[p * K + k]);
            vBary[k/4] = _mm256_add_pd(vBary[k/4], _mm256_cvtps_pd(f));
        }
    }
    __m256d vInvN = _mm256_set1_pd(inv_N);
    for (int32_t k = 0; k < 12; ++k) {
        _mm256_storeu_pd(&barycenter[k * 4], _mm256_mul_pd(vBary[k], vInvN));
    }

    std::memset(C_basic, 0, sizeof(C_basic));
    for (int32_t p = 0; p < num_patches; ++p) {
        double temp[48];
        for (int32_t k = 0; k < K; ++k) temp[k] = static_cast<double>(vol_basic_joint[p * K + k]) - barycenter[k];
        
        for (int32_t i = 0; i < K; ++i) {
            __m256d vA = _mm256_set1_pd(temp[i]);
            int32_t j = 0;
            for (; j <= i - 3; j += 4) {
                __m256d vB = _mm256_loadu_pd(&temp[j]);
                __m256d vC = _mm256_loadu_pd(&C_basic[i * K + j]);
                _mm256_storeu_pd(&C_basic[i * K + j], _mm256_fmadd_pd(vA, vB, vC));
            }
            for (; j <= i; ++j) {
                C_basic[i * K + j] += temp[i] * temp[j];
            }
        }
    }
    for (int32_t i = 0; i < K; ++i) {
        for (int32_t j = 0; j <= i; ++j) {
            C_basic[j * K + i] = (C_basic[i * K + j] *= inv_Nm1); 
        }
    }

    std::memcpy(C_sum, C_basic, K * K * sizeof(double));

    int32_t idxY = std::max(0, std::min(255, static_cast<int32_t>(barycenter[0])));
    int32_t idxU = std::max(0, std::min(255, static_cast<int32_t>(barycenter[16])));
    int32_t idxV = std::max(0, std::min(255, static_cast<int32_t>(barycenter[32])));

    for (int32_t i = 0; i < 16; ++i) {
        for (int32_t j = 0; j < 16; ++j) {
            C_sum[i * K + j]               += static_cast<double>(covLUT_Y[idxY * 256 + i * 16 + j]) * noise_mult;
            C_sum[(i + 16) * K + (j + 16)] += static_cast<double>(covLUT_U[idxU * 256 + i * 16 + j]) * noise_mult;
            C_sum[(i + 32) * K + (j + 32)] += static_cast<double>(covLUT_V[idxV * 256 + i * 16 + j]) * noise_mult;
        }
    }

    if (AVX2_Inverse_Matrix(C_sum, K)) {
        AVX2_Product_Matrix(Shrinkage, C_basic, C_sum, K, K, K);

        for (int32_t p = 0; p < num_patches; ++p) {
            double out_patch[48];
            for (int32_t i = 0; i < K; ++i) {
                __m256d vSum = _mm256_setzero_pd();
                for (int32_t j = 0; j < K; j += 4) {
                    __m256d w = _mm256_loadu_pd(&Shrinkage[i * K + j]);
                    __m128 f = _mm_loadu_ps(&vol_noisy_joint[p * K + j]);
                    __m256d d = _mm256_cvtps_pd(f);
                    __m256d b = _mm256_loadu_pd(&barycenter[j]);
                    __m256d c = _mm256_sub_pd(d, b);
                    vSum = _mm256_fmadd_pd(w, c, vSum);
                }
                out_patch[i] = barycenter[i] + AVX2_HSum_PD(vSum);
            }
            for (int32_t i = 0; i < K; ++i) {
                vol_basic_joint[p * K + i] = static_cast<float>(out_patch[i]);
            }
        }
    }
}

// =========================================================
// VECTORIZED AGGREGATION HELPERS
// =========================================================
inline void AVX2_Aggregate_3D_Group
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

inline void AVX2_Aggregate_3D_Group_Payload
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

inline void AVX2_Aggregate_3D_Group_Joint
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
// MAIN API: DUAL-PASS BAYES FILTER (AVX2)
// =========================================================
void AVX2_Process_Scale_NL_Bayes
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
            
            int32_t N = AVX2_Extract_Similar_Patches
            (
                Y_scale, U_scale, V_scale, width, height, x, y, 
                tau_step1, mem.SearchPool
            );
            
            if (N < 16) continue;

            for (int32_t p = 0; p < N; ++p) {
                int32_t px = mem.SearchPool[p].x;
                int32_t py = mem.SearchPool[p].y;
                for (int32_t i = 0; i < 4; ++i) {
                    for (int32_t j = 0; j < 4; ++j) {
                        const int32_t src_idx = (py + i) * width + px + j;
                        const int32_t dst_idx = p * 16 + (i * 4 + j);
                        vol_Y[dst_idx] = Y_scale[src_idx];
                        vol_U[dst_idx] = U_scale[src_idx];
                        vol_V[dst_idx] = V_scale[src_idx];
                    }
                }
            }

            AVX2_Apply_Bayes_Shrinkage_Step1(vol_Y, N, mem.NoiseCov_Y, noise_variance_multiplier);
            AVX2_Apply_Bayes_Shrinkage_Step1(vol_U, N, mem.NoiseCov_U, noise_variance_multiplier);
            AVX2_Apply_Bayes_Shrinkage_Step1(vol_V, N, mem.NoiseCov_V, noise_variance_multiplier);

            AVX2_Aggregate_3D_Group(vol_Y, N, mem.SearchPool, Pilot_Y, Pilot_W, width);
            AVX2_Aggregate_3D_Group_Payload(vol_U, N, mem.SearchPool, Pilot_U, width);
            AVX2_Aggregate_3D_Group_Payload(vol_V, N, mem.SearchPool, Pilot_V, width);
        }
    }

    // Normalize Pilot
    for (int32_t i = 0; i < frameSize; ++i)
    {
        if (Pilot_W[i] > 0.0f) {
            Pilot_Y[i] /= Pilot_W[i];
            Pilot_U[i] /= Pilot_W[i];
            Pilot_V[i] /= Pilot_W[i];
        } else {
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

    for (int32_t y = 0; y < height - 4; y += 2) {
        for (int32_t x = 0; x < width - 4; x += 2) {
            
            int32_t N = AVX2_Extract_Similar_Patches(
                Pilot_Y, Pilot_U, Pilot_V, width, height, x, y, 
                tau_step2, mem.SearchPool
            );
            
            if (N < 16) continue;

            for (int32_t p = 0; p < N; ++p) {
                int32_t px = mem.SearchPool[p].x;
                int32_t py = mem.SearchPool[p].y;
                for (int32_t i = 0; i < 4; ++i) {
                    for (int32_t j = 0; j < 4; ++j) {
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

            AVX2_Apply_Bayes_Shrinkage_Step2_Joint(
                vol_noisy_joint, vol_basic_joint, N, 
                mem.NoiseCov_Y, mem.NoiseCov_U, mem.NoiseCov_V, 
                noise_variance_multiplier
            );

            AVX2_Aggregate_3D_Group_Joint(
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