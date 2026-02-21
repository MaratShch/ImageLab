#include <cmath>
#include <algorithm>
#include <cstring>
#include "AlgoBayesFilter.hpp"
#include "AlgoMatrixMath.hpp"
#include "AlgoBlockMatch.hpp"

// =========================================================
// HELPER: STEP 1 - SINGLE CHANNEL SHRINKAGE
// =========================================================
inline void Apply_Bayes_Shrinkage_Step1
(
    float* RESTRICT vol_noisy, 
    const int32_t num_patches, 
    const float* RESTRICT covLUT, 
    const float noise_mult
) noexcept
{
    constexpr int32_t K = 16; 
    CACHE_ALIGN double C_empirical[256];
    CACHE_ALIGN double C_noise[256];
    CACHE_ALIGN double C_diff[256];
    CACHE_ALIGN double Shrinkage[256];
    CACHE_ALIGN double mean_P[16] = {0.0};

    const double inv_N = 1.0 / static_cast<double>(num_patches);
    const double inv_Nm1 = 1.0 / static_cast<double>(num_patches - 1);

    // 1. Barycenter (Mean Patch)
    for (int32_t p = 0; p < num_patches; ++p)
    {
        for (int32_t i = 0; i < K; ++i)
        {
            mean_P[i] += static_cast<double>(vol_noisy[p * K + i]);
        }
    }
    for (int32_t i = 0; i < K; ++i) mean_P[i] *= inv_N;

    // 2. Inline Empirical Covariance (Directly from vol_noisy to save stack space)
    for (int32_t i = 0; i < K; ++i)
    {
        for (int32_t j = 0; j <= i; ++j)
        {
            double sum = 0.0;
            for (int32_t p = 0; p < num_patches; ++p)
            {
                sum += (static_cast<double>(vol_noisy[p * K + i]) - mean_P[i]) * (static_cast<double>(vol_noisy[p * K + j]) - mean_P[j]);
            }
            C_empirical[j * K + i] = C_empirical[i * K + j] = sum * inv_Nm1;
        }
    }

    // 3. Noise Covariance
    int32_t intensity = std::max(0, std::min(255, static_cast<int32_t>(mean_P[0]))); 
    for (int32_t i = 0; i < 256; ++i)
    {
        C_noise[i] = static_cast<double>(covLUT[intensity * 256 + i]) * noise_mult;
    }

    // 4. Trace Check & Homogeneous Trick
    double trace_diff = 0.0;
    for (int32_t i = 0; i < K; ++i)
    {
        double val = C_empirical[i * K + i] - C_noise[i * K + i];
        C_diff[i * K + i] = std::max(0.0, val);
        trace_diff += val;
    }
    for (int32_t i = 0; i < K; ++i)
    {
        for (int32_t j = 0; j < K; ++j)
        {
            if (i != j) C_diff[i * K + j] = C_empirical[i * K + j] - C_noise[i * K + j];
        }
    }

    if (trace_diff <= 0.0)
    {
        for (int32_t p = 0; p < num_patches; ++p)
        {
            for (int32_t i = 0; i < K; ++i) vol_noisy[p * K + i] = static_cast<float>(mean_P[i]);
        }
        return;
    }

    // 5. Shrinkage Math
    if (Inverse_Matrix(C_empirical, K))
    {
        Product_Matrix(Shrinkage, C_diff, C_empirical, K, K, K);
        
        for (int32_t p = 0; p < num_patches; ++p)
        {
            CACHE_ALIGN double out_patch[16];
            for (int32_t i = 0; i < K; ++i)
            {
                double sum = 0.0;
                for (int32_t j = 0; j < K; ++j)
                {
                    double centered_val = static_cast<double>(vol_noisy[p * K + j]) - mean_P[j];
                    sum += Shrinkage[i * K + j] * centered_val;
                }
                out_patch[i] = mean_P[i] + sum;
            }
            for (int32_t i = 0; i < K; ++i)
            {
                vol_noisy[p * K + i] = static_cast<float>(out_patch[i]);
            }
        }
    }
}

// =========================================================
// HELPER: STEP 2 - JOINT EMPIRICAL WIENER
// =========================================================
inline void Apply_Bayes_Shrinkage_Step2_Joint
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
    constexpr int32_t K = 48; // 3 channels * 16 pixels
    CACHE_ALIGN double C_basic[48 * 48];
    CACHE_ALIGN double C_sum[48 * 48];
    CACHE_ALIGN double Shrinkage[48 * 48];
    CACHE_ALIGN double barycenter[48];

    const double inv_N = 1.0 / static_cast<double>(num_patches);
    const double inv_Nm1 = 1.0 / static_cast<double>(num_patches - 1);

    // 1. Barycenter of Noisy Data
    for (int32_t k = 0; k < K; ++k)
    {
        double sum = 0.0;
        for (int32_t p = 0; p < num_patches; ++p)
        {
            sum += static_cast<double>(vol_noisy_joint[p * K + k]);
        }
        barycenter[k] = sum * inv_N;
    }

    // 2. Covariance of Basic Data (Pilot) centered around Noisy Barycenter
    for (int32_t i = 0; i < K; ++i)
    {
        for (int32_t j = 0; j <= i; ++j)
        {
            double sum = 0.0;
            for (int32_t p = 0; p < num_patches; ++p)
            {
                sum += (static_cast<double>(vol_basic_joint[p * K + i]) - barycenter[i]) * (static_cast<double>(vol_basic_joint[p * K + j]) - barycenter[j]);
            }
            C_basic[j * K + i] = C_basic[i * K + j] = sum * inv_Nm1;
        }
    }

    // 3. Add Block-Diagonal Noise Covariance (Cp + Cn)
    std::memcpy(C_sum, C_basic, K * K * sizeof(double));

    int32_t idxY = std::max(0, std::min(255, static_cast<int32_t>(barycenter[0])));
    int32_t idxU = std::max(0, std::min(255, static_cast<int32_t>(barycenter[16])));
    int32_t idxV = std::max(0, std::min(255, static_cast<int32_t>(barycenter[32])));

    for (int32_t i = 0; i < 16; ++i)
    {
        for (int32_t j = 0; j < 16; ++j)
        {
            C_sum[i * K + j]               += static_cast<double>(covLUT_Y[idxY * 256 + i * 16 + j]) * noise_mult;
            C_sum[(i + 16) * K + (j + 16)] += static_cast<double>(covLUT_U[idxU * 256 + i * 16 + j]) * noise_mult;
            C_sum[(i + 32) * K + (j + 32)] += static_cast<double>(covLUT_V[idxV * 256 + i * 16 + j]) * noise_mult;
        }
    }

    // 4. Shrinkage
    if (Inverse_Matrix(C_sum, K))
    {
        Product_Matrix(Shrinkage, C_basic, C_sum, K, K, K);

        for (int32_t p = 0; p < num_patches; ++p)
        {
           CACHE_ALIGN  double out_patch[48];
            for (int32_t i = 0; i < K; ++i)
            {
                double sum = 0.0;
                for (int32_t j = 0; j < K; ++j)
                {
                    double centered_noisy = static_cast<double>(vol_noisy_joint[p * K + j]) - barycenter[j];
                    sum += Shrinkage[i * K + j] * centered_noisy;
                }
                out_patch[i] = barycenter[i] + sum;
            }
            // Overwrite basic joint to pipe directly into aggregation
            for (int32_t i = 0; i < K; ++i)
            {
                vol_basic_joint[p * K + i] = static_cast<float>(out_patch[i]);
            }
        }
    }
}

// =========================================================
// AGGREGATION HELPERS
// =========================================================
inline void Aggregate_3D_Group
(
    const float* RESTRICT vol, 
    const int32_t num_patches, 
    const PatchDistance* RESTRICT pool,
    float* RESTRICT accum, 
    float* RESTRICT weight,
    const int32_t width
) noexcept
{
    for (int32_t p = 0; p < num_patches; ++p)
    {
        const int32_t px = pool[p].x;
        const int32_t py = pool[p].y;
        for (int32_t i = 0; i < 4; ++i)
        {
            for (int32_t j = 0; j < 4; ++j)
            {
                if (p == 0 && (i == 1 || i == 2) && (j == 1 || j == 2)) continue;
                
                const int32_t idx = (py + i) * width + (px + j);
                accum[idx] += vol[p * 16 + (i * 4 + j)];
                weight[idx] += 1.0f;
            }
        }
    }
}

inline void Aggregate_3D_Group_Payload
(
    const float* RESTRICT vol, 
    const int32_t num_patches, 
    const PatchDistance* RESTRICT pool,
    float* RESTRICT accum, 
    const int32_t width
) noexcept
{
    for (int32_t p = 0; p < num_patches; ++p)
	{
        const int32_t px = pool[p].x;
        const int32_t py = pool[p].y;
        for (int32_t i = 0; i < 4; ++i)
		{
            for (int32_t j = 0; j < 4; ++j)
			{
                if (p == 0 && (i == 1 || i == 2) && (j == 1 || j == 2)) continue;
                accum[(py + i) * width + (px + j)] += vol[p * 16 + (i * 4 + j)];
            }
        }
    }
}

inline void Aggregate_3D_Group_Joint
(
    const float* RESTRICT vol_joint, 
    const int32_t num_patches, 
    const PatchDistance* RESTRICT pool,
    float* RESTRICT accum_Y,
	float* RESTRICT accum_U,
	float* RESTRICT accum_V,
    float* RESTRICT weight,
	const int32_t width
) noexcept
{
    for (int32_t p = 0; p < num_patches; ++p)
	{
        const int32_t px = pool[p].x;
        const int32_t py = pool[p].y;
        for (int32_t i = 0; i < 4; ++i)
		{
            for (int32_t j = 0; j < 4; ++j)
			{
                if (p == 0 && (i == 1 || i == 2) && (j == 1 || j == 2)) continue;
                
                const int32_t idx = (py + i) * width + (px + j);
                const int32_t offset = p * 48 + (i * 4 + j);
                
                accum_Y[idx] += vol_joint[offset];
                accum_U[idx] += vol_joint[offset + 16];
                accum_V[idx] += vol_joint[offset + 32];
                weight[idx] += 1.0f;
            }
        }
    }
}

// =========================================================
// MAIN API: DUAL-PASS BAYES FILTER
// =========================================================
void Process_Scale_NL_Bayes
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
    
    // Repurpose Oracle Workspace for Pilot 
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
    
    // Memory mapping for 3D workspaces
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
            
            int32_t N = Extract_Similar_Patches
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

            Apply_Bayes_Shrinkage_Step1(vol_Y, N, mem.NoiseCov_Y, noise_variance_multiplier);
            Apply_Bayes_Shrinkage_Step1(vol_U, N, mem.NoiseCov_U, noise_variance_multiplier);
            Apply_Bayes_Shrinkage_Step1(vol_V, N, mem.NoiseCov_V, noise_variance_multiplier);

            // Safe Accumulation
            Aggregate_3D_Group(vol_Y, N, mem.SearchPool, Pilot_Y, Pilot_W, width);
            Aggregate_3D_Group_Payload(vol_U, N, mem.SearchPool, Pilot_U, width);
            Aggregate_3D_Group_Payload(vol_V, N, mem.SearchPool, Pilot_V, width);
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
            
            // Search based on the CLEAN Pilot image
            int32_t N = Extract_Similar_Patches
			(
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

            Apply_Bayes_Shrinkage_Step2_Joint
			(
                vol_noisy_joint, vol_basic_joint, N, 
                mem.NoiseCov_Y, mem.NoiseCov_U, mem.NoiseCov_V, 
                noise_variance_multiplier
            );

            Aggregate_3D_Group_Joint
			(
                vol_basic_joint, N, mem.SearchPool, 
                mem.Accum_Y, mem.Accum_U, mem.Accum_V, mem.Weight_Count, width
            );
        }
    }

    // Normalize Final Accumulators
    for (int32_t i = 0; i < frameSize; ++i)
	{
        if (mem.Weight_Count[i] > 0.0f)
		{
            mem.Accum_Y[i] /= mem.Weight_Count[i];
            mem.Accum_U[i] /= mem.Weight_Count[i];
            mem.Accum_V[i] /= mem.Weight_Count[i];
        } else {
            mem.Accum_Y[i] = Y_scale[i];
            mem.Accum_U[i] = U_scale[i];
            mem.Accum_V[i] = V_scale[i];
        }
    }
	
	return;
}