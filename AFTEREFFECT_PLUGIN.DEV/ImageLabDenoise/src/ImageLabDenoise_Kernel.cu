#include <cstdio>
#include <algorithm>
#include "CUDA/CudaMemHandler.cuh"
#include "ImageLabDenoise_GPU.hpp"

#define MAX_PATCHES 8 // Force 8 for ultra-fast register locking in the hardware

// =========================================================
// HARDWARE ATOMIC HELPER
// =========================================================
__device__ __forceinline__ void atomicMinFloat(float* addr, float value)
{
    int* addr_as_int = (int*)addr;
    int old = *addr_as_int, assumed;
    do {
        assumed = old;
        old = atomicCAS(addr_as_int, assumed,
            __float_as_int(fminf(value, __int_as_float(assumed))));
    } while (assumed != old);
}

// =========================================================
// MEMORY MANAGEMENT
// =========================================================
inline bool alloc_cuda_memory_buffers(CudaMemHandler& mem, int32_t target_tile_width, int32_t target_tile_height)
{
    mem.tileW = target_tile_width;
    mem.tileH = target_tile_height;
    mem.padW = target_tile_width + 32;
    mem.padH = target_tile_height + 32;
    mem.frameSizePadded = mem.padW * mem.padH;

    const size_t bytes_full = mem.frameSizePadded * sizeof(float);
    const size_t bytes_half = (mem.padW / 2) * (mem.padH / 2) * sizeof(float);
    const size_t bytes_quart = (mem.padW / 4) * (mem.padH / 4) * sizeof(float);
    const size_t bytes_lut = 256 * 256 * sizeof(float);
    const size_t bytes_pool = mem.frameSizePadded * 128 * sizeof(int2_coord); // Still sized safely

    CUDA_CHECK(cudaMalloc((void**)&mem.d_Y_planar, bytes_full));
    CUDA_CHECK(cudaMalloc((void**)&mem.d_U_planar, bytes_full));
    CUDA_CHECK(cudaMalloc((void**)&mem.d_V_planar, bytes_full));

    CUDA_CHECK(cudaMalloc((void**)&mem.d_Y_half, bytes_half));
    CUDA_CHECK(cudaMalloc((void**)&mem.d_U_half, bytes_half));
    CUDA_CHECK(cudaMalloc((void**)&mem.d_V_half, bytes_half));

    CUDA_CHECK(cudaMalloc((void**)&mem.d_Y_quart, bytes_quart));
    CUDA_CHECK(cudaMalloc((void**)&mem.d_U_quart, bytes_quart));
    CUDA_CHECK(cudaMalloc((void**)&mem.d_V_quart, bytes_quart));

    CUDA_CHECK(cudaMalloc((void**)&mem.d_NoiseCov_Y, bytes_lut));
    CUDA_CHECK(cudaMalloc((void**)&mem.d_NoiseCov_U, bytes_lut));
    CUDA_CHECK(cudaMalloc((void**)&mem.d_NoiseCov_V, bytes_lut));

    CUDA_CHECK(cudaMalloc((void**)&mem.d_Pilot_Y, bytes_full));
    CUDA_CHECK(cudaMalloc((void**)&mem.d_Pilot_U, bytes_full));
    CUDA_CHECK(cudaMalloc((void**)&mem.d_Pilot_V, bytes_full));

    CUDA_CHECK(cudaMalloc((void**)&mem.d_Accum_Y, bytes_full));
    CUDA_CHECK(cudaMalloc((void**)&mem.d_Accum_U, bytes_full));
    CUDA_CHECK(cudaMalloc((void**)&mem.d_Accum_V, bytes_full));
    CUDA_CHECK(cudaMalloc((void**)&mem.d_Weight_Count, bytes_full));

    CUDA_CHECK(cudaMalloc((void**)&mem.d_SearchPool, bytes_pool));

    return true;
}

inline void free_cuda_memory_buffers(CudaMemHandler& mem)
{
    cudaFree(mem.d_Y_planar);
    cudaFree(mem.d_U_planar);
    cudaFree(mem.d_V_planar);

    cudaFree(mem.d_Y_half);
    cudaFree(mem.d_U_half);
    cudaFree(mem.d_V_half);

    cudaFree(mem.d_Y_quart);
    cudaFree(mem.d_U_quart);
    cudaFree(mem.d_V_quart);

    cudaFree(mem.d_NoiseCov_Y);
    cudaFree(mem.d_NoiseCov_U);
    cudaFree(mem.d_NoiseCov_V);

    cudaFree(mem.d_Pilot_Y);
    cudaFree(mem.d_Pilot_U);
    cudaFree(mem.d_Pilot_V);

    cudaFree(mem.d_Accum_Y);
    cudaFree(mem.d_Accum_U);
    cudaFree(mem.d_Accum_V);
    cudaFree(mem.d_Weight_Count);

    cudaFree(mem.d_SearchPool);
}

// =========================================================
// KERNELS: COLOR CONVERSION
// =========================================================
__global__ void Kernel_Convert_BGRA_32f_YUV
(
    const float4* RESTRICT pInput,
    float* RESTRICT d_Y, float* RESTRICT d_U, float* RESTRICT d_V,
    int width, int height, int srcPitchPixels, int dstPitchPixels
)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height)
    {
        float4 pixel = pInput[y * srcPitchPixels + x];
        float b = pixel.x * 255.0f;
        float g = pixel.y * 255.0f;
        float r = pixel.z * 255.0f;

        d_Y[y * dstPitchPixels + x] = (r + g + b) * 0.57735027f;
        d_U[y * dstPitchPixels + x] = (r - b) * 0.70710678f;
        d_V[y * dstPitchPixels + x] = (r + b - 2.0f * g) * 0.40824829f;
    }
}

__global__ void Kernel_Convert_YUV_to_BGRA_32f
(
    const float* RESTRICT d_Y, const float* RESTRICT d_U, const float* RESTRICT d_V,
    const float4* RESTRICT pInputAlpha, float4* RESTRICT pOutput,
    int width, int height, int srcPitchPixels, int dstPitchPixels
)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height)
    {
        int src_idx = y * srcPitchPixels + x;
        float y_in = d_Y[src_idx];
        float u_in = d_U[src_idx];
        float v_in = d_V[src_idx];

        float r = y_in * 0.57735027f + u_in * 0.70710678f + v_in * 0.40824829f;
        float g = y_in * 0.57735027f - v_in * 0.81649658f;
        float b = y_in * 0.57735027f - u_in * 0.70710678f + v_in * 0.40824829f;

        float r_out = r * 0.00392156862f;
        float g_out = g * 0.00392156862f;
        float b_out = b * 0.00392156862f;

        int dst_idx = y * dstPitchPixels + x;
        float original_alpha = pInputAlpha[dst_idx].w;

        pOutput[dst_idx] = make_float4(b_out, g_out, r_out, original_alpha);
    }
}

// =========================================================
// KERNELS: CORE ALGORITHM (PYRAMID, ORACLE, MATCHING, BAYES)
// =========================================================
__global__ void Kernel_Downsample_Laplacian
(
    const float* RESTRICT src_plane, float* RESTRICT dst_plane,
    int src_width, int src_height, int dst_width, int dst_height,
    int src_pitch, int dst_pitch
)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < dst_width && y < dst_height)
    {
        int src_x = x * 2;
        int src_y = y * 2;
        int sx_1 = min(src_x + 1, src_width - 1);
        int sy_1 = min(src_y + 1, src_height - 1);

        float p00 = src_plane[src_y * src_pitch + src_x];
        float p10 = src_plane[src_y * src_pitch + sx_1];
        float p01 = src_plane[sy_1 * src_pitch + src_x];
        float p11 = src_plane[sy_1 * src_pitch + sx_1];

        dst_plane[y * dst_pitch + x] = (p00 + p10 + p01 + p11) * 0.25f;
    }
}

__global__ void Kernel_Oracle_Init(float* lut_Y, float* lut_U, float* lut_V)
{
    int i = threadIdx.x;
    if (i < 256) {
        lut_Y[i] = 1e30f;
        lut_U[i] = 1e30f;
        lut_V[i] = 1e30f;
    }
}

__global__ void Kernel_Oracle_Gather
(
    const float* __restrict__ src_plane, float* __restrict__ lut_cov,
    int width, int height, int pitch
)
{
    int bx = (blockIdx.x * blockDim.x + threadIdx.x) * 8;
    int by = (blockIdx.y * blockDim.y + threadIdx.y) * 8;

    if (bx + 7 < width && by + 7 < height)
    {
        float mean = 0.0f;
        float sq_mean = 0.0f;

        for (int y = 0; y < 8; ++y) {
            for (int x = 0; x < 8; ++x) {
                float val = src_plane[(by + y) * pitch + (bx + x)];
                mean += val;
                sq_mean += val * val;
            }
        }

        mean *= 0.015625f;
        sq_mean *= 0.015625f;
        float variance = fmaxf(0.0f, sq_mean - (mean * mean));
        int bin = min(255, max(0, (int)(mean + 0.5f)));

        atomicMinFloat(&lut_cov[bin], variance);
    }
}

__global__ void Kernel_Oracle_Finalize
(
    float* lut_Y, float* lut_U, float* lut_V,
    float master_amount, float luma_strength, float chroma_strength
)
{
    if (threadIdx.x == 0 && blockIdx.x == 0)
    {
        auto process_lut = [&](float* lut, float strength)
        {
            float last_valid = 0.0f;
            for (int i = 0; i < 256; ++i) {
                if (lut[i] > 1e10f) lut[i] = last_valid;
                else last_valid = lut[i];
            }
            for (int i = 254; i >= 0; --i) {
                if (lut[i] == 0.0f) lut[i] = lut[i + 1];
            }
            float global_scale = master_amount * strength;
            for (int i = 0; i < 256; ++i) {
                lut[i] *= global_scale;
            }
        };

        process_lut(lut_Y, luma_strength);
        process_lut(lut_U, chroma_strength);
        process_lut(lut_V, chroma_strength);
    }
}

// REGISTER OPTIMIZED SEARCH KERNEL
__global__ void Kernel_L2_BlockMatching
(
    cudaTextureObject_t tex_Y, int2_coord* searchPool,
    int width, int height
)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;

    // These will now safely stay in ultra-fast L1 Registers
    float best_ssds[MAX_PATCHES];
    int2_coord best_coords[MAX_PATCHES];

#pragma unroll
    for (int i = 0; i < MAX_PATCHES; ++i) {
        best_ssds[i] = 1e30f;
        best_coords[i] = { x, y };
    }

    for (int dy = -8; dy <= 7; ++dy)
    {
        for (int dx = -8; dx <= 7; ++dx)
        {
            int cand_x = x + dx;
            int cand_y = y + dy;

            float ssd = 0.0f;
            for (int r = 0; r < 8; ++r) {
                for (int c = 0; c < 8; ++c) {
                    float pRef = tex2D<float>(tex_Y, x + c, y + r);
                    float pCand = tex2D<float>(tex_Y, cand_x + c, cand_y + r);
                    float diff = pRef - pCand;
                    ssd += diff * diff;
                }
            }

            // Register-safe insertion sort
            if (ssd < best_ssds[MAX_PATCHES - 1])
            {
                int insert_pos = MAX_PATCHES - 1;
#pragma unroll
                for (int i = MAX_PATCHES - 1; i > 0; --i) {
                    if (ssd < best_ssds[i - 1]) insert_pos = i - 1;
                }

#pragma unroll
                for (int i = MAX_PATCHES - 1; i > 0; --i) {
                    if (i > insert_pos) {
                        best_ssds[i] = best_ssds[i - 1];
                        best_coords[i] = best_coords[i - 1];
                    }
                }
                best_ssds[insert_pos] = ssd;
                best_coords[insert_pos] = { cand_x, cand_y };
            }
        }
    }

    int pool_offset = (y * width + x) * MAX_PATCHES;
#pragma unroll
    for (int i = 0; i < MAX_PATCHES; ++i) {
        searchPool[pool_offset + i] = best_coords[i];
    }
}

// STRIDED BAYES KERNEL
__global__ void Kernel_CollaborativeBayes_Aggregate
(
    const float* __restrict__ d_Y, const int2_coord* __restrict__ pool,
    const float* __restrict__ oracle_lut, float* __restrict__ accum_Y,
    float* __restrict__ accum_weight, int width, int height, int bayes_step
)
{
    // Apply stride to skip reference pixels and reduce atomic locks
    int ref_x = blockIdx.x * bayes_step;
    int ref_y = blockIdx.y * bayes_step;
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int tid = ty * 8 + tx;

    if (ref_x >= width || ref_y >= height) return;

    extern __shared__ float s_group[];
    int pool_offset = (ref_y * width + ref_x) * MAX_PATCHES;

    float empirical_mean = 0.0f;
    for (int k = 0; k < MAX_PATCHES; ++k)
    {
        int2_coord cand = pool[pool_offset + k];
        int cx = min(max(cand.x + tx, 0), width - 1);
        int cy = min(max(cand.y + ty, 0), height - 1);

        float val = d_Y[cy * width + cx];
        s_group[k * 64 + tid] = val;
        empirical_mean += val;
    }
    empirical_mean /= (float)MAX_PATCHES;

    float empirical_var = 0.0f;
    for (int k = 0; k < MAX_PATCHES; ++k)
    {
        float diff = s_group[k * 64 + tid] - empirical_mean;
        empirical_var += diff * diff;
    }
    empirical_var = fmaxf(0.0f, empirical_var / (float)(MAX_PATCHES - 1));

    int bin = min(255, max(0, (int)(empirical_mean + 0.5f)));
    float noise_var = oracle_lut[bin];

    float signal_var = fmaxf(0.0f, empirical_var - noise_var);
    float shrinkage_weight = signal_var / (signal_var + noise_var + 1e-6f);

    for (int k = 0; k < MAX_PATCHES; ++k)
    {
        int2_coord cand = pool[pool_offset + k];
        int cx = cand.x + tx;
        int cy = cand.y + ty;

        if (cx >= 0 && cx < width && cy >= 0 && cy < height)
        {
            float noisy_val = s_group[k * 64 + tid];
            float denoised_val = empirical_mean + shrinkage_weight * (noisy_val - empirical_mean);
            int global_idx = cy * width + cx;

            atomicAdd(&accum_Y[global_idx], denoised_val);
            atomicAdd(&accum_weight[global_idx], 1.0f);
        }
    }
}

__global__ void Kernel_Normalize_Accumulators
(
    float* __restrict__ accum_Y, float* __restrict__ accum_weight,
    int width, int height
)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height)
    {
        int idx = y * width + x;
        float weight = accum_weight[idx];
        if (weight > 0.0f)
        {
            accum_Y[idx] /= weight;
        }
    }
}

// =========================================================
// MAIN GPU DISPATCHER (Extern C prevents LNK4042)
// =========================================================
CUDA_KERNEL_CALL
void ImageLabDenoise_CUDA
(
    const float* RESTRICT inBuffer,
    float* RESTRICT outBuffer,
    int srcPitch,
    int dstPitch,
    int width,
    int height,
    const AlgoControls* RESTRICT algoGpuParams,
    int frameCount,
    cudaStream_t stream
)
{
    const ProcAccuracy accuracy = algoGpuParams->accuracy;
    const float master_denoise_amount = algoGpuParams->master_denoise_amount;
    const float luma_strength = algoGpuParams->luma_strength;
    const float chroma_strength = algoGpuParams->chroma_strength;
    const float fine_detail_preservation = algoGpuParams->fine_detail_preservation;
    const float coarse_noise_reduction = algoGpuParams->coarse_noise_reduction;

    // --- 1. STATIC MEMORY CACHE (No more stuttering!) ---
    static CudaMemHandler gpuMem{};
    static bool isAllocated = false;
    static int allocW = 0, allocH = 0;

    if (!isAllocated || allocW != width || allocH != height)
    {
        if (isAllocated) free_cuda_memory_buffers(gpuMem);
        if (!alloc_cuda_memory_buffers(gpuMem, width, height)) return;
        isAllocated = true;
        allocW = width;
        allocH = height;
    }

    dim3 threadsPerBlock(32, 16);
    dim3 blocksPerGrid((width + 31) / 32, (height + 15) / 16);

    // --- 2. FORWARD COLOR CONVERSION ---
    Kernel_Convert_BGRA_32f_YUV << <blocksPerGrid, threadsPerBlock, 0, stream >> >
        (
            reinterpret_cast<const float4*>(inBuffer),
            gpuMem.d_Y_planar, gpuMem.d_U_planar, gpuMem.d_V_planar,
            width, height, srcPitch, width
            );

    // --- 3. SPATIAL DOWNSAMPLING (PYRAMID) ---
    const int half_W = width / 2;
    const int half_H = height / 2;
    dim3 blocksHalf((half_W + 31) / 32, (half_H + 15) / 16);

    Kernel_Downsample_Laplacian << <blocksHalf, threadsPerBlock, 0, stream >> >
        (gpuMem.d_Y_planar, gpuMem.d_Y_half, width, height, half_W, half_H, width, half_W);
    Kernel_Downsample_Laplacian << <blocksHalf, threadsPerBlock, 0, stream >> >
        (gpuMem.d_U_planar, gpuMem.d_U_half, width, height, half_W, half_H, width, half_W);
    Kernel_Downsample_Laplacian << <blocksHalf, threadsPerBlock, 0, stream >> >
        (gpuMem.d_V_planar, gpuMem.d_V_half, width, height, half_W, half_H, width, half_W);

    const int quart_W = half_W / 2;
    const int quart_H = half_H / 2;
    dim3 blocksQuart((quart_W + 31) / 32, (quart_H + 15) / 16);

    Kernel_Downsample_Laplacian << <blocksQuart, threadsPerBlock, 0, stream >> >
        (gpuMem.d_Y_half, gpuMem.d_Y_quart, half_W, half_H, quart_W, quart_H, half_W, quart_W);
    Kernel_Downsample_Laplacian << <blocksQuart, threadsPerBlock, 0, stream >> >
        (gpuMem.d_U_half, gpuMem.d_U_quart, half_W, half_H, quart_W, quart_H, half_W, quart_W);
    Kernel_Downsample_Laplacian << <blocksQuart, threadsPerBlock, 0, stream >> >
        (gpuMem.d_V_half, gpuMem.d_V_quart, half_W, half_H, quart_W, quart_H, half_W, quart_W);

    // --- 4. BLIND NOISE ORACLE ---
    Kernel_Oracle_Init << <1, 256, 0, stream >> > (gpuMem.d_NoiseCov_Y, gpuMem.d_NoiseCov_U, gpuMem.d_NoiseCov_V);

    int oracle_grid_W = width / 8;
    int oracle_grid_H = height / 8;
    dim3 oracleBlocks((oracle_grid_W + 31) / 32, (oracle_grid_H + 15) / 16);

    Kernel_Oracle_Gather << <oracleBlocks, threadsPerBlock, 0, stream >> >
        (gpuMem.d_Y_planar, gpuMem.d_NoiseCov_Y, width, height, width);
    Kernel_Oracle_Gather << <oracleBlocks, threadsPerBlock, 0, stream >> >
        (gpuMem.d_U_planar, gpuMem.d_NoiseCov_U, width, height, width);
    Kernel_Oracle_Gather << <oracleBlocks, threadsPerBlock, 0, stream >> >
        (gpuMem.d_V_planar, gpuMem.d_NoiseCov_V, width, height, width);

    Kernel_Oracle_Finalize << <1, 1, 0, stream >> >
        (
            gpuMem.d_NoiseCov_Y, gpuMem.d_NoiseCov_U, gpuMem.d_NoiseCov_V,
            master_denoise_amount, luma_strength, chroma_strength
            );

    // --- 5. L2 BLOCK MATCHING ---
    cudaResourceDesc resDesc = {};
    resDesc.resType = cudaResourceTypePitch2D;
    resDesc.res.pitch2D.devPtr = gpuMem.d_Y_planar;
    resDesc.res.pitch2D.width = width;
    resDesc.res.pitch2D.height = height;
    resDesc.res.pitch2D.pitchInBytes = width * sizeof(float);
    resDesc.res.pitch2D.desc = cudaCreateChannelDesc<float>();

    cudaTextureDesc texDesc = {};
    texDesc.readMode = cudaReadModeElementType;
    texDesc.addressMode[0] = cudaAddressModeClamp;
    texDesc.addressMode[1] = cudaAddressModeClamp;
    texDesc.filterMode = cudaFilterModePoint;
    texDesc.normalizedCoords = 0;

    cudaTextureObject_t tex_Y = 0;
    cudaCreateTextureObject(&tex_Y, &resDesc, &texDesc, nullptr);

    dim3 searchBlocks((width + 15) / 16, (height + 15) / 16);
    dim3 searchThreads(16, 16);

    Kernel_L2_BlockMatching << <searchBlocks, searchThreads, 0, stream >> >
        (tex_Y, gpuMem.d_SearchPool, width, height);

    cudaDestroyTextureObject(tex_Y);

    // --- 6. COLLABORATIVE BAYES & NORMALIZATION (Y, U, V) ---
    cudaMemsetAsync(gpuMem.d_Accum_Y, 0, width * height * sizeof(float), stream);
    cudaMemsetAsync(gpuMem.d_Accum_U, 0, width * height * sizeof(float), stream);
    cudaMemsetAsync(gpuMem.d_Accum_V, 0, width * height * sizeof(float), stream);

    int bayes_step = 3;
    dim3 bayesBlocks((width + bayes_step - 1) / bayes_step, (height + bayes_step - 1) / bayes_step);
    dim3 bayesThreads(8, 8);
    size_t sharedMemBytes = MAX_PATCHES * 64 * sizeof(float);

    dim3 normBlocks((width + 31) / 32, (height + 15) / 16);
    dim3 normThreads(32, 16);

    cudaMemsetAsync(gpuMem.d_Weight_Count, 0, width * height * sizeof(float), stream);
    Kernel_CollaborativeBayes_Aggregate << <bayesBlocks, bayesThreads, sharedMemBytes, stream >> >
        (gpuMem.d_Y_planar, gpuMem.d_SearchPool, gpuMem.d_NoiseCov_Y, gpuMem.d_Accum_Y, gpuMem.d_Weight_Count, width, height, bayes_step);
    Kernel_Normalize_Accumulators << <normBlocks, normThreads, 0, stream >> >
        (gpuMem.d_Accum_Y, gpuMem.d_Weight_Count, width, height);

    cudaMemsetAsync(gpuMem.d_Weight_Count, 0, width * height * sizeof(float), stream);
    Kernel_CollaborativeBayes_Aggregate << <bayesBlocks, bayesThreads, sharedMemBytes, stream >> >
        (gpuMem.d_U_planar, gpuMem.d_SearchPool, gpuMem.d_NoiseCov_U, gpuMem.d_Accum_U, gpuMem.d_Weight_Count, width, height, bayes_step);
    Kernel_Normalize_Accumulators << <normBlocks, normThreads, 0, stream >> >
        (gpuMem.d_Accum_U, gpuMem.d_Weight_Count, width, height);

    cudaMemsetAsync(gpuMem.d_Weight_Count, 0, width * height * sizeof(float), stream);
    Kernel_CollaborativeBayes_Aggregate << <bayesBlocks, bayesThreads, sharedMemBytes, stream >> >
        (gpuMem.d_V_planar, gpuMem.d_SearchPool, gpuMem.d_NoiseCov_V, gpuMem.d_Accum_V, gpuMem.d_Weight_Count, width, height, bayes_step);
    Kernel_Normalize_Accumulators << <normBlocks, normThreads, 0, stream >> >
        (gpuMem.d_Accum_V, gpuMem.d_Weight_Count, width, height);

    // --- 7. BACKWARD COLOR CONVERSION ---
    Kernel_Convert_YUV_to_BGRA_32f << <blocksPerGrid, threadsPerBlock, 0, stream >> >
        (
            gpuMem.d_Accum_Y, gpuMem.d_Accum_U, gpuMem.d_Accum_V,
            reinterpret_cast<const float4*>(inBuffer),
            reinterpret_cast<float4*>(outBuffer),
            width, height, width, dstPitch
            );

    // --- 8. CLEANUP & SYNC ---
    // Keeps the UI smooth while the core computes
    cudaStreamSynchronize(stream);

    return;
}
