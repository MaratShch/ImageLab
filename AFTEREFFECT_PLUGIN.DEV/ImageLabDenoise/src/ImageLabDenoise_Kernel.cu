#include <cstdio>
#include <algorithm>
#include <cuda_runtime.h>
#include "CUDA/CudaMemHandler.cuh"
#include "ImageLabDenoise_GPU.hpp"

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
inline bool alloc_cuda_memory_buffers (CudaMemHandler& mem, int32_t target_tile_width, int32_t target_tile_height)
{
    mem.tileW = target_tile_width;
    mem.tileH = target_tile_height;
    mem.padW = target_tile_width + 32;
    mem.padH = target_tile_height + 32;
    mem.frameSizePadded = mem.padW * mem.padH;

    // K=32 Max patches for high accuracy mode
    const size_t bytes_full   = mem.frameSizePadded * sizeof(float);
    const size_t bytes_half   = (mem.padW / 2) * (mem.padH / 2) * sizeof(float);
    const size_t bytes_quart  = (mem.padW / 4) * (mem.padH / 4) * sizeof(float);
    const size_t bytes_lut    = 256 * sizeof(float); 
    const size_t bytes_pool   = mem.frameSizePadded * 32 * sizeof(int2_coord); 

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

    CUDA_CHECK(cudaMalloc((void**)&mem.d_Accum_Y, bytes_full));
    CUDA_CHECK(cudaMalloc((void**)&mem.d_Accum_U, bytes_full));
    CUDA_CHECK(cudaMalloc((void**)&mem.d_Accum_V, bytes_full));
    CUDA_CHECK(cudaMalloc((void**)&mem.d_Weight_Count, bytes_full));

    CUDA_CHECK(cudaMalloc((void**)&mem.d_SearchPool, bytes_pool));

    return true;
}

inline void free_cuda_memory_buffers (CudaMemHandler& mem)
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
        // Convert 0.0-1.0 float to 0.0-255.0 range for internal processing
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

        // Convert back to 0.0-1.0 range for host format
        float r_out = r * 0.00392156862f;
        float g_out = g * 0.00392156862f;
        float b_out = b * 0.00392156862f;

        int dst_idx = y * dstPitchPixels + x;
        float original_alpha = pInputAlpha[dst_idx].w;

        pOutput[dst_idx] = make_float4(b_out, g_out, r_out, original_alpha);
    }
}

// =========================================================
// KERNELS: CORE ALGORITHM
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
        // Simple 2x2 box filter averaging for downsampling
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

// Initialize LUT with a huge value that will be overwritten by atomicMin
__global__ void Kernel_Oracle_Init(float* lut_Y, float* lut_U, float* lut_V) 
{
    int i = threadIdx.x; 
    if (i < 256) {
        lut_Y[i] = 1e30f;
        lut_U[i] = 1e30f;
        lut_V[i] = 1e30f;
    }
}

// FIX 1: LOWER THRESHOLD
// The previous 0.5f threshold was too high, ignoring real sensor noise on smooth surfaces.
__global__ void Kernel_Oracle_Gather
(
    const float* __restrict__ src_plane, float* __restrict__ lut_cov,
    int width, int height, int pitch, float color_shift
) 
{
    // Process every 8x8 block
    int bx = (blockIdx.x * blockDim.x + threadIdx.x) * 8;
    int by = (blockIdx.y * blockDim.y + threadIdx.y) * 8;

    if (bx + 7 < width && by + 7 < height) 
    {
        float mean = 0.0f;
        float sq_mean = 0.0f;

        // Calculate mean and variance of the 8x8 block
        for (int y = 0; y < 8; ++y) {
            for (int x = 0; x < 8; ++x) {
                float val = src_plane[(by + y) * pitch + (bx + x)];
                mean += val;
                sq_mean += val * val;
            }
        }

        mean *= 0.015625f; // divide by 64
        sq_mean *= 0.015625f;
        float variance = fmaxf(0.0f, sq_mean - (mean * mean));
        
        // Determine intensity bin (shifted for U/V)
        int bin = min(255, max(0, (int)(mean + color_shift + 0.5f)));

        // THRESHOLD LOWERED from 0.5f to 0.001f to catch real noise on flat surfaces
        if (variance > 0.001f) {
            atomicMinFloat(&lut_cov[bin], variance);
        }
    }
}

// FIX 2: ROBUST GAP FILLING (CRITICAL FIX)
// Prevents zero-noise bins that cause denoising failure.
// Prevents huge-noise bins that cause excessive blurring.
__global__ void Kernel_Oracle_Finalize(float* lut_Y, float* lut_U, float* lut_V) 
{
    if (threadIdx.x == 0 && blockIdx.x == 0) 
    {
        auto process_lut = [&](float* lut) 
        {
            // 1. Find first valid variance to anchor the start
            float first_valid = 1e30f;
            for (int i = 0; i < 256; ++i) {
                if (lut[i] < 1e29f) { // Found something that isn't init value
                    first_valid = lut[i];
                    break;
                }
            }
            // Fallback if the entire image is textured and no flat blocks exist
            if (first_valid > 1e29f) first_valid = 1.0f; 

            // 2. Forward pass: Fill gaps with the last known valid variance
            float last_valid = first_valid;
            for (int i = 0; i < 256; ++i) {
                if (lut[i] > 1e29f) lut[i] = last_valid;
                else last_valid = lut[i];
            }

            // 3. Backward pass: Catch any remaining gaps at the end and ensure monotonicity
            for (int i = 254; i >= 0; --i) {
                 // Ensure we don't propagate huge values backwards
                if (lut[i] > 1e29f) lut[i] = lut[i + 1];
            }

            // 4. Safety floor: Noise variance can never mathematically be zero.
            // This prevents division-by-zero and ensures *some* denoising always happens.
            for(int i = 0; i < 256; ++i) {
                 lut[i] = fmaxf(lut[i], 0.01f);
            }
        };
        process_lut(lut_Y);
        process_lut(lut_U);
        process_lut(lut_V);
    }
}

// ---------------------------------------------------------
// L2 BLOCK MATCHING ON LAPLACIAN PYRAMID
// Searches half-res image for structural matches to avoid fitting to noise.
// ---------------------------------------------------------
__global__ void Kernel_L2_BlockMatching_Pyramid
(
    const float* __restrict__ src_half, 
    int2_coord* searchPool,
    int width, int height,
    int half_w, int half_h, int half_pitch,
    int max_patches
)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;

    // Map current full-res reference pixel to half-res plane coordinates
    int half_x = x / 2;
    int half_y = y / 2;

    // Local registers for sorting best matches
    float best_ssds[32];
    int2_coord best_coords[32];

    for(int i = 0; i < max_patches; ++i) {
        best_ssds[i] = 1e30f;
        best_coords[i] = {x, y}; // Initialize with self-match
    }

    // Search window on HALF plane (effectively double size on FULL plane)
    for (int dy = -7; dy <= 7; ++dy) 
    {
        for (int dx = -7; dx <= 7; ++dx) 
        {
            int cand_hx = half_x + dx;
            int cand_hy = half_y + dy;

            float ssd = 0.0f;
            
            // 4x4 patch comparison on HALF plane (effectively 8x8 on FULL plane)
            #pragma unroll
            for (int r = 0; r < 4; ++r) {
                #pragma unroll
                for (int c = 0; c < 4; ++c) {
                    // Clamp coordinates to image boundaries
                    int cx1 = min(max(half_x + c, 0), half_w - 1);
                    int cy1 = min(max(half_y + r, 0), half_h - 1);
                    int cx2 = min(max(cand_hx + c, 0), half_w - 1);
                    int cy2 = min(max(cand_hy + r, 0), half_h - 1);
                    
                    float diff = src_half[cy1 * half_pitch + cx1] - src_half[cy2 * half_pitch + cx2];
                    ssd += diff * diff;
                }
            }

            // Insertion sort to keep top 'max_patches' matches
            if (ssd < best_ssds[max_patches - 1]) 
            {
                int insert_pos = max_patches - 1;
                while (insert_pos > 0 && ssd < best_ssds[insert_pos - 1]) {
                    insert_pos--;
                }
                for (int i = max_patches - 1; i > insert_pos; --i) {
                    best_ssds[i] = best_ssds[i - 1];
                    best_coords[i] = best_coords[i - 1];
                }
                best_ssds[insert_pos] = ssd;
                
                // Convert half-res match coordinates back to full-res
                int full_cand_x = min(max(x + (dx * 2), 0), width - 1);
                int full_cand_y = min(max(y + (dy * 2), 0), height - 1);
                best_coords[insert_pos] = {full_cand_x, full_cand_y};
            }
        }
    }

    // Write best matches to global memory pool
    int pool_offset = (y * width + x) * max_patches;
    for(int i = 0; i < max_patches; ++i) {
        searchPool[pool_offset + i] = best_coords[i];
    }
}

// ---------------------------------------------------------
// COLLABORATIVE BAYES & ATOMIC AGGREGATION
// Uses Block-Uniform Shrinkage stabilized by shared memory reduction.
// ---------------------------------------------------------
__global__ void Kernel_CollaborativeBayes_Aggregate
(
    const float* __restrict__ src_plane, 
    const int2_coord* __restrict__ pool,
    const float* __restrict__ oracle_lut, 
    float* __restrict__ accum_Y,
    float* __restrict__ accum_weight, 
    int width, int height, int pitch,
    int max_patches, int bayes_step, float color_shift,
    float strength_param
)
{
    int ref_x = blockIdx.x * bayes_step;
    int ref_y = blockIdx.y * bayes_step;
    int tx = threadIdx.x; // 0-7
    int ty = threadIdx.y; // 0-7
    int tid = ty * 8 + tx; // 0-63 flattened index

    if (ref_x >= width || ref_y >= height) return;

    // Shared memory segment for this block.
    // Bottom part stores patch data, top part used for reduction buffer.
    extern __shared__ float s_common[]; 
    float* s_group = s_common; 
    float* s_variance_reduce = (float*)&s_group[max_patches * 64]; 

    int pool_offset = (ref_y * width + ref_x) * max_patches;

    // --- 1. Load 3D group & Calculate Per-Pixel Mean ---
    float pixel_mean = 0.0f;
    for (int k = 0; k < max_patches; ++k)
    {
        int2_coord cand = pool[pool_offset + k];
        int cx = min(max(cand.x + tx, 0), width - 1);
        int cy = min(max(cand.y + ty, 0), height - 1);
        
        float val = src_plane[cy * pitch + cx];
        s_group[k * 64 + tid] = val; // Store in shared mem for later use
        pixel_mean += val;
    }
    pixel_mean /= (float)max_patches;

    // --- 2. Calculate Per-Pixel Variance across the stack ---
    float pixel_var = 0.0f;
    for (int k = 0; k < max_patches; ++k)
    {
        float diff = s_group[k * 64 + tid] - pixel_mean;
        pixel_var += diff * diff;
    }
    // Unbiased variance estimate
    pixel_var = fmaxf(0.0f, pixel_var / (float)(max_patches - 1));

    // --- 3. SHARED MEMORY REDUCTION ---
    // Average the variance across the entire 8x8 block to stabilize the estimate
    // and eliminate single-pixel noise spikes (green/purple dots).
    s_variance_reduce[tid] = pixel_var;
    __syncthreads(); // Wait for all 64 threads to write their variance

    // Standard parallel reduction pattern
    if (tid < 32) s_variance_reduce[tid] += s_variance_reduce[tid + 32]; __syncthreads();
    if (tid < 16) s_variance_reduce[tid] += s_variance_reduce[tid + 16]; __syncthreads();
    if (tid < 8)  s_variance_reduce[tid] += s_variance_reduce[tid + 8];  __syncthreads();
    if (tid < 4)  s_variance_reduce[tid] += s_variance_reduce[tid + 4];  __syncthreads();
    if (tid < 2)  s_variance_reduce[tid] += s_variance_reduce[tid + 2];  __syncthreads();
    if (tid < 1)  s_variance_reduce[tid] += s_variance_reduce[tid + 1];  __syncthreads();

    // Thread 0 calculates the single stable block-uniform weight
    if (tid == 0)
    {
        // Total block variance (Signal + Noise)
        float block_total_var = s_variance_reduce[0] / 64.0f;

        // Calculate mean of the reference patch (k=0) for reliable Oracle bin lookup
        float ref_patch_mean = 0.0f;
        #pragma unroll
        for (int i = 0; i < 64; ++i) {
            ref_patch_mean += s_group[i]; // s_group[0*64 + i] is the ref patch
        }
        ref_patch_mean *= 0.015625f; // divide by 64

        int bin = min(255, max(0, (int)(ref_patch_mean + color_shift + 0.5f)));
        float noise_var = oracle_lut[bin] * strength_param;

        // Standard Wiener Shrinkage: Weight = (TotalVar - NoiseVar) / TotalVar
        float signal_var = fmaxf(0.0f, block_total_var - noise_var);
        
        // Broadcast the stable weight back to s_variance_reduce[0] for all threads
        s_variance_reduce[0] = signal_var / (signal_var + noise_var + 1e-6f);
    }
    __syncthreads(); // Wait for thread 0 to calculate weight

    // All threads read the same stable weight
    float stable_shrinkage_weight = s_variance_reduce[0];

    // --- 4. Aggregation with Stable Weight ---
    for (int k = 0; k < max_patches; ++k)
    {
        int2_coord cand = pool[pool_offset + k];
        int cx = cand.x + tx;
        int cy = cand.y + ty;

        if (cx >= 0 && cx < width && cy >= 0 && cy < height)
        {
            float noisy_val = s_group[k * 64 + tid];
            // Denoise toward the group mean using the uniform weight
            float denoised_val = pixel_mean + stable_shrinkage_weight * (noisy_val - pixel_mean);
            int global_idx = cy * width + cx;

            // Atomic accumulation to handle overlapping blocks
            atomicAdd(&accum_Y[global_idx], denoised_val);
            atomicAdd(&accum_weight[global_idx], 1.0f);
        }
    }
}

// Normalizes the accumulated pixel values by the total weight count
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
        if (weight > 1e-6f)
        {
            accum_Y[idx] /= weight;
        }
    }
}

// =========================================================
// MAIN GPU DISPATCHER
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
    // Strict mapping of UI parameters
    const ProcAccuracy accuracy             = algoGpuParams->accuracy;
    const float master_denoise_amount       = algoGpuParams->master_denoise_amount;
    const float luma_strength               = algoGpuParams->luma_strength;
    const float chroma_strength             = algoGpuParams->chroma_strength;

    // Number of patches mapped dynamically based on accuracy mode
    const int max_patches_per_group = (accuracy == ProcAccuracy::AccHigh) ? 32 : 16;
    
    // Stride between reference blocks. 3 ensures overlap for smooth results 
    // while reducing atomic contention.
    int bayes_step = 3; 

    // --- 1. STATIC MEMORY CACHE ---
    // Re-allocates VRAM only if image dimensions change
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
    
    // --- 2. FORWARD COLOR CONVERSION (BGRA -> YUV Planar) ---
    Kernel_Convert_BGRA_32f_YUV <<<blocksPerGrid, threadsPerBlock, 0, stream >>>
    (
        reinterpret_cast<const float4*>(inBuffer),
        gpuMem.d_Y_planar, gpuMem.d_U_planar, gpuMem.d_V_planar,
        width, height, srcPitch, width
    );

    // --- 3. SPATIAL DOWNSAMPLING (Create Half-Res Luma for Matching) ---
    const int half_W = width / 2;
    const int half_H = height / 2;
    dim3 blocksHalf((half_W + 31) / 32, (half_H + 15) / 16);

    // We only need Y for block matching, but converting all keeps symmetry clean
    Kernel_Downsample_Laplacian <<<blocksHalf, threadsPerBlock, 0, stream >>>
        (gpuMem.d_Y_planar, gpuMem.d_Y_half, width, height, half_W, half_H, width, half_W);
    // U/V half-planes are currently unused but available for future expansions
    Kernel_Downsample_Laplacian <<<blocksHalf, threadsPerBlock, 0, stream >>>
        (gpuMem.d_U_planar, gpuMem.d_U_half, width, height, half_W, half_H, width, half_W);
    Kernel_Downsample_Laplacian <<<blocksHalf, threadsPerBlock, 0, stream >>>
        (gpuMem.d_V_planar, gpuMem.d_V_half, width, height, half_W, half_H, width, half_W);

    // --- 4. BLIND NOISE ORACLE ---
    // Initialize LUT with huge values
    Kernel_Oracle_Init<<<1, 256, 0, stream>>>(gpuMem.d_NoiseCov_Y, gpuMem.d_NoiseCov_U, gpuMem.d_NoiseCov_V);

    int oracle_grid_W = width / 8;
    int oracle_grid_H = height / 8;
    dim3 oracleBlocks((oracle_grid_W + 31) / 32, (oracle_grid_H + 15) / 16);

    // Gather minimum variance from flat blocks across the image
    Kernel_Oracle_Gather<<<oracleBlocks, threadsPerBlock, 0, stream>>>
        (gpuMem.d_Y_planar, gpuMem.d_NoiseCov_Y, width, height, width, 0.0f);
    // Shift U/V by +128.0f to map -128..127 range to 0..255 bin index
    Kernel_Oracle_Gather<<<oracleBlocks, threadsPerBlock, 0, stream>>>
        (gpuMem.d_U_planar, gpuMem.d_NoiseCov_U, width, height, width, 128.0f);
    Kernel_Oracle_Gather<<<oracleBlocks, threadsPerBlock, 0, stream>>>
        (gpuMem.d_V_planar, gpuMem.d_NoiseCov_V, width, height, width, 128.0f);

    // Interpolate gaps and ensure robust noise curve
    Kernel_Oracle_Finalize<<<1, 1, 0, stream>>>
        (gpuMem.d_NoiseCov_Y, gpuMem.d_NoiseCov_U, gpuMem.d_NoiseCov_V);

    // --- 5. L2 BLOCK MATCHING (Performed on Half-Res Luma) ---
    // Generates a single shared pooling map based on structural information.
    dim3 searchBlocks((width + 15) / 16, (height + 15) / 16);
    dim3 searchThreads(16, 16);

    Kernel_L2_BlockMatching_Pyramid<<<searchBlocks, searchThreads, 0, stream>>>
        (gpuMem.d_Y_half, gpuMem.d_SearchPool, width, height, half_W, half_H, half_W, max_patches_per_group);

    // --- 6. COLLABORATIVE BAYES & NORMALIZATION ---
    // Reset accumulators
    cudaMemsetAsync(gpuMem.d_Accum_Y, 0, width * height * sizeof(float), stream);
    cudaMemsetAsync(gpuMem.d_Accum_U, 0, width * height * sizeof(float), stream);
    cudaMemsetAsync(gpuMem.d_Accum_V, 0, width * height * sizeof(float), stream);
    
    dim3 bayesBlocks((width + bayes_step - 1) / bayes_step, (height + bayes_step - 1) / bayes_step);
    dim3 bayesThreads(8, 8); 
    
    // Calculate required shared memory: Patch Data + Reduction Buffer
    size_t sharedMemBytes = (max_patches_per_group * 64 * sizeof(float)) + (64 * sizeof(float));
    
    dim3 normBlocks((width + 31) / 32, (height + 15) / 16);
    dim3 normThreads(32, 16);

    // Process Y Plane
    cudaMemsetAsync(gpuMem.d_Weight_Count, 0, width * height * sizeof(float), stream);
    Kernel_CollaborativeBayes_Aggregate<<<bayesBlocks, bayesThreads, sharedMemBytes, stream>>>
        (gpuMem.d_Y_planar, gpuMem.d_SearchPool, gpuMem.d_NoiseCov_Y, gpuMem.d_Accum_Y, gpuMem.d_Weight_Count, width, height, width, max_patches_per_group, bayes_step, 0.0f, master_denoise_amount * luma_strength);
    Kernel_Normalize_Accumulators<<<normBlocks, normThreads, 0, stream>>>
        (gpuMem.d_Accum_Y, gpuMem.d_Weight_Count, width, height);

    // Process U Plane (shifted +128)
    cudaMemsetAsync(gpuMem.d_Weight_Count, 0, width * height * sizeof(float), stream);
    Kernel_CollaborativeBayes_Aggregate<<<bayesBlocks, bayesThreads, sharedMemBytes, stream>>>
        (gpuMem.d_U_planar, gpuMem.d_SearchPool, gpuMem.d_NoiseCov_U, gpuMem.d_Accum_U, gpuMem.d_Weight_Count, width, height, width, max_patches_per_group, bayes_step, 128.0f, master_denoise_amount * chroma_strength);
    Kernel_Normalize_Accumulators<<<normBlocks, normThreads, 0, stream>>>
        (gpuMem.d_Accum_U, gpuMem.d_Weight_Count, width, height);

    // Process V Plane (shifted +128)
    cudaMemsetAsync(gpuMem.d_Weight_Count, 0, width * height * sizeof(float), stream);
    Kernel_CollaborativeBayes_Aggregate<<<bayesBlocks, bayesThreads, sharedMemBytes, stream>>>
        (gpuMem.d_V_planar, gpuMem.d_SearchPool, gpuMem.d_NoiseCov_V, gpuMem.d_Accum_V, gpuMem.d_Weight_Count, width, height, width, max_patches_per_group, bayes_step, 128.0f, master_denoise_amount * chroma_strength);
    Kernel_Normalize_Accumulators<<<normBlocks, normThreads, 0, stream>>>
        (gpuMem.d_Accum_V, gpuMem.d_Weight_Count, width, height);

    // --- 7. BACKWARD COLOR CONVERSION (YUV Planar -> BGRA) ---
    Kernel_Convert_YUV_to_BGRA_32f <<<blocksPerGrid, threadsPerBlock, 0, stream >>>
    (
        gpuMem.d_Accum_Y, gpuMem.d_Accum_U, gpuMem.d_Accum_V,
        reinterpret_cast<const float4*>(inBuffer), 
        reinterpret_cast<float4*>(outBuffer),
        width, height, width, dstPitch
    );

    // --- 8. CLEANUP & SYNC ---
//    cudaStreamSynchronize(stream);
    cudaDeviceSynchronize();

    return;
}