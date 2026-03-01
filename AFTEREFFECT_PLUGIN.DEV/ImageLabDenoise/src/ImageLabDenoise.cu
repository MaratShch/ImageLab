#include <cstdio>
#include <algorithm>
#include "CUDA/CudaMemHandler.cuh"
#include "ImageLabDenoise_GPU.hpp"


inline bool alloc_cuda_memory_buffers (CudaMemHandler& mem, int32_t target_tile_width, int32_t target_tile_height)
{
    // Add 16 pixels of halo padding to all sides (32 total per dimension)
    mem.tileW = target_tile_width;
    mem.tileH = target_tile_height;
    mem.padW = target_tile_width + 32;
    mem.padH = target_tile_height + 32;
    mem.frameSizePadded = mem.padW * mem.padH;

    const size_t bytes_full   = mem.frameSizePadded * sizeof(float);
    const size_t bytes_half   = (mem.padW / 2) * (mem.padH / 2) * sizeof(float);
    const size_t bytes_quart  = (mem.padW / 4) * (mem.padH / 4) * sizeof(float);
    const size_t bytes_lut    = 256 * 256 * sizeof(float); // 256 intensities x 256 matrix elements

    // Max patches per reference pixel: 128. 
    // This defines our maximum VRAM footprint for the 3D block matching coordinates.
    const size_t bytes_pool   = mem.frameSizePadded * 128 * sizeof(int2_coord);

    // 1. ALLOCATE PYRAMID LEVELS
    CUDA_CHECK(cudaMalloc((void**)&mem.d_Y_planar, bytes_full));
    CUDA_CHECK(cudaMalloc((void**)&mem.d_U_planar, bytes_full));
    CUDA_CHECK(cudaMalloc((void**)&mem.d_V_planar, bytes_full));

    CUDA_CHECK(cudaMalloc((void**)&mem.d_Y_half, bytes_half));
    CUDA_CHECK(cudaMalloc((void**)&mem.d_U_half, bytes_half));
    CUDA_CHECK(cudaMalloc((void**)&mem.d_V_half, bytes_half));

    CUDA_CHECK(cudaMalloc((void**)&mem.d_Y_quart, bytes_quart));
    CUDA_CHECK(cudaMalloc((void**)&mem.d_U_quart, bytes_quart));
    CUDA_CHECK(cudaMalloc((void**)&mem.d_V_quart, bytes_quart));

    // 2. ALLOCATE ORACLE COVARIANCE LUTS
    CUDA_CHECK(cudaMalloc((void**)&mem.d_NoiseCov_Y, bytes_lut));
    CUDA_CHECK(cudaMalloc((void**)&mem.d_NoiseCov_U, bytes_lut));
    CUDA_CHECK(cudaMalloc((void**)&mem.d_NoiseCov_V, bytes_lut));

    // 3. ALLOCATE PILOT & ACCUMULATORS
    CUDA_CHECK(cudaMalloc((void**)&mem.d_Pilot_Y, bytes_full));
    CUDA_CHECK(cudaMalloc((void**)&mem.d_Pilot_U, bytes_full));
    CUDA_CHECK(cudaMalloc((void**)&mem.d_Pilot_V, bytes_full));

    CUDA_CHECK(cudaMalloc((void**)&mem.d_Accum_Y, bytes_full));
    CUDA_CHECK(cudaMalloc((void**)&mem.d_Accum_U, bytes_full));
    CUDA_CHECK(cudaMalloc((void**)&mem.d_Accum_V, bytes_full));
    CUDA_CHECK(cudaMalloc((void**)&mem.d_Weight_Count, bytes_full));

    // 4. ALLOCATE THE SEARCH POOL 
    CUDA_CHECK(cudaMalloc((void**)&mem.d_SearchPool, bytes_pool));

    return true;
}

inline void free_cuda_memory_buffers (CudaMemHandler& mem)
{
    // Safely free all VRAM buffers back to the system
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
// KERNEL: CONVERT BGRA_32f TO ORTHONORMAL YUV PLANAR
// =========================================================
__global__ void Kernel_Convert_BGRA_32f_YUV
(
    const float4* RESTRICT pInput, // Cast PF_Pixel_BGRA_32f to float4 for 128-bit loads
    float* RESTRICT d_Y,           // Dest Planar Y
    float* RESTRICT d_U,           // Dest Planar U
    float* RESTRICT d_V,           // Dest Planar V
    int width,
    int height,
    int srcPitchPixels,                // Interleaved input pitch
    int dstPitchPixels                 // Planar output pitch
)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height)
    {
        // 1. Single 128-bit Coalesced Read (BGRA layout: x=B, y=G, z=R, w=A)
        float4 pixel = pInput[y * srcPitchPixels + x];

        // 2. Scale up to [0.0f, 255.0f] range for the Oracle
        float b = pixel.x * 255.0f;
        float g = pixel.y * 255.0f;
        float r = pixel.z * 255.0f;

        // 3. Orthonormal Math
        float y_out = (r + g + b) * 0.57735027f;
        float u_out = (r - b) * 0.70710678f;
        float v_out = (r + b - 2.0f * g) * 0.40824829f;

        // 4. Store to Planar Buffers
        int dst_idx = y * dstPitchPixels + x;
        d_Y[dst_idx] = y_out;
        d_U[dst_idx] = u_out;
        d_V[dst_idx] = v_out;
    }
}

// =========================================================
// KERNEL: CONVERT ORTHONORMAL YUV PLANAR TO BGRA_32f
// =========================================================
__global__ void Kernel_Convert_YUV_to_BGRA_32f
(
    const float* RESTRICT d_Y,           // Source Planar Y
    const float* RESTRICT d_U,           // Source Planar U
    const float* RESTRICT d_V,           // Source Planar V
    const float4* RESTRICT pInputAlpha,  // Original input buffer (to preserve Alpha)
    float4* RESTRICT pOutput,            // Dest interleaved BGRA_32f
    int width,
    int height,
    int srcPitchPixels,                      // Planar input pitch
    int dstPitchPixels                       // Interleaved output pitch
)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height)
    {
        int src_idx = y * srcPitchPixels + x;

        // 1. Read Planar Data
        float y_in = d_Y[src_idx];
        float u_in = d_U[src_idx];
        float v_in = d_V[src_idx];

        // 2. Inverse Orthonormal Math
        float r = y_in * 0.57735027f + u_in * 0.70710678f + v_in * 0.40824829f;
        float g = y_in * 0.57735027f - v_in * 0.81649658f;
        float b = y_in * 0.57735027f - u_in * 0.70710678f + v_in * 0.40824829f;

        // 3. Scale back down to [0.0f, 1.0f] range. 
        // NOTE: No min/max clamping to allow HDR overbrights (>1.0f) to survive!
        float r_out = r * 0.00392156862f;
        float g_out = g * 0.00392156862f;
        float b_out = b * 0.00392156862f;

        // 4. Retrieve exact original Alpha float from the source buffer
        int dst_idx = y * dstPitchPixels + x;
        float original_alpha = pInputAlpha[dst_idx].w;

        // 5. Single 128-bit Coalesced Write
        pOutput[dst_idx] = make_float4(b_out, g_out, r_out, original_alpha);
    }
}

// =========================================================
// KERNEL: LAPLACIAN PYRAMID DOWNSAMPLE
// =========================================================
__global__ void Kernel_Downsample_Laplacian
(
    const float* RESTRICT src_plane,
    float* RESTRICT dst_plane,
    int src_width, 
    int src_height,
    int dst_width, 
    int dst_height
)
{
    // Global 2D Thread Coordinates
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < dst_width && y < dst_height)
    {
        // Calculate corresponding source coordinates (2x scale)
        int src_x = x * 2;
        int src_y = y * 2;

        // Ensure we don't read out of bounds on the source edges
        int sx_1 = std::min(src_x + 1, src_width - 1);
        int sy_1 = std::min(src_y + 1, src_height - 1);

        // 2x2 Box Filter (Standard Haar/Laplacian downsample)
        float p00 = src_plane[src_y * src_width + src_x];
        float p10 = src_plane[src_y * src_width + sx_1];
        float p01 = src_plane[sy_1 * src_width + src_x];
        float p11 = src_plane[sy_1 * src_width + sx_1];

        dst_plane[y * dst_width + x] = (p00 + p10 + p01 + p11) * 0.25f;
    }
}



CUDA_KERNEL_CALL
void ImageLabDenoise_CUDA
(
    const float* RESTRICT inBuffer, // source (input) interleaved buffer [Host]
    float* RESTRICT outBuffer,      // destination (output) interleaved buffer [Host]
    int srcPitch,                   // source buffer pitch in pixels
    int dstPitch,                   // destination buffer pitch in pixels
    int width,                      // horizontal image size in pixels
    int height,                     // vertical image size in lines
    const AlgoControls* RESTRICT algoGpuParams, // algorithm controls
    int frameCount,
    cudaStream_t stream
)
{
    // 1. Acquire algorithm controls parameters
    const ProcAccuracy accuracy             = algoGpuParams->accuracy;
    const float master_denoise_amount       = algoGpuParams->master_denoise_amount;
    const float luma_strength               = algoGpuParams->luma_strength;
    const float chroma_strength             = algoGpuParams->chroma_strength;
    const float fine_detail_preservation    = algoGpuParams->fine_detail_preservation;
    const float coarse_noise_reduction      = algoGpuParams->coarse_noise_reduction;

    // =========================================================
    // 2. ALLOCATE GPU MEMORY
    // =========================================================
    CudaMemHandler gpuMem{};
    if (false == alloc_cuda_memory_buffers(gpuMem, width, height))
    {
        return;
    }

    // Calculate transfer sizes based on pitch (4 floats per pixel for BGRA_32f)
    const size_t in_bytes  = height * srcPitch * 4 * sizeof(float);
    const size_t out_bytes = height * dstPitch * 4 * sizeof(float);

    // =========================================================
    // 3. CUDA THREAD GRID SETUP
    // =========================================================
    dim3 threadsPerBlock(32, 16);
    dim3 blocksPerGrid((width + 31) / 32, (height + 15) / 16);
    
    // =========================================================
    // 4. KERNEL DISPATCH: FORWARD COLOR CONVERSION
    // =========================================================
    Kernel_Convert_BGRA_32f_YUV <<<blocksPerGrid, threadsPerBlock, 0, stream >>>
    (
        reinterpret_cast<const float4*>(inBuffer),
        gpuMem.d_Y_planar,
        gpuMem.d_U_planar,
        gpuMem.d_V_planar,
        width,
        height,
        srcPitch,
        width
    );

    // =========================================================
    // 5. START CORE DENOISE ALGORITHM
    // =========================================================

    // --> Kernel_BuildLaplacianPyramid<<<...>>>
    // --> Kernel_EstimateBlindOracle<<<...>>>
    // --> Kernel_L2_BlockMatching<<<...>>>
    // --> Kernel_CollaborativeBayesShrinkage<<<...>>>
    // --> Kernel_AtomicAggregation<<<...>>>

    // =========================================================
    // 7. KERNEL DISPATCH: BACKWARD COLOR CONVERSION
    // =========================================================
    Kernel_Convert_YUV_to_BGRA_32f <<<blocksPerGrid, threadsPerBlock, 0, stream >>>
    (
        gpuMem.d_Accum_Y, // Reading from the final denoised accumulator
        gpuMem.d_Accum_U,
        gpuMem.d_Accum_V,
        reinterpret_cast<const float4*>(inBuffer), // Source for original Alpha
        reinterpret_cast<float4*>(outBuffer),
        width,
        height,
        width,
        dstPitch
    );

    // Wait till all GPU memory copies and kernels in this stream are completed
    cudaStreamSynchronize (stream);

    // Release VRAM back to the system
    free_cuda_memory_buffers (gpuMem);

    return;
}