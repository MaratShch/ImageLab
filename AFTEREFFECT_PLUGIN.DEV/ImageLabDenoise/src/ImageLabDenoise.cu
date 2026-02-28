#include <cstdio>
#include "CUDA/CudaMemHandler.cuh"
#include "ImageLabDenoise_GPU.hpp"


bool alloc_cuda_memory_buffers (CudaMemHandler& mem, int32_t target_tile_width, int32_t target_tile_height)
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

void free_cuda_memory_buffers(CudaMemHandler& mem)
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



CUDA_KERNEL_CALL
void ImageLabDenoise_CUDA
(
    const float* RESTRICT inBuffer, // source (input) buffer
    float* RESTRICT outBuffer,      // destination (output) buffer
    int srcPitch,                   // source buffer pitch in pixels 
    int dstPitch,                   // destination buffer pitch in pixels
    int width,                      // horizontal image size in pixels
    int height,                     // vertical image size in lines
    const AlgoControls* algoGpuParams, // algorithm controls
    int frameCount,
    cudaStream_t stream
)
{
    const int srcPitchBytes = srcPitch * sizeof(float4);
    const int dstPitchBytes = dstPitch * sizeof(float4);

    return;
}