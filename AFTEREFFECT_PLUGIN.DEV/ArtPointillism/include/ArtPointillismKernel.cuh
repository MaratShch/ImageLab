#ifndef __IMAGE_LAB_ART_POINTILISM_GPU_ACCELERATOR_KERNELS_DEFINITIONS__
#define __IMAGE_LAB_ART_POINTILISM_GPU_ACCELERATOR_KERNELS_DEFINITIONS__

#include <cuda_runtime.h>

// --- ADOBE INTEROP TYPE ---
// 16-byte aligned float4. Corresponds to BGRA_32f.
// Access: .x=Blue, .y=Green, .z=Red, .w=Alpha
typedef float4 PixelRGBA_32f;

// --- INTERNAL TYPES ---

// 1. Structure Map (Phase 1)
// We pack Density and Orientation into one float2 to save bandwidth.
// .x = Density (0.0 - 1.0)
// .y = Orientation Angle (Radians)
typedef float2 DensityInfo;

// 2. The Dot Definition (Phase 2 Output)
struct __align__(16) GPUDot
{
    float pos_x;
    float pos_y;
    // We can pack encoded color/state here later if needed
    float padding1;
    float padding2;
};

// 3. JFA Cell (Phase 3)
// .x = Seed Index (int cast to float, or use int2)
// .y = Seed X
// .z = Seed Y
// .w = Padding (or Distance cache)
typedef int4 JFACell; // Using int4 is cleaner for the Index

                      // 4. Dot Color Accumulator (Phase 4)
                      // Used for atomicAdd. 
                      // .x = Sum L, .y = Sum a, .z = Sum b, .w = Pixel Count
typedef float4 DotColorAccumulator;

// 5. Global Counters (The CPU replacement)
// index 0: actual_dot_count
// index 1: debug_flag
// index 2: ...
typedef int GlobalCounters;

// --- CONSTANTS ---
#define BLOCK_SIZE_X 32
#define BLOCK_SIZE_Y 32
// Max dots to allocate VRAM for (e.g. 500,000 for 4K)
#define MAX_GPU_DOTS 1000000


class GpuContext final
{
public:
    // --- VRAM Buffers ---
    DensityInfo* d_densityMap = nullptr;
    JFACell*     d_jfaPing = nullptr;
    JFACell*     d_jfaPong = nullptr;
    GPUDot*      d_dots = nullptr;
    float4*      d_dotColors = nullptr; // Accumulator
    int*         d_counters = nullptr; // [0]: DotCount
                                       // --- State ---
    size_t current_alloc_pixels = 0;

    ~GpuContext()
    { 
        Cleanup();
        return;
    }

    void Cleanup (void)
    {
        if (d_densityMap)
            cudaFree(d_densityMap);
        if (d_jfaPing)
            cudaFree(d_jfaPing);
        if (d_jfaPong)
            cudaFree(d_jfaPong);
        if (d_dots)
            cudaFree(d_dots);
        if (d_dotColors)
            cudaFree(d_dotColors);
        if (d_counters)
            cudaFree(d_counters);
        
        current_alloc_pixels = 0;

        d_densityMap = nullptr;
        d_jfaPing = nullptr;
        d_jfaPong = nullptr;
        d_dots = nullptr;
        d_dotColors = nullptr;
        d_counters = nullptr;

        return;
    }

    // Smart Reallocation: Only mallocs if resolution increases
    void CheckAndReallocate (int width, int height)
    {
        size_t needed_pixels = width * height;

        // Add 10% buffer or check if size changed significantly to avoid thrashing
        if (needed_pixels > current_alloc_pixels)
        {
            Cleanup();

            // Allocate VRAM
            cudaMalloc(&d_densityMap, needed_pixels * sizeof(DensityInfo));
            cudaMalloc(&d_jfaPing, needed_pixels * sizeof(JFACell));
            cudaMalloc(&d_jfaPong, needed_pixels * sizeof(JFACell));

            // Dot buffers are fixed size based on MAX constant
            cudaMalloc(&d_dots, MAX_GPU_DOTS * sizeof(GPUDot));
            cudaMalloc(&d_dotColors, MAX_GPU_DOTS * sizeof(float4));

            // Counters (Tiny)
            cudaMalloc(&d_counters, 32 * sizeof(int)); // Small scratch space

            current_alloc_pixels = needed_pixels;
        }
        return;
    }

};



#endif // __IMAGE_LAB_ART_POINTILISM_GPU_ACCELERATOR_KERNELS_DEFINITIONS__