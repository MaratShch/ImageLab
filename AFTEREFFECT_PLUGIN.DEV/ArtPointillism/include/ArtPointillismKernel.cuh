#ifndef __IMAGE_LAB_ART_POINTILISM_GPU_ACCELERATOR_KERNELS_DEFINITIONS__
#define __IMAGE_LAB_ART_POINTILISM_GPU_ACCELERATOR_KERNELS_DEFINITIONS__

#include <cuda_runtime.h>
#include <vector>
#include <algorithm>

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


// Constant for Palette Padding
// We pad unused palette slots with "Infinity" so the math ignores them.
static const float GPU_PALETTE_INF = 10000.0f;

// Stores the artistic properties of a dot after Decomposition
struct __align__(16) DotRenderInfo
{
    // Colors are indices into the Palette (0..31)
    int   colorIndex1;
    int   colorIndex2;
    float ratio;        // Mix ratio 0..1
    float orientation;  // Angle in radians (for Van Gogh)
                        // We can pack these tighter if needed, but 16-byte alignment is good for memory fetch
};

class GpuContext final
{
public:
    // --- VRAM BUFFERS ---
    DensityInfo*         d_densityMap = nullptr;
    JFACell*             d_jfaPing = nullptr;
    JFACell*             d_jfaPong = nullptr;

    // Dot Data
    GPUDot*              d_dots = nullptr; // Position
    DotColorAccumulator* d_dotColors = nullptr; // Sum of colors (Phase 4A)
    DotRenderInfo*       d_dotInfo = nullptr; // <--- MISSING MEMBER ADDED HERE

    int*                 d_counters = nullptr; // Atomic counters
    float4*              d_palette = nullptr; // Palette

                                              // --- STATE TRACKING ---
    size_t m_currentAllocPixels = 0;
    int    m_currentPainterId = -1;

public:
    GpuContext() = default;
    ~GpuContext() { Cleanup(); }

    void Cleanup (void)
    {
        if (d_densityMap) cudaFree(d_densityMap);
        if (d_jfaPing)    cudaFree(d_jfaPing);
        if (d_jfaPong)    cudaFree(d_jfaPong);
        if (d_dots)       cudaFree(d_dots);
        if (d_dotColors)  cudaFree(d_dotColors);
        if (d_dotInfo)    cudaFree(d_dotInfo); // <--- FREE HERE
        if (d_counters)   cudaFree(d_counters);
        if (d_palette)    cudaFree(d_palette);

        d_densityMap = nullptr;
        d_jfaPing = nullptr; d_jfaPong = nullptr;
        d_dots = nullptr; d_dotColors = nullptr; d_dotInfo = nullptr;
        d_counters = nullptr; d_palette = nullptr;

        m_currentAllocPixels = 0;
        m_currentPainterId = -1;

        return;
    }

    void CheckAndReallocate(int width, int height)
    {
        const size_t needed_pixels = static_cast<size_t>(width * height);

        if (needed_pixels > m_currentAllocPixels)
        {
            // 1. Free Image-Dependent Buffers
            if (d_densityMap) cudaFree(d_densityMap);
            if (d_jfaPing)    cudaFree(d_jfaPing);
            if (d_jfaPong)    cudaFree(d_jfaPong);

            // 2. Alloc Image-Dependent Buffers
            cudaMalloc(&d_densityMap, needed_pixels * sizeof(DensityInfo));
            cudaMalloc(&d_jfaPing, needed_pixels * sizeof(JFACell));
            cudaMalloc(&d_jfaPong, needed_pixels * sizeof(JFACell));

            // 3. Alloc Constant-Size Buffers (One-time check)
            if (!d_dots)      cudaMalloc(&d_dots, MAX_GPU_DOTS * sizeof(GPUDot));
            if (!d_dotColors) cudaMalloc(&d_dotColors, MAX_GPU_DOTS * sizeof(DotColorAccumulator));

            // <--- ALLOCATE DOT INFO HERE
            if (!d_dotInfo)   cudaMalloc(&d_dotInfo, MAX_GPU_DOTS * sizeof(DotRenderInfo));

            if (!d_counters)  cudaMalloc(&d_counters, 32 * sizeof(int));
            if (!d_palette)   cudaMalloc(&d_palette, 32 * sizeof(float4));

            m_currentAllocPixels = needed_pixels;
        }
        return;
    }

    void UpdatePaletteFromPlanar
    (
        const float* pal_L,
        const float* pal_a,
        const float* pal_b,
        int count,
        int style_id,
        cudaStream_t stream
    )
    {
        if (style_id != m_currentPainterId || d_palette == nullptr)
        {
            float4 temp_buffer[32];
            for (int i = 0; i < 32; ++i) {
                if (i < count) {
                    temp_buffer[i] = make_float4(pal_L[i], pal_a[i], pal_b[i], 0.0f);
                }
                else {
                    temp_buffer[i] = make_float4(GPU_PALETTE_INF, GPU_PALETTE_INF, GPU_PALETTE_INF, 0.0f);
                }
            }
            cudaMemcpyAsync(d_palette, temp_buffer, 32 * sizeof(float4), cudaMemcpyHostToDevice, stream);
            m_currentPainterId = style_id;
        }
        return;
    }
};

#endif // __IMAGE_LAB_ART_POINTILISM_GPU_ACCELERATOR_KERNELS_DEFINITIONS__