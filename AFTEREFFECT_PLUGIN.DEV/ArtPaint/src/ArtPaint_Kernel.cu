#include "ArtPaint_GPU.hpp"
#include "CompileTimeUtils.hpp"
#include "ImageLabCUDA.hpp"
#include <algorithm>
#include <cmath>


// ============================================================================
// INTERNAL ARENA MANAGER
// ============================================================================
// Static pointers to hold our persistent memory across frames
static void* g_arenaMem = nullptr;
static size_t g_currentArenaSize = 0;

// Track the state that dictates our memory size
static int g_cachedWidth = 0;
static int g_cachedHeight = 0;
static RenderQuality g_cachedQuality = RenderQuality::Fast_HalfSize;
static int g_cachedRadius = 0;

// Re-using the alignment logic from your host code
constexpr size_t CUDA_ALIGNMENT = 256ull;
inline constexpr size_t AlignSizeCuda (const size_t size) noexcept
{
    return (size + CUDA_ALIGNMENT - 1) & ~(CUDA_ALIGNMENT - 1);
}

// Internal function to compute required bytes (matches host logic exactly)
static size_t GetRequiredBytes (int width, int height, RenderQuality quality, int radius) noexcept
{
    int proc_width = width;
    int proc_height = height;

    if (quality == RenderQuality::Fast_HalfSize)
    {
        proc_width >>= 1;
        proc_height >>= 1;
    }

    size_t frameSize = static_cast<size_t>(proc_width) * static_cast<size_t>(proc_height);
    size_t alignedFloatSize = AlignSizeCuda(frameSize * sizeof(float));

    size_t max_edges = frameSize * (2 * radius + 1) * 2;
    size_t alignedEdgeIdxSize = AlignSizeCuda(max_edges * sizeof(int));
    size_t alignedEdgeWeightSize = AlignSizeCuda(max_edges * sizeof(float));

    size_t totalBytes = 0;
    totalBytes += 12 * alignedFloatSize; // Y,U,V, Tensors, Lambda, Eig, imProc1, imProc2
    totalBytes += 2 * alignedEdgeIdxSize; // I, J
    totalBytes += 1 * alignedEdgeWeightSize; // LogW

    return totalBytes;
}

// Guarantees we have the right amount of memory without allocating every frame
static bool EnsureArenaMemory (int width, int height, RenderQuality quality, int radius) noexcept
{
    // If nothing changed that affects size, return success immediately! (Zero overhead)
    if (g_arenaMem != nullptr &&
        g_cachedWidth == width &&
        g_cachedHeight == height &&
        g_cachedQuality == quality &&
        g_cachedRadius == radius)
    {
        return true;
    }

    // Parameters changed. Calculate new size.
    size_t requiredBytes = GetRequiredBytes(width, height, quality, radius);

    // If we need more memory than we currently have, reallocate
    if (requiredBytes > g_currentArenaSize)
    {
        if (g_arenaMem != nullptr)
        {
            cudaFree(g_arenaMem);
            g_arenaMem = nullptr;
        }

        cudaError_t err = cudaMalloc(&g_arenaMem, requiredBytes);
        if (err != cudaSuccess)
        {
            return false; // Out of memory! Host should catch this.
        }
        g_currentArenaSize = requiredBytes;
    }

    // Update cached state
    g_cachedWidth = width;
    g_cachedHeight = height;
    g_cachedQuality = quality;
    g_cachedRadius = radius;

    return true;
}

// Optional cleanup function to be called when the plugin is completely destroyed
void FreeArtPaintArena() noexcept
{
    if (g_arenaMem != nullptr)
    {
        cudaFree(g_arenaMem);
        g_arenaMem = nullptr;
        g_currentArenaSize = 0;
    }
}


inline __device__ float4 HalfToFloat4(const Pixel16& in) noexcept
{
    return make_float4(__half2float(in.x), __half2float(in.y), __half2float(in.z), __half2float(in.w));
}

inline __device__ Pixel16 FloatToHalf4(const float4& in) noexcept
{
    Pixel16 v;
    v.x = __float2half_rn(in.x); v.y = __float2half_rn(in.y); v.z = __float2half_rn(in.z); v.w = __float2half_rn(in.w);
    return v;
}




CUDA_KERNEL_CALL
void ArtPaint_CUDA
(
    const void* RESTRICT inBuffer, // source (input) buffer
    void* RESTRICT outBuffer,      // destination (output) buffer
    int srcPitch,                   // source buffer pitch in pixels 
    int dstPitch,                   // destination buffer pitch in pixels
    int width,                      // horizontal image size in pixels
    int height,                     // vertical image size in lines
    const AlgoControls* algoGpuParams, // algorithm controls
    int frameCounter,
    bool isFloat16,                // true - compute in float16, false - compute in float32
    cudaStream_t stream
)
{
    const StrokeBias    bias    = algoGpuParams->bias;
    const RenderQuality quality = algoGpuParams->quality;
    const float         sigma   = algoGpuParams->sigma;
    const float         angular = algoGpuParams->angular;
    const float         angle   = algoGpuParams->angle;
    const int32_t       iter    = algoGpuParams->iter;

    if (true == EnsureArenaMemory (width, height, quality, 7))
    {

    }

    cudaDeviceSynchronize();

    return;
}