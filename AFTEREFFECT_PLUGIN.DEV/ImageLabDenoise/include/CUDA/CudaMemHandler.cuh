#pragma once

// ============================================================================
// File: CudaMemHandler.cuh
// Target: CUDA 10.2 / C++14 / NVIDIA GTX-1060 (3GB VRAM) & RTX-2000
// Description: Single-arena GPU memory management for the Lebrun NL-Bayes
//              denoising pipeline. All device-side buffers are sliced from
//              a single cudaMalloc'd arena; only one cudaFree is needed.
//
// Layout (Lebrun-faithful):
//   * Full 16x16 noise covariance per intensity bin per channel
//   * Per-bin mean intensity tables (for linear interpolation across bins)
//   * Shared weight accumulator (one buffer for all channels - see note below)
// ============================================================================

#include <cstdint>
#include <cstdio>
#include <cuda_runtime.h>
#include "ImageLabCUDA.hpp"

// ---------------------------------------------------------------------------
// CUDA Error Checking Macro
// ---------------------------------------------------------------------------
#define CUDA_CHECK(call)                                                        \
    do {                                                                        \
        cudaError_t err = call;                                                 \
        if (err != cudaSuccess) {                                               \
            fprintf(stderr, "CUDA Error at %s:%d - %s\n",                       \
                    __FILE__, __LINE__, cudaGetErrorString(err));               \
            return false;                                                       \
        }                                                                       \
    } while (0)


// ---------------------------------------------------------------------------
// COMPILE-TIME CONSTANTS
// ---------------------------------------------------------------------------
// Noise estimation: 256 intensity bins cover the [0, 255] integer range of
// pixel values (image is in [0, 1] float -> multiplied by 255 for binning).
constexpr int   NOISE_BINS                 = 256;

// Patch geometry (Lebrun fixes these at 4x4 = 16 pixels per patch)
constexpr int   PATCH_SIZE                 = 4;
constexpr int   PATCH_ELEMS                = PATCH_SIZE * PATCH_SIZE;     // 16
constexpr int   PATCH_ELEMS_SQ             = PATCH_ELEMS * PATCH_ELEMS;   // 256

// Apron padding for shared-memory tiles (covers search-window overhang).
// 32 is a safe over-allocation that keeps buffer rows warp-aligned.
constexpr int   PAD_APRON                  = 32;


// ---------------------------------------------------------------------------
// Arena alignment helper: round a byte count up to the next 256-byte boundary.
// 256-byte alignment guarantees coalesced global loads on all CC >= 5.0.
// ---------------------------------------------------------------------------
constexpr size_t Align256(size_t size)
{
    return (size + 255) & ~static_cast<size_t>(255);
}


// ---------------------------------------------------------------------------
// CudaMemHandler - single-arena memory layout
//
// All d_* pointers are aliases into d_arena_pool; none are independently
// allocated. To tear down, call free_cuda_memory_buffers which issues exactly
// one cudaFree on d_arena_pool.
//
// ---------------------------------------------------------------------------
struct CudaMemHandler
{
    // --- MASTER ARENA ---
    void*       d_arena_pool            = nullptr;
    size_t      total_allocated_bytes   = 0;

    // --- IMAGE DIMENSIONS ---
    int32_t     tileW                   = 0;   // input width, pixels
    int32_t     tileH                   = 0;   // input height, pixels
    int32_t     padW                    = 0;   // tileW + PAD_APRON
    int32_t     padH                    = 0;   // tileH + PAD_APRON
    int32_t     frameSizePadded         = 0;   // padW * padH

    // --- LEVEL 0: FULL-RESOLUTION PLANAR YUV ---
    float*      d_Y_planar              = nullptr;
    float*      d_U_planar              = nullptr;
    float*      d_V_planar              = nullptr;

    // --- MULTISCALE PING-PONG MOSAICS ---
    float*      d_MosaicA_Y             = nullptr;
    float*      d_MosaicA_U             = nullptr;
    float*      d_MosaicA_V             = nullptr;

    float*      d_MosaicB_Y             = nullptr;  // also used as "pilot" estimate
    float*      d_MosaicB_U             = nullptr;
    float*      d_MosaicB_V             = nullptr;

    // --- DIFFERENCE PYRAMID (reserved for future multi-scale loop) ---
    float*      d_Diff_Y                = nullptr;
    float*      d_Diff_U                = nullptr;
    float*      d_Diff_V                = nullptr;

    // --- NL-BAYES AGGREGATION BUFFERS ---
    float*      d_Accum_Y               = nullptr;
    float*      d_Accum_U               = nullptr;
    float*      d_Accum_V               = nullptr;

    float*      d_Weight                = nullptr;  // shared across Y/U/V (see note above)

    // --- NOISE ESTIMATION TABLES ---
    // Full 16x16 covariance matrix per bin per channel: 256 bins * 256 floats
    float*      d_NoiseCov_Y            = nullptr;
    float*      d_NoiseCov_U            = nullptr;
    float*      d_NoiseCov_V            = nullptr;

    // Per-bin mean intensity: 256 floats per channel
    // Used for intensity-domain interpolation of noise statistics
    float*      d_NoiseMean_Y           = nullptr;
    float*      d_NoiseMean_U           = nullptr;
    float*      d_NoiseMean_V           = nullptr;

    // Per-bin patch count: 256 ints per channel. Used for variance
    // normalization (divides accumulated squared DCT coeffs by count).
    int*        d_NoiseCounts_Y         = nullptr;
    int*        d_NoiseCounts_U         = nullptr;
    int*        d_NoiseCounts_V         = nullptr;
};


// ===========================================================================
// MEMORY MANAGEMENT (ARENA ALLOCATOR)
// ===========================================================================
inline bool alloc_cuda_memory_buffers(CudaMemHandler& mem,
                                      int32_t target_tile_width,
                                      int32_t target_tile_height)
{
    // 1. Record dimensions
    mem.tileW           = target_tile_width;
    mem.tileH           = target_tile_height;
    mem.padW            = target_tile_width  + PAD_APRON;
    mem.padH            = target_tile_height + PAD_APRON;
    mem.frameSizePadded = mem.padW * mem.padH;

    // 2. Per-slice raw byte sizes
    const size_t bytes_full_channel  = static_cast<size_t>(mem.frameSizePadded) * sizeof(float);
    const size_t bytes_noise_cov     = static_cast<size_t>(NOISE_BINS) * PATCH_ELEMS_SQ * sizeof(float);  // 256*256 floats = 256 KB
    const size_t bytes_noise_mean    = static_cast<size_t>(NOISE_BINS) * sizeof(float);                    // 1 KB
    const size_t bytes_noise_counts  = static_cast<size_t>(NOISE_BINS) * sizeof(int);                      // 1 KB

    // 3. Align each to 256-byte boundary (coalescing guarantee)
    const size_t a_full_ch      = Align256(bytes_full_channel);
    const size_t a_noise_cov    = Align256(bytes_noise_cov);
    const size_t a_noise_mean   = Align256(bytes_noise_mean);
    const size_t a_noise_counts = Align256(bytes_noise_counts);

    // 4. Total arena size
    //    16 full-channel buffers:
    //      3 (YUV planar) + 3 (MosaicA) + 3 (MosaicB) + 3 (Diff) + 3 (Accum) + 1 (Weight)
    //    3 noise-cov LUTs
    //    3 noise-mean LUTs
    //    3 noise-count LUTs
    mem.total_allocated_bytes =
          (a_full_ch      * 16)
        + (a_noise_cov    *  3)
        + (a_noise_mean   *  3)
        + (a_noise_counts *  3);

    // 5. Single master allocation
    CUDA_CHECK(cudaMalloc(&mem.d_arena_pool, mem.total_allocated_bytes));

    // 6. Slicer: returns current cursor, advances by `aligned_size`
    uint8_t* current_offset = static_cast<uint8_t*>(mem.d_arena_pool);

    auto SliceMemory = [&current_offset](size_t aligned_size) -> void* {
        void* ptr = current_offset;
        current_offset += aligned_size;
        return ptr;
    };

    // 7. Distribute pointers (order must be stable: it defines the layout)

    // --- full-resolution planar YUV ---
    mem.d_Y_planar     = static_cast<float*>(SliceMemory(a_full_ch));
    mem.d_U_planar     = static_cast<float*>(SliceMemory(a_full_ch));
    mem.d_V_planar     = static_cast<float*>(SliceMemory(a_full_ch));

    // --- mosaic ping-pong A ---
    mem.d_MosaicA_Y    = static_cast<float*>(SliceMemory(a_full_ch));
    mem.d_MosaicA_U    = static_cast<float*>(SliceMemory(a_full_ch));
    mem.d_MosaicA_V    = static_cast<float*>(SliceMemory(a_full_ch));

    // --- mosaic ping-pong B (also doubles as pilot estimate for pass 2) ---
    mem.d_MosaicB_Y    = static_cast<float*>(SliceMemory(a_full_ch));
    mem.d_MosaicB_U    = static_cast<float*>(SliceMemory(a_full_ch));
    mem.d_MosaicB_V    = static_cast<float*>(SliceMemory(a_full_ch));

    // --- diff pyramid (reserved for multi-scale loop) ---
    mem.d_Diff_Y       = static_cast<float*>(SliceMemory(a_full_ch));
    mem.d_Diff_U       = static_cast<float*>(SliceMemory(a_full_ch));
    mem.d_Diff_V       = static_cast<float*>(SliceMemory(a_full_ch));

    // --- NL-Bayes accumulators ---
    mem.d_Accum_Y      = static_cast<float*>(SliceMemory(a_full_ch));
    mem.d_Accum_U      = static_cast<float*>(SliceMemory(a_full_ch));
    mem.d_Accum_V      = static_cast<float*>(SliceMemory(a_full_ch));

    // --- shared weight accumulator ---
    mem.d_Weight       = static_cast<float*>(SliceMemory(a_full_ch));

    // --- noise covariance LUTs (full 16x16 per bin) ---
    mem.d_NoiseCov_Y   = static_cast<float*>(SliceMemory(a_noise_cov));
    mem.d_NoiseCov_U   = static_cast<float*>(SliceMemory(a_noise_cov));
    mem.d_NoiseCov_V   = static_cast<float*>(SliceMemory(a_noise_cov));

    // --- per-bin mean intensity tables ---
    mem.d_NoiseMean_Y  = static_cast<float*>(SliceMemory(a_noise_mean));
    mem.d_NoiseMean_U  = static_cast<float*>(SliceMemory(a_noise_mean));
    mem.d_NoiseMean_V  = static_cast<float*>(SliceMemory(a_noise_mean));

    // --- per-bin patch counts ---
    mem.d_NoiseCounts_Y = static_cast<int*>(SliceMemory(a_noise_counts));
    mem.d_NoiseCounts_U = static_cast<int*>(SliceMemory(a_noise_counts));
    mem.d_NoiseCounts_V = static_cast<int*>(SliceMemory(a_noise_counts));

    return true;
}


// ---------------------------------------------------------------------------
// Single cudaFree on the arena.
// All d_* pointers become invalid after this call.
// ---------------------------------------------------------------------------
inline void free_cuda_memory_buffers(CudaMemHandler& mem)
{
    if (mem.d_arena_pool)
    {
        cudaFree(mem.d_arena_pool);
        mem.d_arena_pool          = nullptr;
        mem.total_allocated_bytes = 0;
    }

    // Defensive: null out slice pointers so stale use is obvious.
    mem.d_Y_planar      = nullptr;
    mem.d_U_planar      = nullptr;
    mem.d_V_planar      = nullptr;
    mem.d_MosaicA_Y     = nullptr;  mem.d_MosaicA_U     = nullptr;  mem.d_MosaicA_V     = nullptr;
    mem.d_MosaicB_Y     = nullptr;  mem.d_MosaicB_U     = nullptr;  mem.d_MosaicB_V     = nullptr;
    mem.d_Diff_Y        = nullptr;  mem.d_Diff_U        = nullptr;  mem.d_Diff_V        = nullptr;
    mem.d_Accum_Y       = nullptr;  mem.d_Accum_U       = nullptr;  mem.d_Accum_V       = nullptr;
    mem.d_Weight        = nullptr;
    mem.d_NoiseCov_Y    = nullptr;  mem.d_NoiseCov_U    = nullptr;  mem.d_NoiseCov_V    = nullptr;
    mem.d_NoiseMean_Y   = nullptr;  mem.d_NoiseMean_U   = nullptr;  mem.d_NoiseMean_V   = nullptr;
    mem.d_NoiseCounts_Y = nullptr;  mem.d_NoiseCounts_U = nullptr;  mem.d_NoiseCounts_V = nullptr;
}
