#ifndef __IMAGE_LAB_GUIDED_FILTER_BUFFER_HANDLER__
#define __IMAGE_LAB_GUIDED_FILTER_BUFFER_HANDLER__

#include <cstdint>

struct MemHandler
{
    // --- INFRASTRUCTURE TRACKING ---
    int64_t memBlockId;
    uint8_t* SuperBufferHead;

    // --- MEMORY GEOMETRY ---
    // Stride measured in elements (pixels), not bytes, matching your sizeX/sizeY rule
    int32_t strideElements; 

    // --- INCOMING BUFFERS (float32) ---
    // Orthonormal YUV planes + the Skin Mask
    const float* in_Y;
    const float* in_U;
    const float* in_V;
    const float* in_Mask;

    // --- OUTGOING BUFFERS (float32) ---
    // Processed Orthonormal YUV planes
    float* out_Y;
    float* out_U;
    float* out_V;

    // --- INTERMEDIATE SCRATCHPAD BUFFERS (float32) ---
    // Used by the Guided Filter math. Reused sequentially for Y, U, and V processing.
    float* temp_blur; // Holds the 1D intermediate pass for the separable box blur
    float* mean_I;    // Holds the blurred input plane
    float* mean_II;   // Holds the blurred squared input plane
    float* coef_a;    // Holds the variance/'a' coefficient, then overwritten with mean_a
    float* coef_b;    // Holds the 'b' coefficient, then overwritten with mean_b
};

MemHandler alloc_memory_buffers (int32_t sizeX, int32_t sizeY, const bool dbgPrn = false) noexcept;
void free_memory_buffers (MemHandler& algoMemHandler) noexcept;

inline bool mem_handler_valid(const MemHandler& hndl) noexcept
{
    // If the arena is valid, the 32-byte aligned slices are guaranteed to be valid.
    return (hndl.memBlockId >= 0 && hndl.SuperBufferHead != nullptr);
}

#endif // __IMAGE_LAB_GUIDED_FILTER_BUFFER_HANDLER__