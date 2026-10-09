#ifndef __IMAGE_LAB_GUIDED_FILTER_BUFFER_HANDLER__
#define __IMAGE_LAB_GUIDED_FILTER_BUFFER_HANDLER__

#include <cstdint>
#include "Common.hpp"

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
    const float* RESTRICT in_Y;
    const float* RESTRICT in_U;
    const float* RESTRICT in_V;
    const float* RESTRICT in_Mask;

    // --- OUTGOING BUFFERS (float32) ---
    // Processed Orthonormal YUV planes
    float* RESTRICT out_Y;
    float* RESTRICT out_U;
    float* RESTRICT out_V;

    // --- INTERMEDIATE SCRATCHPAD BUFFERS (float32) ---
    // Used by the Guided Filter math. Reused sequentially for Y, U, and V processing.
    float* RESTRICT temp_blur; // Holds the 1D intermediate pass for the separable box blur
    float* RESTRICT mean_I;    // Holds the blurred input plane
    float* RESTRICT mean_II;   // Holds the blurred squared input plane
    float* RESTRICT coef_a;    // Holds the variance/'a' coefficient, then overwritten with mean_a
    float* RESTRICT coef_b;    // Holds the 'b' coefficient, then overwritten with mean_b
};

MemHandler alloc_memory_buffers (int32_t sizeX, int32_t sizeY) noexcept;
void free_memory_buffers (MemHandler& algoMemHandler) noexcept;


inline bool mem_handler_valid(const MemHandler& hndl) noexcept
{
    return (hndl.memBlockId >= 0 && hndl.SuperBufferHead != nullptr) ? true : false;
}

#endif // __IMAGE_LAB_GUIDED_FILTER_BUFFER_HANDLER__