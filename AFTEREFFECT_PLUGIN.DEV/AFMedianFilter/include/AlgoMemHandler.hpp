#ifndef __IMAGE_LAB_ART_POINTILISM_MEMORY_BUFFERS_HANDLER__
#define __IMAGE_LAB_ART_POINTILISM_MEMORY_BUFFERS_HANDLER__

#include <cstdint>

struct MemHandler
{
    int64_t memBlockId;
    uint8_t* SuperBufferHead;

    // Incoming (Read-Only during processing)
    float* proc_Y;
    float* proc_U;
    float* proc_V;

    // Outgoing / Temporary
    float* out_Y;
    float* scratch_Y;

    // Metadata
    int32_t strideY_Elements; // The cache-aligned width of the physical memory row
    int32_t maxPadding;       // The worst-case radius from the UI
};


MemHandler alloc_memory_buffers (int32_t sizeX, int32_t sizeY, const bool dbgPrn = false);
void free_memory_buffers (MemHandler& algoMemHandler);

inline bool mem_handler_valid(const MemHandler& hndl) noexcept
{
    return (hndl.memBlockId >= 0 && hndl.SuperBufferHead != nullptr) ? true : false;
}

#endif // __IMAGE_LAB_ART_POINTILISM_MEMORY_BUFFERS_HANDLER__ 