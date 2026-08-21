#include "Common.hpp"
#include "CompileTimeUtils.hpp" 
#include "AlgoMemHandler.hpp"
#include "ImageLabMemInterface.hpp"

// Allocates input(3) + output(3) [+ scratch(3) when iterating] cache-aligned planes
// from one backing block. Single-shot needs no scratch.
MemHandler alloc_memory_buffers (const int32_t sizeX, const int32_t sizeY) noexcept
{
    MemHandler h{};

    if (sizeX <= 0 || sizeY <= 0)
        return h;

    size_t totalBytes = 0ull;
    size_t srcBufOffset = 0ull;
    size_t dstBufOffset = 0ull;

    const size_t frameSize = static_cast<int64_t>(sizeX) * static_cast<int64_t>(sizeY);
    const size_t frameMemSize = frameSize * sizeof(float) * 3ull; // RGB - f32 per every channel
    const size_t frameMemSizeAligned = CreateAlignment (frameMemSize, static_cast<size_t>(CACHE_LINE));     

    dstBufOffset += frameMemSizeAligned;
    totalBytes = dstBufOffset + frameMemSizeAligned;
        
    void* ptr = nullptr;
    const int32_t blockId = ::GetMemoryBlock(static_cast<int32_t>(totalBytes), 0, &ptr);

    if (nullptr != ptr && blockId >= 0)
    {
        h.SuperBufferHead = static_cast<uint8_t*>(ptr);
        h.totalMemory = totalBytes;
        h.memBlockId = blockId;

        h.input_f32_interleaved  = reinterpret_cast<float*>(h.SuperBufferHead);
        h.output_f32_interleaved = reinterpret_cast<float*>(h.SuperBufferHead + dstBufOffset);
    }

    return h;
}

void free_memory_buffers (MemHandler& mem) noexcept
{
    if (nullptr != mem.SuperBufferHead && mem.memBlockId >= 0)
        ::FreeMemBlock (mem.memBlockId);

    mem = {};   // zero out to prevent use-after-free
}
