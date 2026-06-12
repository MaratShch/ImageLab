#include "Common.hpp"
#include "CompileTimeUtils.hpp"
#include "AlgoMemHandler.hpp"
#include "ImageLabMemInterface.hpp"

// Allocates input(3) + output(3) [+ scratch(3) when iterating] cache-aligned planes
// from one backing block. Single-shot needs no scratch.
MemHandler alloc_memory_buffers (const int32_t sizeX, const int32_t sizeY) noexcept
{
    MemHandler h{};

    const int64_t N          = static_cast<int64_t>(sizeX) * static_cast<int64_t>(sizeY);
    const size_t  planeBytes = static_cast<size_t>(N) * sizeof(float);
    const size_t  planeStride = CreateAlignment(planeBytes, static_cast<size_t>(CACHE_LINE));

    const int    nPlanes  = 6;                          // in + out [+ scratch]
    const size_t total    = static_cast<size_t>(nPlanes) * planeStride;

    void* ptr = nullptr;
    const int32_t blockId = ::GetMemoryBlock(static_cast<int32_t>(total), 0, &ptr);

    if (blockId >= 0 && nullptr != ptr)
    {
        h.totalMemory = total;
        h.memBlockId = blockId;
        h.SuperBufferHead = reinterpret_cast<uint8_t*>(ptr);

        uint8_t* const base = h.SuperBufferHead;
        size_t off = 0;
        auto next = [&base, &off, planeStride]() noexcept -> float*
        {
            float* q = reinterpret_cast<float*>(base + off);
            off += planeStride;
            return q;
        };

        h.input.R = next(); h.input.G = next(); h.input.B = next();
        h.output.R = next(); h.output.G = next(); h.output.B = next();
        h.scratch.R = h.scratch.G = h.scratch.B = nullptr;
    }

    return h;
}

void free_memory_buffers (MemHandler& mem) noexcept
{
    if (nullptr != mem.SuperBufferHead && mem.memBlockId >= 0)
        FreeMemoryBlock(mem.memBlockId);

    mem = {};   // zero out to prevent use-after-free
}
