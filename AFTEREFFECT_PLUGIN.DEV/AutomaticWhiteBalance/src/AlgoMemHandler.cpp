#include "Common.hpp"
#include "CompileTimeUtils.hpp"
#include "AlgoMemHandler.hpp"
#include "ImageLabMemInterface.hpp" 

// Allocates input(3) + output(3) [+ scratch(3) when iterating] cache-aligned planes
// from one backing block. Single-shot needs no scratch.
MemHandler alloc_memory_buffers (const int32_t sizeX, const int32_t sizeY, const int32_t iterCnt) noexcept
{
    MemHandler h{};

    if (sizeX <= 0 || sizeY <= 0)
        return h;

    constexpr size_t cacheLine = static_cast<size_t>(CACHE_LINE); // usually 64
    const int64_t N          = static_cast<int64_t>(sizeX) * static_cast<int64_t>(sizeY);
    const size_t  planeBytes = static_cast<size_t>(N) * sizeof(float);

    const size_t  planeStride = CreateAlignment (planeBytes, cacheLine);

    const bool   iterate  = (iterCnt > 1);
    const int    nPlanes  = iterate ? 9 : 6;                          // in + out [+ scratch]
    const size_t total    = static_cast<size_t>(nPlanes) * planeStride;

    void* ptr = nullptr;
    const int32_t blockId = ::GetMemoryBlock(static_cast<int32_t>(total), 0, &ptr);

    if (nullptr != ptr && blockId >= 0)
    {
        h.SuperBufferHead = static_cast<uint8_t*>(ptr);
        h.totalMemory = total;
        h.memBlockId = blockId;

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

        if (iterate)
        {
            h.scratch.R = next(); h.scratch.G = next(); h.scratch.B = next();
        }
        else
        {
            h.scratch.R = h.scratch.G = h.scratch.B = nullptr;
        }
    }

    return h;
}

void free_memory_buffers (MemHandler& mem) noexcept
{
    if (nullptr != mem.SuperBufferHead && mem.memBlockId >= 0)
        FreeMemoryBlock(mem.memBlockId);

    mem = {};   // zero out to prevent use-after-free
}
