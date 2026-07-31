#include "Common.hpp"
#include "CompileTimeUtils.hpp"
#include "AlgoMemHandler.hpp"
#include "ImageLabMemInterface.hpp"

MemHandler alloc_memory_buffers (const int32_t sizeX, const int32_t sizeY) 
{
    MemHandler h{};

    if (sizeX <= 0 || sizeY <= 0)
        return h;

    return h;
}

void free_memory_buffers (MemHandler& mem) 
{
    mem = {};   // zero out to prevent use-after-free
}