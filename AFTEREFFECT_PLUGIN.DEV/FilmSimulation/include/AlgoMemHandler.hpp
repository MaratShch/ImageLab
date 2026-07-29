#ifndef __IMAGE_LAB_AWB_MEM_HANDLER__
#define __IMAGE_LAB_AWB_MEM_HANDLER__

#include <cstdint>
#include <cstddef>


struct MemHandler
{
    int64_t  memBlockId;
    size_t   totalMemory;
    uint8_t* SuperBufferHead;   // single backing allocation for all planes below

};

MemHandler alloc_memory_buffers (int32_t sizeX, int32_t sizeY, int32_t iterCnt = 1) noexcept;
void       free_memory_buffers  (MemHandler& mem) noexcept;

inline bool mem_handler_valid (const MemHandler& h) noexcept
{
    return (nullptr != h.SuperBufferHead);
}

#endif // __IMAGE_LAB_AWB_MEM_HANDLER__