#ifndef __IMAGE_LAB_AWB_MEM_HANDLER__
#define __IMAGE_LAB_AWB_MEM_HANDLER__

#include <cstdint>
#include <cstddef>


// -----------------------------------------------------------------------------
// All algorithm-owned working memory. Nothing here comes from the Adobe engine;
// every plane is allocated explicitly for the algorithm and freed in one place.
//
// Integration flow:
//   1. mem = alloc_memory_buffers(sizeX, sizeY, ctrl.sliderIterCnt);
//   2. adapter: decode host frame  ->  mem.input   (BGRA/ARGB/VUYA -> linear RGB)
//   3. Algorithm_Main(mem, sizeX, sizeY, ctrl);     // mem.input -> mem.output
//   4. adapter: encode mem.output  ->  host frame   (linear RGB -> BGRA/ARGB/VUYA)
//   5. free_memory_buffers(mem);
//
// 'input' and 'output' are always allocated. 'scratch' is allocated only when
// sliderIterCnt > 1; otherwise its planes are nullptr and the core runs single-shot.
// All planes are backed by the single SuperBufferHead allocation.
// -----------------------------------------------------------------------------
struct MemHandler
{
    int64_t  memBlockId;
    size_t   totalMemory;
    uint8_t* SuperBufferHead;   // single backing allocation for all planes below

    float* input_f32_interleaved;
    float* output_f32_interleaved;
};

MemHandler alloc_memory_buffers (int32_t sizeX, int32_t sizeY) noexcept;
void       free_memory_buffers  (MemHandler& mem) noexcept;


inline bool mem_handler_valid (const MemHandler& mem) noexcept
{
    return (mem.memBlockId >= 0 && nullptr != mem.SuperBufferHead);
}

#endif // __IMAGE_LAB_AWB_MEM_HANDLER__