#ifndef __IMAGE_LAB_AWB_MEM_HANDLER__
#define __IMAGE_LAB_AWB_MEM_HANDLER__

#include <cstdint>
#include <cstddef>

// -----------------------------------------------------------------------------
// Planar linear RGB_32f plane view.
//
// Three CONTIGUOUS float planes, each sizeX*sizeY elements, in Rec.709 / sRGB
// primaries, scene-linear. No alpha: it is carried through entirely by the
// adapter layer (copied straight to the host output frame on egress).
// -----------------------------------------------------------------------------
struct RGBPlanes
{
    float* R;
    float* G;
    float* B;
};

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

    RGBPlanes input;    // decoded linear-RGB source  (filled by adapter; read-only to core)
    RGBPlanes output;   // balanced linear-RGB result (written by core; read by adapter)
    RGBPlanes scratch;  // internal ping/pong for iteration ({nullptr,...} when single-shot)
};

MemHandler alloc_memory_buffers (int32_t sizeX, int32_t sizeY, int32_t iterCnt = 1) noexcept;
void       free_memory_buffers  (MemHandler& mem) noexcept;
void       print_mem_handler    (const MemHandler& mem) noexcept;

inline bool mem_handler_valid (const MemHandler& h) noexcept
{
    return (nullptr != h.SuperBufferHead &&
            nullptr != h.input.R  && nullptr != h.input.G  && nullptr != h.input.B &&
            nullptr != h.output.R && nullptr != h.output.G && nullptr != h.output.B);
}

#endif // __IMAGE_LAB_AWB_MEM_HANDLER__