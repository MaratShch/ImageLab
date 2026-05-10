#include <iomanip>
#include "Common.hpp"
#include "CompileTimeUtils.hpp"
#include "AlgoMemHandler.hpp"
#include "AFMedianFilterEnum.hpp"
#include "ImageLabMemInterface.hpp"


MemHandler alloc_memory_buffers(int32_t sizeX, int32_t sizeY, const bool dbgPrn)
{
    MemHandler algoMemHandler{};
    
    // Use the single source of truth from AlgoControls
    constexpr int32_t maxPadding = kernelRadiusMax; 
    algoMemHandler.maxPadding = maxPadding;

    // 1. Calculate physical dimensions (Logical Size + 2x Padding)
    int32_t physicalSizeX = sizeX + (maxPadding * 2);
    int32_t physicalSizeY = sizeY + (maxPadding * 2);

    constexpr int32_t cacheLine = static_cast<int32_t>(CACHE_LINE);

    // 2. Calculate the Cache-Aligned Stride using PHYSICAL width
    const int32_t rowBytes = physicalSizeX * sizeof(float);
    const int32_t alignedStrideBytes = CreateAlignment (rowBytes, cacheLine);
    
    // Save the stride in elements for Algorithm_Main
    algoMemHandler.strideY_Elements = alignedStrideBytes / sizeof(float);

    // 3. Calculate the total size for ONE color channel using PHYSICAL height
    const size_t channelSizeBytes = static_cast<size_t>(alignedStrideBytes) * physicalSizeY;

    // 4. Calculate the SuperBuffer total size.
    //    7 blocks supports both luma and RGB modes:
    //       proc_Y / proc_U / proc_V   -> 3 input planes  (Y/U/V or R/G/B)
    //       out_Y  / out_U  / out_V    -> 3 output planes (Y/U/V or R/G/B)
    //       scratch_Y                  -> 1 shared scratch plane (reused
    //                                     across channels in RGB mode since
    //                                     channels are processed sequentially)
    constexpr size_t kNumBlocks = 7;
    const     size_t totalSuperBufferSize = channelSizeBytes * kNumBlocks;

    // 5. One-Shot Allocation
    void* ptr = nullptr;
    const int32_t blockId = ::GetMemoryBlock(static_cast<int32_t>(totalSuperBufferSize), 0, &ptr);

    if (nullptr != ptr && blockId >= 0)
    {
        algoMemHandler.memBlockId = blockId;
        algoMemHandler.SuperBufferHead = reinterpret_cast<uint8_t*>(ptr);

        // 6. Slice the Arena into raw starting pointers for all 7 channels.
        //    Each block is exactly channelSizeBytes wide and (because
        //    alignedStrideBytes is a multiple of cacheLine) all block starts
        //    inherit the SuperBufferHead's alignment.
        float* raw_proc_Y = reinterpret_cast<float*>(algoMemHandler.SuperBufferHead + (channelSizeBytes * 0));
        float* raw_proc_U = reinterpret_cast<float*>(algoMemHandler.SuperBufferHead + (channelSizeBytes * 1));
        float* raw_proc_V = reinterpret_cast<float*>(algoMemHandler.SuperBufferHead + (channelSizeBytes * 2));
        float* raw_out_Y  = reinterpret_cast<float*>(algoMemHandler.SuperBufferHead + (channelSizeBytes * 3));
        float* raw_out_U  = reinterpret_cast<float*>(algoMemHandler.SuperBufferHead + (channelSizeBytes * 4));
        float* raw_out_V  = reinterpret_cast<float*>(algoMemHandler.SuperBufferHead + (channelSizeBytes * 5));
        float* raw_scr_Y  = reinterpret_cast<float*>(algoMemHandler.SuperBufferHead + (channelSizeBytes * 6));

        // 7. [THE PADDING SERVICE] Shift all 7 pointers identically.
        //    Move each pointer inwards by (maxPadding rows) and (maxPadding cols).
        //    Now index [0] natively points to the true (0,0) pixel of the image,
        //    and negative indices reach legally into the halo region for filter
        //    overreads at the borders.
        const int32_t paddingOffsetElements = (maxPadding * algoMemHandler.strideY_Elements) + maxPadding;

        algoMemHandler.proc_Y = raw_proc_Y + paddingOffsetElements;
        algoMemHandler.proc_U = raw_proc_U + paddingOffsetElements;
        algoMemHandler.proc_V = raw_proc_V + paddingOffsetElements;
        algoMemHandler.out_Y  = raw_out_Y  + paddingOffsetElements;
        algoMemHandler.out_U  = raw_out_U  + paddingOffsetElements;
        algoMemHandler.out_V  = raw_out_V  + paddingOffsetElements;
        algoMemHandler.scratch_Y = raw_scr_Y + paddingOffsetElements;
    }
    return algoMemHandler;
}

void free_memory_buffers (MemHandler& algoMemHandler)
{
    if (algoMemHandler.SuperBufferHead != nullptr)
    {
        ::FreeMemoryBlock(algoMemHandler.memBlockId);
    }

    // Zero out the struct to prevent Use-After-Free bugs
    algoMemHandler = {};
}