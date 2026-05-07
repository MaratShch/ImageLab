#include "Common.hpp"
#include "CompileTimeUtils.hpp"
#include "AlgoMemHandler.hpp"
#include "AFMedianFilterEnum.hpp"
#include "ImageLabMemInterface.hpp"

MemHandler alloc_memory_buffers (int32_t sizeX, int32_t sizeY, const bool dbgPrn)
{
    MemHandler algoMemHandler{};
    
    // Use the single source of truth from AlgoControls
    constexpr int32_t maxPadding = kernelRadiusMax; 
    algoMemHandler.maxPadding = maxPadding;

    // 1. Calculate physical dimensions (Logical Size + 2x Padding)
    int32_t physicalSizeX = sizeX + (maxPadding * 2);
    int32_t physicalSizeY = sizeY + (maxPadding * 2);

    constexpr int32_t cacheLine = static_cast<int32_t>(CACHE_LINE); // Usually 64

    // 2. Calculate the Cache-Aligned Stride using PHYSICAL width
    const int32_t rowBytes = physicalSizeX * sizeof(float);
    const int32_t alignedStrideBytes = CreateAlignment (rowBytes, cacheLine);
    
    // Save the stride in elements for Algorithm_Main
    algoMemHandler.strideY_Elements = alignedStrideBytes / sizeof(float);

    // 3. Calculate the total size for ONE color channel using PHYSICAL height
    const size_t channelSizeBytes = static_cast<size_t>(alignedStrideBytes) * physicalSizeY;

    // 4. Calculate the SuperBuffer total size
    // We now need 5 blocks: proc_Y, proc_U, proc_V, out_Y, scratch_Y
    const size_t totalSuperBufferSize = channelSizeBytes * 5;

    // 5. One-Shot Allocation
    void* ptr = nullptr;
    const int32_t blockId = ::GetMemoryBlock(static_cast<int32_t>(totalSuperBufferSize), 0, &ptr);
    uint8_t* superBuffer = reinterpret_cast<uint8_t*>(ptr);

    if (blockId >= 0 && superBuffer != nullptr)
    {
        algoMemHandler.memBlockId = static_cast<int64_t>(blockId);
        // Store Head for Deallocation
        algoMemHandler.SuperBufferHead = superBuffer;

        // 6. Slice the Arena into raw starting pointers for all 5 channels
        float* raw_Y = reinterpret_cast<float*>(algoMemHandler.SuperBufferHead);
        float* raw_U = reinterpret_cast<float*>(algoMemHandler.SuperBufferHead + channelSizeBytes);
        float* raw_V = reinterpret_cast<float*>(algoMemHandler.SuperBufferHead + (channelSizeBytes * 2));
        float* raw_out = reinterpret_cast<float*>(algoMemHandler.SuperBufferHead + (channelSizeBytes * 3));
        float* raw_scr = reinterpret_cast<float*>(algoMemHandler.SuperBufferHead + (channelSizeBytes * 4));

        // 7. [THE PADDING SERVICE] Shift all 5 pointers identically!
        // Move the pointers inwards by (Radius rows) and (Radius columns).
        // Now index [0] natively points to the true (0,0) pixel of the image.
        int32_t paddingOffsetElements = (maxPadding * algoMemHandler.strideY_Elements) + maxPadding;

        algoMemHandler.proc_Y = raw_Y + paddingOffsetElements;
        algoMemHandler.proc_U = raw_U + paddingOffsetElements;
        algoMemHandler.proc_V = raw_V + paddingOffsetElements;
        algoMemHandler.out_Y = raw_out + paddingOffsetElements;
        algoMemHandler.scratch_Y = raw_scr + paddingOffsetElements;
    }

    return algoMemHandler;
}

void free_memory_buffers (MemHandler& algoMemHandler)
{
    if (algoMemHandler.SuperBufferHead != nullptr && algoMemHandler.memBlockId >= 0)
    {
        ::FreeMemoryBlock(algoMemHandler.memBlockId);
    }

    algoMemHandler = {};
    algoMemHandler.memBlockId = -1;

    return;
}