#include <iostream>
#include <iomanip>
#include "Common.hpp"
#include "CompileTimeUtils.hpp"
#include "AlgoMemHandler.hpp" 
#include "ImageLabMemInterface.hpp"

MemHandler alloc_memory_buffers(const int32_t sizeX, const int32_t sizeY) noexcept
{
    CACHE_ALIGN MemHandler algoMemHandler{};

    // 32-byte alignment is standard for AVX2 (256-bit registers)
    constexpr int32_t cacheLine = static_cast<int32_t>(CACHE_LINE);
    const int32_t frameSize = sizeX * sizeY;
    const int32_t rawFloatSize = frameSize * static_cast<int32_t>(sizeof(float));

    // ==================================================================================
    // 1. CALCULATE ALIGNED SIZES
    // ==================================================================================
    // Since all our buffers are planar float32 arrays of identical size, 
    // we only need to compute the aligned size once and apply it uniformly.
    const int32_t alignedPlane = CreateAlignment(rawFloatSize, cacheLine);

    // ==================================================================================
    // 2. CALCULATE OFFSETS (THE STACK)
    // ==================================================================================
    size_t currentOffset = 0;

    // --- Inputs ---
    const size_t off_in_Y    = currentOffset; currentOffset += alignedPlane;
    const size_t off_in_U    = currentOffset; currentOffset += alignedPlane;
    const size_t off_in_V    = currentOffset; currentOffset += alignedPlane;
    const size_t off_in_Mask = currentOffset; currentOffset += alignedPlane;

    // --- Outputs ---
    const size_t off_out_Y   = currentOffset; currentOffset += alignedPlane;
    const size_t off_out_U   = currentOffset; currentOffset += alignedPlane;
    const size_t off_out_V   = currentOffset; currentOffset += alignedPlane;

    // --- Intermediate Scratchpads ---
    const size_t off_temp_blur = currentOffset; currentOffset += alignedPlane;
    const size_t off_mean_I    = currentOffset; currentOffset += alignedPlane;
    const size_t off_mean_II   = currentOffset; currentOffset += alignedPlane;
    const size_t off_coef_a    = currentOffset; currentOffset += alignedPlane;
    const size_t off_coef_b    = currentOffset; currentOffset += alignedPlane;

    const size_t totalBytes = currentOffset;

    // ==================================================================================
    // 3. ALLOCATION & POINTER MAPPING
    // ==================================================================================
    void* pBlock = nullptr;

    const int32_t blockId = GetMemoryBlock(static_cast<int32_t>(totalBytes), 0, &pBlock);
    if (blockId < 0 || nullptr == pBlock)
        return algoMemHandler;

    uint8_t* superBuffer = reinterpret_cast<uint8_t*>(pBlock);

    algoMemHandler.SuperBufferHead = superBuffer; 
    algoMemHandler.memBlockId = blockId;
        
    // Stride is exactly sizeX because padding is applied at the END of the buffer,
    // not at the end of each row. Our 1D moving sums don't need row padding.
    algoMemHandler.strideElements = sizeX; 
        
    // --- Map Inputs ---
    algoMemHandler.in_Y    = reinterpret_cast<const float*>(superBuffer + off_in_Y);
    algoMemHandler.in_U    = reinterpret_cast<const float*>(superBuffer + off_in_U);
    algoMemHandler.in_V    = reinterpret_cast<const float*>(superBuffer + off_in_V);
    algoMemHandler.in_Mask = reinterpret_cast<const float*>(superBuffer + off_in_Mask);

    // --- Map Outputs ---
    algoMemHandler.out_Y   = reinterpret_cast<float*>(superBuffer + off_out_Y);
    algoMemHandler.out_U   = reinterpret_cast<float*>(superBuffer + off_out_U);
    algoMemHandler.out_V   = reinterpret_cast<float*>(superBuffer + off_out_V);

    // --- Map Scratchpads ---
    algoMemHandler.temp_blur = reinterpret_cast<float*>(superBuffer + off_temp_blur);
    algoMemHandler.mean_I    = reinterpret_cast<float*>(superBuffer + off_mean_I);
    algoMemHandler.mean_II   = reinterpret_cast<float*>(superBuffer + off_mean_II);
    algoMemHandler.coef_a    = reinterpret_cast<float*>(superBuffer + off_coef_a);
    algoMemHandler.coef_b    = reinterpret_cast<float*>(superBuffer + off_coef_b);

    return algoMemHandler;
}

void free_memory_buffers(MemHandler& algoMemHandler) noexcept
{
    // Safe on an already-zeroed handler and safe to call twice: the null test
    // covers both, so a double free cannot reach the pool.
    if (nullptr != algoMemHandler.SuperBufferHead)
        FreeMemoryBlock(static_cast<int32_t>(algoMemHandler.memBlockId));

    // Zero the whole structure, so a stale pointer cannot be dereferenced after the
    // free. Assigning a fresh zero-initialised aggregate rather than clearing field
    // by field, so a field added to the struct later cannot be missed here.
    algoMemHandler = MemHandler{};

    return;
}