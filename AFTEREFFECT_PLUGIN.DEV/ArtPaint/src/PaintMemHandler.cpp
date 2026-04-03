#include <iostream>
#include <iomanip>
#include <algorithm>
#include "CompileTimeUtils.hpp"
#include "PaintMemHandler.hpp"
#include "ImageLabMemInterface.hpp"


MemHandler alloc_memory_buffers(const int32_t sizeX, const int32_t sizeY, const bool dbgPrn) noexcept
{
    MemHandler algoMemHandler{};
    
    constexpr int32_t cacheLine = static_cast<int32_t>(CACHE_LINE);
    const int32_t frameSize = sizeX * sizeY;
    const int32_t rawFloatSize = frameSize * static_cast<int32_t>(sizeof(float));

    const int32_t rawFloatAlignedSize = CreateAlignment(rawFloatSize, cacheLine);

    // ==================================================================================
    // 1. CALCULATE PHASE 4 SIZES (GRAPH FLAT EDGE-LIST)
    // ==================================================================================
    algoMemHandler.max_edges = static_cast<size_t>(frameSize * 10);
    
    const size_t rawEdgeIdxSize = algoMemHandler.max_edges * sizeof(A_long);
    const size_t rawEdgeWeightSize = algoMemHandler.max_edges * sizeof(float);
    
    const size_t alignedEdgeIdxSize = CreateAlignment(static_cast<int32_t>(rawEdgeIdxSize), cacheLine);
    const size_t alignedEdgeWeightSize = CreateAlignment(static_cast<int32_t>(rawEdgeWeightSize), cacheLine);

    // ==================================================================================
    // 2. CALCULATE OFFSETS (THE STACK)
    // ==================================================================================
    size_t currentOffset = 0;
    const size_t off_Y_planar = currentOffset; currentOffset += rawFloatAlignedSize;
    const size_t off_U_planar = currentOffset; currentOffset += rawFloatAlignedSize;
    const size_t off_V_planar = currentOffset; currentOffset += rawFloatAlignedSize;
    
    // Phase 1 Offsets (gX and gY are officially gone)
    const size_t off_tensorA  = currentOffset; currentOffset += rawFloatAlignedSize;
    const size_t off_tensorB  = currentOffset; currentOffset += rawFloatAlignedSize;
    const size_t off_tensorC  = currentOffset; currentOffset += rawFloatAlignedSize;
    
    // Phase 2 Offsets
    const size_t off_tensorA_sm = currentOffset; currentOffset += rawFloatAlignedSize;
    const size_t off_tensorB_sm = currentOffset; currentOffset += rawFloatAlignedSize;
    const size_t off_tensorC_sm = currentOffset; currentOffset += rawFloatAlignedSize;
    const size_t off_tmpBlur    = currentOffset; currentOffset += rawFloatAlignedSize;
    
    // Phase 3 Offsets
    const size_t off_Lambda1  = currentOffset; currentOffset += rawFloatAlignedSize;
    const size_t off_Lambda2  = currentOffset; currentOffset += rawFloatAlignedSize;
    const size_t off_EigVectX = currentOffset; currentOffset += rawFloatAlignedSize;
    const size_t off_EigVectY = currentOffset; currentOffset += rawFloatAlignedSize;
    
    // Phase 4 Offsets 
    const size_t off_pI       = currentOffset; currentOffset += alignedEdgeIdxSize;
    const size_t off_pJ       = currentOffset; currentOffset += alignedEdgeIdxSize;
    const size_t off_pLogW    = currentOffset; currentOffset += alignedEdgeWeightSize;

    // Phase 5 Offsets
    const size_t off_imProc1  = currentOffset; currentOffset += rawFloatAlignedSize;
    const size_t off_imProc2  = currentOffset; currentOffset += rawFloatAlignedSize;
    
    const size_t totalBytes = currentOffset;
    
    // ==================================================================================
    // 3. ALLOCATION & POINTER MAPPING
    // ==================================================================================
    // Allocate ONE contiguous block.
    void* ptr = nullptr;
    const int32_t blockId = ::GetMemoryBlock(static_cast<int32_t>(totalBytes), 0, &ptr);

    if (ptr != nullptr)
    {
        uint8_t* superBuffer = reinterpret_cast<uint8_t*>(ptr);
        // Store Head for Deallocation
        algoMemHandler.SuperBufferHead = superBuffer;
        algoMemHandler.memBlockId = static_cast<int64_t>(blockId);

        algoMemHandler.Y_planar = reinterpret_cast<float*>(superBuffer + off_Y_planar);
        algoMemHandler.U_planar = reinterpret_cast<float*>(superBuffer + off_U_planar);
        algoMemHandler.V_planar = reinterpret_cast<float*>(superBuffer + off_V_planar);

        // Phase 1 Pointers
        algoMemHandler.tensorA  = reinterpret_cast<float*>(superBuffer + off_tensorA);
        algoMemHandler.tensorB  = reinterpret_cast<float*>(superBuffer + off_tensorB);
        algoMemHandler.tensorC  = reinterpret_cast<float*>(superBuffer + off_tensorC);
        
        // Phase 2 Pointers
        algoMemHandler.tensorA_sm = reinterpret_cast<float*>(superBuffer + off_tensorA_sm);
        algoMemHandler.tensorB_sm = reinterpret_cast<float*>(superBuffer + off_tensorB_sm);
        algoMemHandler.tensorC_sm = reinterpret_cast<float*>(superBuffer + off_tensorC_sm);
        algoMemHandler.tmpBlur    = reinterpret_cast<float*>(superBuffer + off_tmpBlur);

        // Phase 3 Pointers
        algoMemHandler.Lambda1  = reinterpret_cast<float*>(superBuffer + off_Lambda1);
        algoMemHandler.Lambda2  = reinterpret_cast<float*>(superBuffer + off_Lambda2);
        algoMemHandler.EigVectX = reinterpret_cast<float*>(superBuffer + off_EigVectX);
        algoMemHandler.EigVectY = reinterpret_cast<float*>(superBuffer + off_EigVectY);

        // Phase 4 Pointers
        algoMemHandler.pI_arena    = reinterpret_cast<A_long*>(superBuffer + off_pI);
        algoMemHandler.pJ_arena    = reinterpret_cast<A_long*>(superBuffer + off_pJ);
        algoMemHandler.pLogW_arena = reinterpret_cast<float*>(superBuffer + off_pLogW);

        // Phase 5 Pointers
        algoMemHandler.imProc1 = reinterpret_cast<float*>(superBuffer + off_imProc1);
        algoMemHandler.imProc2 = reinterpret_cast<float*>(superBuffer + off_imProc2);
    }
    
    return algoMemHandler;
}


void free_memory_buffers (MemHandler& algoMemHandler) noexcept
{
    if (algoMemHandler.SuperBufferHead != nullptr) 
    {
        ::FreeMemoryBlock (algoMemHandler.memBlockId);
    }
    
    // Zero out the struct to prevent Use-After-Free bugs
    algoMemHandler = {};
    algoMemHandler.memBlockId = -1;

    return;
}