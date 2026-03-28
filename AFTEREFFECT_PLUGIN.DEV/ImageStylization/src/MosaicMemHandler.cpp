#define NOMINMAX
#include <algorithm> 
#include <iomanip>
#include "CompileTimeUtils.hpp"
#include "MosaicMemHandler.hpp"
#include "ImageLabMemInterface.hpp"


MemHandler alloc_memory_buffers (const int32_t sizeX, const int32_t sizeY, const int32_t K) 
{
    MemHandler algoMemHandler{};
    
    constexpr int32_t cacheLine = static_cast<int32_t>(CACHE_LINE);
    const int32_t frameSize = sizeX * sizeY;
    const int32_t rawFloatSize = frameSize * static_cast<int32_t>(sizeof(float));
    const int32_t rawInt32Size = frameSize * static_cast<int32_t>(sizeof(int32_t));

    // ==================================================================================
    // 1. CALCULATE ALIGNED SIZES
    // ==================================================================================

    const int32_t alignedR = CreateAlignment(rawFloatSize, cacheLine);
    const int32_t alignedG = CreateAlignment(rawFloatSize, cacheLine);
    const int32_t alignedB = CreateAlignment(rawFloatSize, cacheLine);

    const int32_t alignedL = CreateAlignment(rawInt32Size, cacheLine);
    const int32_t alignedD = CreateAlignment(rawFloatSize, cacheLine);
    const int32_t alignedCC= CreateAlignment(rawInt32Size, cacheLine);
    const int32_t aligned_D_CC = std::max(alignedD, alignedCC);
    
    const int32_t K_float = static_cast<int32_t>(static_cast<float>(K * sizeof(float))  * 1.25);
    const int32_t K_int32 = static_cast<int32_t>(static_cast<float>(K * sizeof(int32_t)) * 1.25);
    
    const int32_t aligned_spX = CreateAlignment(K_float, cacheLine);
    const int32_t aligned_spY = CreateAlignment(K_float, cacheLine);
    const int32_t aligned_spR = CreateAlignment(K_float, cacheLine);
    const int32_t aligned_spG = CreateAlignment(K_float, cacheLine);
    const int32_t aligned_spB = CreateAlignment(K_float, cacheLine);
    const int32_t aligned_spCount = CreateAlignment(K_int32, cacheLine);

    const int32_t aligned_accX = CreateAlignment(K_float, cacheLine);
    const int32_t aligned_accY = CreateAlignment(K_float, cacheLine);
    const int32_t aligned_accR = CreateAlignment(K_float, cacheLine);
    const int32_t aligned_accG = CreateAlignment(K_float, cacheLine);
    const int32_t aligned_accB = CreateAlignment(K_float, cacheLine);
    const int32_t aligned_accCount = CreateAlignment(K_int32, cacheLine);

    const int32_t aligned_bfsQueue = CreateAlignment(rawFloatSize, cacheLine);

    // ==================================================================================
    // 2. CALCULATE OFFSETS (THE STACK)
    // ==================================================================================
    size_t currentOffset = 0;
    const size_t off_alignedR = currentOffset; currentOffset += alignedR;
    const size_t off_alignedG = currentOffset; currentOffset += alignedG;
    const size_t off_alignedB = currentOffset; currentOffset += alignedB;
    const size_t off_alignedL = currentOffset; currentOffset += alignedL;
    const size_t off_alignedD = currentOffset; 
    const size_t off_alignedCC = currentOffset; currentOffset += aligned_D_CC;

    const size_t off_aligned_spX = currentOffset; currentOffset += aligned_spX;
    const size_t off_aligned_spY = currentOffset; currentOffset += aligned_spY;
    const size_t off_aligned_spR = currentOffset; currentOffset += aligned_spR;
    const size_t off_aligned_spG = currentOffset; currentOffset += aligned_spG;
    const size_t off_aligned_spB = currentOffset; currentOffset += aligned_spB;
    const size_t off_aligned_spCount = currentOffset; currentOffset += aligned_spCount;

    const size_t off_aligned_accX = currentOffset; currentOffset += aligned_accX;
    const size_t off_aligned_accY = currentOffset; currentOffset += aligned_accY;
    const size_t off_aligned_accR = currentOffset; currentOffset += aligned_accR;
    const size_t off_aligned_accG = currentOffset; currentOffset += aligned_accG;
    const size_t off_aligned_accB = currentOffset; currentOffset += aligned_accB;
    const size_t off_aligned_accCount = currentOffset; currentOffset += aligned_accCount;

    const size_t off_aligned_bfsQueue = currentOffset; currentOffset += aligned_bfsQueue;
 
    const size_t totalBytes = currentOffset;

    // ==================================================================================
    // 3. ALLOCATION & POINTER MAPPING
    // ==================================================================================
    void* ptr = nullptr;
    const int32_t blockId = ::GetMemoryBlock(static_cast<int32_t>(totalBytes), 0, &ptr);

    if (nullptr != ptr && blockId >= 0)
    {
        uint8_t* superBuffer = reinterpret_cast<uint8_t*>(ptr);

        algoMemHandler.SuperBufferHead = superBuffer; 
        algoMemHandler.memBlockId = blockId;
        algoMemHandler.totalSize = totalBytes;
        
        algoMemHandler.R_planar = reinterpret_cast<float*>(superBuffer + off_alignedR);
        algoMemHandler.G_planar = reinterpret_cast<float*>(superBuffer + off_alignedG);
        algoMemHandler.B_planar = reinterpret_cast<float*>(superBuffer + off_alignedB);

        algoMemHandler.L  = reinterpret_cast<int32_t*>(superBuffer + off_alignedL);
        algoMemHandler.D  = reinterpret_cast<float*>(superBuffer + off_alignedD);
        algoMemHandler.CC = reinterpret_cast<int32_t*>(superBuffer + off_alignedCC);
        
        algoMemHandler.sp_X = reinterpret_cast<float*>(superBuffer + off_aligned_spX);
        algoMemHandler.sp_Y = reinterpret_cast<float*>(superBuffer + off_aligned_spY);
        algoMemHandler.sp_R = reinterpret_cast<float*>(superBuffer + off_aligned_spR);
        algoMemHandler.sp_G = reinterpret_cast<float*>(superBuffer + off_aligned_spG);
        algoMemHandler.sp_B = reinterpret_cast<float*>(superBuffer + off_aligned_spB);
        algoMemHandler.sp_Count = reinterpret_cast<int32_t*>(superBuffer + off_aligned_spCount);

        algoMemHandler.acc_X = reinterpret_cast<float*>(superBuffer + off_aligned_accX);
        algoMemHandler.acc_Y = reinterpret_cast<float*>(superBuffer + off_aligned_accY);
        algoMemHandler.acc_R = reinterpret_cast<float*>(superBuffer + off_aligned_accR);
        algoMemHandler.acc_G = reinterpret_cast<float*>(superBuffer + off_aligned_accG);
        algoMemHandler.acc_B = reinterpret_cast<float*>(superBuffer + off_aligned_accB);
        algoMemHandler.acc_Count = reinterpret_cast<int32_t*>(superBuffer + off_aligned_accCount);

        algoMemHandler.bfs_Queue = reinterpret_cast<int32_t*>(superBuffer + off_aligned_bfsQueue);
    } // if (nullptr != ptr && blockId >= 0)

    return algoMemHandler;
}


void free_memory_buffers(MemHandler& algoMemHandler) 
{
    if (algoMemHandler.SuperBufferHead != nullptr && algoMemHandler.memBlockId >= 0)
    {
        ::FreeMemoryBlock(algoMemHandler.memBlockId);
    }

    // Zero out the struct to prevent Use-After-Free bugs
    algoMemHandler = {};
    algoMemHandler.memBlockId = -1;
    return;
}