#include <iomanip>
#include "AlgoMemHandler.hpp"
#include "CompileTimeUtils.hpp"
#include "ImageLabMemInterface.hpp"


MemHandler alloc_memory_buffers (int32_t sizeX, int32_t sizeY, const bool dbgPrn)
{
    CACHE_ALIGN MemHandler algoMemHandler{};
    
    // ==================================================================================
    // 1. CALCULATE SIZES (BYTE ALIGNMENT MATH)
    // ==================================================================================
    
    const int32_t frameSize = sizeX * sizeY;
    constexpr int32_t cacheLine = static_cast<int32_t>(CACHE_LINE);

    // --- Image Buffers ---
    // Float Plane (4 bytes/px)
    const int32_t rawFloatSize   = frameSize * sizeof(float);
    const int32_t alignedFloat   = CreateAlignment(rawFloatSize, cacheLine);
    
    // Big Buffer (12 bytes/px)
    const int32_t rawBigSize     = frameSize * 12; 
    const int32_t alignedBig     = CreateAlignment(rawBigSize, cacheLine);

    // --- Dot Buffers ---
    // (250 / 10000) * 7.0 = 0.175
    constexpr float max_density_factor = static_cast<float>((250.0 / 10000.0) * 7.0);
    const int32_t max_dots = static_cast<int32_t>((static_cast<float>(frameSize) * max_density_factor * 1.15f) + 0.5f);

    // Point2D (8 bytes)
    const int32_t rawPointSize   = max_dots * sizeof(Point2D);
    const int32_t alignedPoint   = CreateAlignment(rawPointSize, cacheLine);

    // Accumulators (4 bytes)
    const int32_t rawAccSize     = max_dots * sizeof(float);
    const int32_t alignedAcc     = CreateAlignment(rawAccSize, cacheLine);

    // Scratch Complex (Mixed types)
    const int32_t rawScratchSize = (rawAccSize * 3) + (max_dots * sizeof(int32_t)) + (max_dots * sizeof(fCIELabPix));
    const int32_t alignedScratch = CreateAlignment(rawScratchSize, cacheLine);

    // Node Pool (Quadtree)
    const int32_t nodeElemNumber = frameSize / 16;
    const int32_t rawNodeSize    = static_cast<int32_t>(sizeof(FlatQuadNode) * nodeElemNumber);
    const int32_t alignedNode    = CreateAlignment(rawNodeSize, cacheLine);

    // ==================================================================================
    // 2. OFFSET MAPPING (BUILDING THE STACK)
    // ==================================================================================
    
    size_t currentOffset = 0;

    // 1. Source L
    const size_t off_L = currentOffset;
    currentOffset += alignedFloat;

    // 2. Source AB (2x float planes)
    const size_t off_AB = currentOffset;
    currentOffset += (alignedFloat * 2);

    // 3. Density / Luma1
    const size_t off_Density = currentOffset;
    currentOffset += alignedFloat;

    // 4. Big Buffer 1 (Edges/Ping)
    const size_t off_Big1 = currentOffset;
    currentOffset += alignedBig;

    // 5. Big Buffer 2 (Pong/Canvas)
    const size_t off_Big2 = currentOffset;
    currentOffset += alignedBig;

    // 6. Node Pool
    const size_t off_Nodes = currentOffset;
    currentOffset += alignedNode;

    // 7. Point Out
    const size_t off_Points = currentOffset;
    currentOffset += alignedPoint;

    // 8. Accumulators (X, Y, W)
    const size_t off_AccX = currentOffset;
    currentOffset += alignedAcc;

    const size_t off_AccY = currentOffset;
    currentOffset += alignedAcc;

    const size_t off_AccW = currentOffset;
    currentOffset += alignedAcc;

    // 9. Scratch
    const size_t off_Scratch = currentOffset;
    currentOffset += alignedScratch;

    // Final Total Size
    const size_t totalBytes = currentOffset;

    // ==================================================================================
    // 3. THE SUPER ALLOCATION
    // ==================================================================================
    
    // Allocate ONE block.
    void* pMemoryBlock = nullptr;
    int32_t blockId = GetMemoryBlock(static_cast<int32_t>(totalBytes), 0, &pMemoryBlock);

    if (nullptr != pMemoryBlock && blockId >= 0)
    {
        algoMemHandler.memBlockId = static_cast<int64_t>(blockId);
        uint8_t* superBuffer = reinterpret_cast<uint8_t*>(pMemoryBlock);

        // Store the Head Pointer (Critical for Freeing later!)
        // I assume you added a void* member to MemHandler, e.g., 'SuperBufferHead'
        // If not, we can assume 'L' is the head, but explicit is safer.
        algoMemHandler.SuperBufferHead = superBuffer; 

        // ==================================================================================
        // 4. POINTER ASSIGNMENT (ARITHMETIC)
        // ==================================================================================

        // Permanent Source
        algoMemHandler.L  = reinterpret_cast<float*>(superBuffer + off_L);
        algoMemHandler.ab = reinterpret_cast<float*>(superBuffer + off_AB);

        // Aliased Maps
        float* pDensity   = reinterpret_cast<float*>(superBuffer + off_Density);
        uint8_t* pBig1    = superBuffer + off_Big1;
        uint8_t* pBig2    = superBuffer + off_Big2;

        algoMemHandler.Luma1      = pDensity;
        algoMemHandler.Luma2      = reinterpret_cast<float*>(pBig1); // Alias Big1
        algoMemHandler.DencityMap = pDensity; // Alias Density

        // Dot Data
        algoMemHandler.NodePool       = reinterpret_cast<FlatQuadNode*>(superBuffer + off_Nodes);
        algoMemHandler.NodeElemNumber = nodeElemNumber;
        algoMemHandler.PointOut       = reinterpret_cast<Point2D*>(superBuffer + off_Points);

        // Phase 3
        algoMemHandler.JfaBufferPing = reinterpret_cast<JFAPixel*>(pBig1); // Alias Big1
        algoMemHandler.JfaBufferPong = reinterpret_cast<JFAPixel*>(pBig2); // Alias Big2
        algoMemHandler.AccumX        = reinterpret_cast<float*>(superBuffer + off_AccX);
        algoMemHandler.AccumY        = reinterpret_cast<float*>(superBuffer + off_AccY);
        algoMemHandler.AccumW        = reinterpret_cast<float*>(superBuffer + off_AccW);

        // Phase 4
        algoMemHandler.CanvasLab = reinterpret_cast<float*>(pBig2); // Alias Big2

        // Scratch Internal Mapping
        uint8_t* pScratchBase = superBuffer + off_Scratch;
        uint8_t* ptr = pScratchBase;
        
        // Map sub-pointers inside the scratch region
        // Note: These do not need alignment padding between them if packed tight, 
        // but sticking to previous logic is safe.
        algoMemHandler.Scratch.acc_L      = reinterpret_cast<float*>(ptr); ptr += rawAccSize; // reusing calculated size
        algoMemHandler.Scratch.acc_a      = reinterpret_cast<float*>(ptr); ptr += rawAccSize;
        algoMemHandler.Scratch.acc_b      = reinterpret_cast<float*>(ptr); ptr += rawAccSize;
        algoMemHandler.Scratch.acc_count  = reinterpret_cast<int32_t*>(ptr); ptr += (max_dots * sizeof(int32_t));
        algoMemHandler.Scratch.avg_colors = reinterpret_cast<fCIELabPix*>(ptr);

        // ==================================================================================
        // 5. DEBUG PRINT
        // ==================================================================================
        if (true == dbgPrn)
        {
            const double totalMb = static_cast<double>(totalBytes) / 1000000.0;
            const double bpp = static_cast<double>(totalBytes) / static_cast<double>(frameSize);

            std::cout << "\n=== PRO-LEVEL SUPER-BUFFER ALLOCATION ===\n";
            std::cout << "Base Address:     0x" << std::hex << reinterpret_cast<uintptr_t>(superBuffer) << std::dec << "\n";
            std::cout << "Total Memory:     " << totalMb << " MB\n";
            std::cout << "Bytes Per Pixel:  " << bpp << "\n\n";
            
            std::cout << "Sub-Buffer Layout [Size | Offset | Address]:\n";
            
            auto printBuf = [&](const char* name, int32_t size, size_t offset, void* ptr) {
                std::cout << std::left << std::setw(15) << name 
                          << " | Sz: " << std::setw(10) << size 
                          << " | Off: " << std::setw(10) << offset 
                          << " | Ptr: 0x" << std::hex << reinterpret_cast<uintptr_t>(ptr) << std::dec << "\n";
            };

            printBuf("1. Source L",   alignedFloat, off_L, algoMemHandler.L);
            printBuf("2. Source AB",  alignedFloat*2, off_AB, algoMemHandler.ab);
            printBuf("3. Density",    alignedFloat, off_Density, algoMemHandler.DencityMap);
            printBuf("4. BigBuf1",    alignedBig, off_Big1, pBig1);
            printBuf("5. BigBuf2",    alignedBig, off_Big2, pBig2);
            printBuf("6. Nodes",      alignedNode, off_Nodes, algoMemHandler.NodePool);
            printBuf("7. Points",     alignedPoint, off_Points, algoMemHandler.PointOut);
            printBuf("8. AccumX",     alignedAcc, off_AccX, algoMemHandler.AccumX);
            printBuf("9. AccumY",     alignedAcc, off_AccY, algoMemHandler.AccumY);
            printBuf("10.AccumW",     alignedAcc, off_AccW, algoMemHandler.AccumW);
            printBuf("11.Scratch",    alignedScratch, off_Scratch, pScratchBase);
            
            std::cout << "=========================================\n";
        }
    }
    else
    {
        // Allocation failed
        algoMemHandler = {};
        algoMemHandler.memBlockId = -1;
    }

    return algoMemHandler;
}


void free_memory_buffers (MemHandler& algoMemHandler)
{
    // PRO-LEVEL CLEANUP:
    // We only free the Master Block. 
    // All other pointers are just offsets into this block and die automatically.

    if (algoMemHandler.memBlockId >= 0 &&  algoMemHandler.SuperBufferHead != nullptr)
    {
        const int32_t memBlockId = static_cast<int32_t>(algoMemHandler.memBlockId);
        FreeMemoryBlock (memBlockId);
    }

    // Zero out the struct to prevent Use-After-Free bugs
    algoMemHandler = {};
    algoMemHandler.memBlockId = -1;

    return;
}