#include <iomanip>
#include "AlgoMemHandler.hpp"
#include "CompileTimeUtils.hpp"
#include "ImageLabMemInterface.hpp"

MemHandler alloc_memory_buffers(int32_t sizeX, int32_t sizeY, const bool dbgPrn)
{
    MemHandler algoMemHandler{};
    
    // ==================================================================================
    // 1. CALCULATE SIZES (BYTE ALIGNMENT MATH)
    // ==================================================================================
    
    const int32_t frameSize = sizeX * sizeY;
    const int32_t cacheLine = static_cast<int32_t>(CACHE_LINE);

    // --- Image Buffers (Per Pixel) ---
    
    // 1. Standard Float Plane (4 bytes/px) 
    // Used for: Source L, Density Map, Output L
    const int32_t rawFloatSize   = frameSize * sizeof(float);
    const int32_t alignedFloat   = CreateAlignment(rawFloatSize, cacheLine);
    
    // 2. Big Buffer (12 bytes/px) 
    // Used for: Edges (Float), JFA (Struct), Canvas (Float x3), Output AB (Float x2)
    // We assume sizeof(JFAPixel) == 12 and sizeof(float)*3 == 12.
    // Note: Output AB only needs 8 bytes/px, fitting easily into this 12 byte/px slot.
    const int32_t rawBigSize     = frameSize * 12; 
    const int32_t alignedBig     = CreateAlignment(rawBigSize, cacheLine);

    // --- Dot Buffers (Per Dot) ---
    
    // Logic: (250 dots / 10000 px) * 7.0 (Max Slider) = 0.175 dots/px.
    constexpr float max_density_factor = static_cast<float>((250.0 / 10000.0) * 7.0);
    
    // Calculate Max Dots with 15% safety padding
    const int32_t max_dots = static_cast<int32_t>((static_cast<float>(frameSize) * max_density_factor * 1.15f) + 0.5f);

    // Point2D (8 bytes)
    const int32_t rawPointSize   = max_dots * sizeof(Point2D);
    const int32_t alignedPoint   = CreateAlignment(rawPointSize, cacheLine);

    // Accumulators (4 bytes)
    const int32_t rawAccSize     = max_dots * sizeof(float);
    const int32_t alignedAcc     = CreateAlignment(rawAccSize, cacheLine);

    // Scratch Complex (Mixed types for Phase 4)
    // 3 floats (acc_L,a,b) + 1 int (count) + 1 struct (avg_color = 3 floats)
    // Total 7 * 4 bytes = 28 bytes per dot.
    const int32_t rawScratchSize = (rawAccSize * 3) + (max_dots * sizeof(int32_t)) + (max_dots * sizeof(fCIELabPix));
    const int32_t alignedScratch = CreateAlignment(rawScratchSize, cacheLine);

    // Node Pool (Quadtree - Phase 2)
    const int32_t nodeElemNumber = frameSize / 16;
    const int32_t rawNodeSize    = static_cast<int32_t>(sizeof(FlatQuadNode) * nodeElemNumber);
    const int32_t alignedNode    = CreateAlignment(rawNodeSize, cacheLine);

    // ==================================================================================
    // 2. OFFSET MAPPING (BUILDING THE STACK)
    // ==================================================================================
    
    size_t currentOffset = 0;

    // 1. Source L (Permanent)
    const size_t off_L = currentOffset;
    currentOffset += alignedFloat;

    // 2. Source AB (Permanent - 2 planes)
    const size_t off_AB = currentOffset;
    currentOffset += (alignedFloat * 2);

    // 3. Shared Buffer A (4 bytes/px)
    // Lifecycle: Inverted Luma -> Density Map -> Output dst_L
    const size_t off_SharedA = currentOffset;
    currentOffset += alignedFloat;

    // 4. Shared Big Buffer 1 (12 bytes/px)
    // Lifecycle: Edges -> JFA Ping -> Voronoi Map -> Output dst_AB
    const size_t off_Big1 = currentOffset;
    currentOffset += alignedBig;

    // 5. Shared Big Buffer 2 (12 bytes/px)
    // Lifecycle: JFA Pong -> Canvas Lab
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
    
    // Allocate ONE contiguous block.
    void* ptr = nullptr;
    const int32_t blockId = ::GetMemoryBlock (static_cast<int32_t>(totalBytes), 0, &ptr);

    if (ptr != nullptr)
    {
        uint8_t* superBuffer = reinterpret_cast<uint8_t*>(ptr);
        // Store Head for Deallocation
        algoMemHandler.SuperBufferHead = superBuffer; 
        algoMemHandler.memBlockId = static_cast<int64_t>(blockId);

        // ==================================================================================
        // 4. POINTER MAPPING (ARITHMETIC)
        // ==================================================================================

        // --- Permanent Inputs ---
        algoMemHandler.L  = reinterpret_cast<float*>(superBuffer + off_L);
        algoMemHandler.ab = reinterpret_cast<float*>(superBuffer + off_AB);

        // --- Physical Shared Bases ---
        float*   pSharedA = reinterpret_cast<float*>(superBuffer + off_SharedA);
        uint8_t* pBig1    = superBuffer + off_Big1;
        uint8_t* pBig2    = superBuffer + off_Big2;

        // --- Phase 1 & 2 (Density Logic) ---
        algoMemHandler.Luma1      = pSharedA; // Step 1.2 Dest
        algoMemHandler.Luma2      = reinterpret_cast<float*>(pBig1); // Step 1.3 Dest (Edges) reusing Big1
        algoMemHandler.DencityMap = pSharedA; // Step 1.4 Dest (Overwrites Luma1)

        // --- Phase 2 (Seeding) ---
        algoMemHandler.NodePool       = reinterpret_cast<FlatQuadNode*>(superBuffer + off_Nodes);
        algoMemHandler.NodeElemNumber = nodeElemNumber;
        algoMemHandler.PointOut       = reinterpret_cast<Point2D*>(superBuffer + off_Points);

        // --- Phase 3 (Refinement - PingPong) ---
        algoMemHandler.JfaBufferPing = reinterpret_cast<JFAPixel*>(pBig1); // Reuse Big1
        algoMemHandler.JfaBufferPong = reinterpret_cast<JFAPixel*>(pBig2); // Reuse Big2
        
        algoMemHandler.AccumX = reinterpret_cast<float*>(superBuffer + off_AccX);
        algoMemHandler.AccumY = reinterpret_cast<float*>(superBuffer + off_AccY);
        algoMemHandler.AccumW = reinterpret_cast<float*>(superBuffer + off_AccW);

        // --- Phase 4 (Rendering) ---
        // CRITICAL: Canvas overwrites JFA Pong.
        algoMemHandler.CanvasLab = reinterpret_cast<float*>(pBig2); 

        // --- Phase 5 (Output Re-Use) ---
        // 1. dst_L reuses the Density Map buffer (pSharedA).
        //    (4 bytes/px required, 4 bytes/px available).
        algoMemHandler.dst_L = pSharedA;

        // 2. dst_ab reuses Big Buffer 1 (pBig1).
        //    (8 bytes/px required, 12 bytes/px available).
        //    Note: Big1 was used for JFA Ping/Edges, which are dead by Phase 5.
        algoMemHandler.dst_ab = reinterpret_cast<float*>(pBig1);

        // --- Scratch Internal Mapping ---
        uint8_t* ptr = superBuffer + off_Scratch;
        algoMemHandler.Scratch.acc_L      = reinterpret_cast<float*>(ptr);      ptr += alignedAcc; // Reuse calced size
        algoMemHandler.Scratch.acc_a      = reinterpret_cast<float*>(ptr);      ptr += alignedAcc;
        algoMemHandler.Scratch.acc_b      = reinterpret_cast<float*>(ptr);      ptr += alignedAcc;
        algoMemHandler.Scratch.acc_count  = reinterpret_cast<int32_t*>(ptr);    ptr += (max_dots * sizeof(int32_t));
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
            std::cout << "Bytes Per Pixel:  " << bpp << "\n";
            std::cout << "Max Dots Capacity:" << max_dots << "\n\n";
            
            std::cout << "Sub-Buffer Layout [Size | Offset | Address]:\n";
            
            // Lambda for clean printing
            auto printBuf = [&](const char* name, int32_t size, size_t offset, void* ptr) {
                std::cout << std::left << std::setw(15) << name 
                          << " | Sz: " << std::setw(10) << size 
                          << " | Off: " << std::setw(10) << offset 
                          << " | Ptr: 0x" << std::hex << reinterpret_cast<uintptr_t>(ptr) << std::dec << "\n";
            };

            printBuf("1. Source L",   alignedFloat, off_L, algoMemHandler.L);
            printBuf("2. Source AB",  alignedFloat*2, off_AB, algoMemHandler.ab);
            printBuf("3. Density",    alignedFloat, off_SharedA, algoMemHandler.DencityMap);
            printBuf("4. BigBuf1",    alignedBig, off_Big1, pBig1);
            printBuf("5. BigBuf2",    alignedBig, off_Big2, pBig2);
            printBuf("6. Nodes",      alignedNode, off_Nodes, algoMemHandler.NodePool);
            printBuf("7. Points",     alignedPoint, off_Points, algoMemHandler.PointOut);
            printBuf("8. Scratch",    alignedScratch, off_Scratch, algoMemHandler.Scratch.acc_L);
            printBuf("9. Output L",   alignedFloat, off_SharedA, algoMemHandler.dst_L);
            printBuf("10.Output AB",  alignedBig, off_Big1, algoMemHandler.dst_ab);
            
            std::cout << "=========================================\n";
        }
    }
    else
    {
        // Allocation failed
        algoMemHandler = {};
    }

    return algoMemHandler;
}


void free_memory_buffers(MemHandler& algoMemHandler)
{
    // PRO-LEVEL CLEANUP:
    // We only free the Master Block. 
    // All other pointers are just offsets into this block and die automatically.

    if (algoMemHandler.SuperBufferHead != nullptr && algoMemHandler.memBlockId >= 0)
    {
        ::FreeMemoryBlock (algoMemHandler.memBlockId);
    }

    // Zero out the struct to prevent Use-After-Free bugs
    algoMemHandler = {};
    algoMemHandler.memBlockId = -1;

    return;
}