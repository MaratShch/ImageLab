#include <iostream>
#include <iomanip>
#include "CompileTimeUtils.hpp"
#include "AlgoMemHandler.hpp" 
#include "ImageLabMemInterface.hpp"


MemHandler alloc_memory_buffers (const int32_t sizeX, const int32_t sizeY) noexcept
{
    MemHandler algoMemHandler{};
    
    constexpr int32_t cacheLine = static_cast<int32_t>(CACHE_LINE);

    // Round up to nearest multiple of 4
    const int32_t padX = CreateAlignment(sizeX, 4);
    const int32_t padY = CreateAlignment(sizeY, 4);
    const int32_t frameSize = padX * padY;

    algoMemHandler.padW = padX;
    algoMemHandler.padH = padY;
    
    // ==================================================================================
    // 1. CALCULATE ALIGNED SIZES
    // ==================================================================================
    const int32_t rawFloatSize = frameSize * sizeof(float);
    
    // Pyramid Scales
    const int32_t alignedFull    = CreateAlignment(rawFloatSize, cacheLine);
    const int32_t alignedHalf    = CreateAlignment(rawFloatSize / 4, cacheLine);
    const int32_t alignedQuarter = CreateAlignment(rawFloatSize / 16, cacheLine);

    // Sub-Block 2: Noise Oracle Sizes
    // 256 intensity levels * 16x16 matrix (65,536 floats per channel)
    const int32_t covLutSize = CreateAlignment(static_cast<int32_t>(256 * 16 * 16 * sizeof(float)), cacheLine);
    
    // Workspace needs enough space for means, sparse distances, and 4x4 DCT buffers
    // 4x full frame size is extremely safe for the grid step striding
    const int32_t oracleWorkspaceSize = CreateAlignment(alignedFull * 4, cacheLine);

    // Sub-Blocks 4 & 5: Bayesian Filter Sizes
    const int32_t poolSize = CreateAlignment(static_cast<int32_t>(512 * sizeof(PatchDistance)), cacheLine);
    const int32_t accumSize = alignedFull; // Size of full frame for 2D aggregation
    
    // Scratch3D: 16 (patch size) * 512 (max patches) * 6 (3 channels * 2 states) floats
    const int32_t scratch3DSize = CreateAlignment(static_cast<int32_t>(16 * 512 * 6 * sizeof(float)), cacheLine);

    // ==================================================================================
    // 2. CALCULATE OFFSETS (THE STACK)
    // ==================================================================================
    size_t currentOffset = 0;

    // L0: Full Resolution Planes & Differences
    const size_t off_Y_planar = currentOffset; currentOffset += alignedFull;
    const size_t off_U_planar = currentOffset; currentOffset += alignedFull;
    const size_t off_V_planar = currentOffset; currentOffset += alignedFull;
    const size_t off_Y_diff_full = currentOffset; currentOffset += alignedFull;
    const size_t off_U_diff_full = currentOffset; currentOffset += alignedFull;
    const size_t off_V_diff_full = currentOffset; currentOffset += alignedFull;

    // L1: Half Resolution Planes & Differences
    const size_t off_Y_half = currentOffset; currentOffset += alignedHalf;
    const size_t off_U_half = currentOffset; currentOffset += alignedHalf;
    const size_t off_V_half = currentOffset; currentOffset += alignedHalf;
    const size_t off_Y_diff_half = currentOffset; currentOffset += alignedHalf;
    const size_t off_U_diff_half = currentOffset; currentOffset += alignedHalf;
    const size_t off_V_diff_half = currentOffset; currentOffset += alignedHalf;

    // L2: Quarter Resolution Planes
    const size_t off_Y_quart = currentOffset; currentOffset += alignedQuarter;
    const size_t off_U_quart = currentOffset; currentOffset += alignedQuarter;
    const size_t off_V_quart = currentOffset; currentOffset += alignedQuarter;

    // Oracle Covariance LUTs and Workspace
    const size_t off_NoiseCov_Y = currentOffset; currentOffset += covLutSize;
    const size_t off_NoiseCov_U = currentOffset; currentOffset += covLutSize;
    const size_t off_NoiseCov_V = currentOffset; currentOffset += covLutSize;
    const size_t off_OracleWorkspace = currentOffset; currentOffset += oracleWorkspaceSize;

    // Bayesian Filter Accums and Workspaces
    const size_t off_SearchPool = currentOffset; currentOffset += poolSize;
    const size_t off_Accum_Y = currentOffset; currentOffset += accumSize;
    const size_t off_Accum_U = currentOffset; currentOffset += accumSize;
    const size_t off_Accum_V = currentOffset; currentOffset += accumSize;
    const size_t off_Weight_Count = currentOffset; currentOffset += accumSize;
    const size_t off_Scratch3D = currentOffset; currentOffset += scratch3DSize;

    const size_t totalBytes = currentOffset;

    // ==================================================================================
    // 3. ALLOCATION & POINTER MAPPING
    // ==================================================================================
    void* ptr = nullptr;
    const int32_t blockId = ::GetMemoryBlock (static_cast<int32_t>(totalBytes), 0, &ptr);
    uint8_t* superBuffer = reinterpret_cast<uint8_t*>(ptr);

    if (ptr != nullptr)
    {
        algoMemHandler.memBlockId = static_cast<int64_t>(blockId);
        // Store Head for Deallocation
        algoMemHandler.SuperBufferHead = superBuffer; 
        algoMemHandler.totalSize = totalBytes;

        // Pyramid Mappings
        algoMemHandler.Y_planar = reinterpret_cast<float*>(superBuffer + off_Y_planar);
        algoMemHandler.U_planar = reinterpret_cast<float*>(superBuffer + off_U_planar);
        algoMemHandler.V_planar = reinterpret_cast<float*>(superBuffer + off_V_planar);
        algoMemHandler.Y_diff_full = reinterpret_cast<float*>(superBuffer + off_Y_diff_full);
        algoMemHandler.U_diff_full = reinterpret_cast<float*>(superBuffer + off_U_diff_full);
        algoMemHandler.V_diff_full = reinterpret_cast<float*>(superBuffer + off_V_diff_full);

        algoMemHandler.Y_half = reinterpret_cast<float*>(superBuffer + off_Y_half);
        algoMemHandler.U_half = reinterpret_cast<float*>(superBuffer + off_U_half);
        algoMemHandler.V_half = reinterpret_cast<float*>(superBuffer + off_V_half);
        algoMemHandler.Y_diff_half = reinterpret_cast<float*>(superBuffer + off_Y_diff_half);
        algoMemHandler.U_diff_half = reinterpret_cast<float*>(superBuffer + off_U_diff_half);
        algoMemHandler.V_diff_half = reinterpret_cast<float*>(superBuffer + off_V_diff_half);

        algoMemHandler.Y_quart = reinterpret_cast<float*>(superBuffer + off_Y_quart);
        algoMemHandler.U_quart = reinterpret_cast<float*>(superBuffer + off_U_quart);
        algoMemHandler.V_quart = reinterpret_cast<float*>(superBuffer + off_V_quart);

        // Oracle Mappings
        algoMemHandler.NoiseCov_Y = reinterpret_cast<float*>(superBuffer + off_NoiseCov_Y);
        algoMemHandler.NoiseCov_U = reinterpret_cast<float*>(superBuffer + off_NoiseCov_U);
        algoMemHandler.NoiseCov_V = reinterpret_cast<float*>(superBuffer + off_NoiseCov_V);
        algoMemHandler.OracleWorkspace = reinterpret_cast<float*>(superBuffer + off_OracleWorkspace);

        // Bayesian Filter Mappings
        algoMemHandler.SearchPool = reinterpret_cast<PatchDistance*>(superBuffer + off_SearchPool);
        algoMemHandler.Accum_Y = reinterpret_cast<float*>(superBuffer + off_Accum_Y);
        algoMemHandler.Accum_U = reinterpret_cast<float*>(superBuffer + off_Accum_U);
        algoMemHandler.Accum_V = reinterpret_cast<float*>(superBuffer + off_Accum_V);
        algoMemHandler.Weight_Count = reinterpret_cast<float*>(superBuffer + off_Weight_Count);
        algoMemHandler.Scratch3D = reinterpret_cast<float*>(superBuffer + off_Scratch3D);
    }
    
    return algoMemHandler;
}


void free_memory_buffers(MemHandler& algoMemHandler) noexcept
{
    if (algoMemHandler.SuperBufferHead != nullptr && algoMemHandler.memBlockId >= 0)
    {
        ::FreeMemoryBlock(algoMemHandler.memBlockId);
    }

    algoMemHandler = {};
    algoMemHandler.memBlockId = -1;

    return;
}