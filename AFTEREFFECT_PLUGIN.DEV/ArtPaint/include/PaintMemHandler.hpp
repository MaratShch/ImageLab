#pragma once

#include <cstdint>
#include "Common.hpp"
#include "AefxDevPatch.hpp"


struct MemHandler
{
    int64_t memBlockId;
    uint8_t* RESTRICT SuperBufferHead;
    size_t totalSize;

    // Internal Planar Image Buffers (Range: 0.0f to 255.0f)
    float* Y_planar;
    float* U_planar;
    float* V_planar;

    // Phase 1: Initial Tensors (Fused - gX and gY are gone!)
    float* tensorA;
    float* tensorB;
    float* tensorC;
    
    // Phase 2: Smoothed Tensors
    float* tensorA_sm;
    float* tensorB_sm;
    float* tensorC_sm;
    float* tmpBlur; // Reusable buffer for separable horizontal convolution    

    // Phase 3: Diagonalization Tensors
    float* Lambda1;
    float* Lambda2;
    float* EigVectX;
    float* EigVectY;

    // Phase 4 Graph Construction (Flat Edge-List)
    size_t max_edges;    // Safety cap to prevent buffer overruns
    A_long* pI_arena;    // Pre-allocated Source pixel indices
    A_long* pJ_arena;    // Pre-allocated Target pixel indices
    float* pLogW_arena;  // Pre-allocated Log Weights

    // Phase 5: Morphology Ping-Pong Buffers
    float* imProc1;
    float* imProc2;
};

MemHandler alloc_memory_buffers (const int32_t sizeX, const int32_t sizeY, const bool dbgPrn = false) noexcept;
void free_memory_buffers (MemHandler& algoMemHandler) noexcept;

inline bool check_memory_buffers (const MemHandler& algoMemHandler) noexcept
{
    return (algoMemHandler.memBlockId >= 0 && nullptr != algoMemHandler.SuperBufferHead) ? true : false;
}
