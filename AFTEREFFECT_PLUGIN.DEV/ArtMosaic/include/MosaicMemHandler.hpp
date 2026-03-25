#pragma once

#include <cstdint>
#include "Common.hpp"

struct MemHandler
{
    int64_t memBlockId;
    uint8_t* RESTRICT SuperBufferHead;
    size_t totalSize;

    // Internal Planar Image Buffers (Range: 0.0f to 255.0f)
    float* R_planar;
    float* G_planar;
    float* B_planar;

    // SLIC Algorithm Buffers
    int32_t* L;    // Label buffer (Size: sizeX * sizeY)
    float* D;      // Distance buffer (Size: sizeX * sizeY)
    int32_t* CC;   // Used ONLY during enforceConnectivity at the end
    
    // Superpixel Data (Size: K. Flattened for AVX2 rather than structs)
    float* sp_X;
    float* sp_Y;
    float* sp_R;
    float* sp_G;
    float* sp_B;
    int32_t* sp_Count; // Used during the update step to average colors

    float* acc_X;
    float* acc_Y;
    float* acc_R;
    float* acc_G;
    float* acc_B;
    int32_t* acc_Count;
    
    int32_t* bfs_Queue;
};

MemHandler alloc_memory_buffers(const int32_t sizeX, const int32_t sizeY, const int32_t = 1000);
void free_memory_buffers(MemHandler& algoMemHandler);

inline bool mem_handler_valid(const MemHandler& hndl)
{
    return (hndl.memBlockId >= 0 && hndl.SuperBufferHead != nullptr) ? true : false;
}
