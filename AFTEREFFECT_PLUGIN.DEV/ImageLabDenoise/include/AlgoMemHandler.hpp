#ifndef __IMAGE_LAB_ALGO_MEMORY_BUFFERS_HANDLER__
#define __IMAGE_LAB_ALGO_MEMORY_BUFFERS_HANDLER__

#include <cstdint>
#include <type_traits>
#include "Common.hpp"

// Defined here so the MemHandler knows the type size
struct PatchDistance
{
    int32_t x;
    int32_t y;
    float distance;
    
    inline bool operator < (const PatchDistance& other) const noexcept
    {
        return distance < other.distance;
    }
};

struct MemHandler
{
    int64_t memBlockId;
    uint8_t* RESTRICT SuperBufferHead;

    int32_t padW; // Padded Width
    int32_t padH; // Padded Height
    
    // Scale 0: Full Resolution (L0) & Laplacian Differences
    float* Y_planar;
	float* U_planar;
	float* V_planar;
    float* Y_diff_full;
	float* U_diff_full;
	float* V_diff_full;
    
    // Scale 1: Half Resolution (L1) & Laplacian Differences
    float* Y_half;
	float* U_half;
	float* V_half;
    float* Y_diff_half;
	float* U_diff_half;
	float* V_diff_half;
    
    // Scale 2: Quarter Resolution (L2 - Base scale)
    float* Y_quart;
	float* U_quart;
	float* V_quart;
    
    // Oracle Covariance LUTs (Sub-Block 2)
    float* NoiseCov_Y;    
    float* NoiseCov_U;    
    float* NoiseCov_V;
    float* OracleWorkspace; 
    
    // =========================================================
    // NEW: BAYESIAN FILTER BUFFERS (Sub-Blocks 4 & 5)
    // =========================================================
    PatchDistance* SearchPool;
    
    // Aggregation Accumulators
    float* Accum_Y;
    float* Accum_U;
    float* Accum_V;
    float* Weight_Count; // Replaces the Hann Window (NC uses flat counting)
    
    // 3D Volume Workspace for Matrix Math
    float* Scratch3D; 
    
    size_t totalSize; 
};

MemHandler alloc_memory_buffers(int32_t sizeX, int32_t sizeY, int32_t blockSize, const bool dbgPrn = false) noexcept;

void free_memory_buffers(MemHandler& algoMemHandler) noexcept;

inline bool mem_handler_valid(const MemHandler& hndl) noexcept
{
    return (hndl.memBlockId >= 0 && hndl.SuperBufferHead != nullptr);
}


#endif // __IMAGE_LAB_ALGO_MEMORY_BUFFERS_HANDLER__