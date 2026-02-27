#pragma once

#include <cstdint>
#include <cuda_runtime.h>
#include "Common.hpp"

// CUDA Error Checking Macro
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA Error at %s:%d - %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
            return false; \
        } \
    } while(0)

// A single 3D patch coordinate
struct int2_coord
{
    int32_t x;
    int32_t y;
};

struct CudaMemHandler
{
    // --- DIMENSIONS ---
    int32_t tileW;          // Active tile width (e.g., 960)
    int32_t tileH;          // Active tile height (e.g., 540)
    int32_t padW;           // Padded width (tileW + 32)
    int32_t padH;           // Padded height (tileH + 32)
    int32_t frameSizePadded; 

    // --- LEVEL 0: FULL RESOLUTION ---
    float* d_Y_planar;
    float* d_U_planar;
    float* d_V_planar;

    // --- LEVEL 1: HALF RESOLUTION ---
    float* d_Y_half;
    float* d_U_half;
    float* d_V_half;

    // --- LEVEL 2: QUARTER RESOLUTION ---
    float* d_Y_quart;
    float* d_U_quart;
    float* d_V_quart;

    // --- THE BLIND ORACLE LUTS ---
    float* d_NoiseCov_Y; // 256 * 256 bytes
    float* d_NoiseCov_U; 
    float* d_NoiseCov_V; 

    // --- PHASE 1 (BASIC ESTIMATE) PILOT BUFFERS ---
    float* d_Pilot_Y;
    float* d_Pilot_U;
    float* d_Pilot_V;

    // --- PHASE 2 AGGREGATION ACCUMULATORS ---
    float* d_Accum_Y;
    float* d_Accum_U;
    float* d_Accum_V;
    float* d_Weight_Count;

    // --- 3D SEARCH POOL (The VRAM Eater) ---
    // Stores the (x,y) coordinates of similar patches found by the block matcher.
    // Sized for max threads in a tile * max patches per group (e.g., 128)
    int2_coord* d_SearchPool; 
};

// API
bool alloc_cuda_memory_buffers (CudaMemHandler& mem, int32_t target_tile_width, int32_t target_tile_height);
void free_cuda_memory_buffers (CudaMemHandler& mem);