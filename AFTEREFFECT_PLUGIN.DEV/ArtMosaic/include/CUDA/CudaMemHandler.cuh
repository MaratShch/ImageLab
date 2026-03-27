#pragma once

#include <cstdint>
#include <cuda_runtime.h>
#include "Common.hpp"
#include "ImageLabCUDA.hpp"

// CUDA Error Checking Macro
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA Error at %s:%d - %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
            return false; \
        } \
    } while(0)

struct GpuMemHandler
{
    // Central VRAM Arena Block
    void* master_arena;
    int32_t safe_k;
    int32_t step_size;

    // Internal Planar Image Buffers (Range: 0.0f to 255.0f)
    float* RESTRICT d_r;
    float* RESTRICT d_g;
    float* RESTRICT d_b;

    // SLIC Algorithm Core Buffers
    float* RESTRICT d_distances;
    int32_t* RESTRICT d_labels;

    // Superpixel Cluster Data
    float* RESTRICT d_cluster_x;
    float* RESTRICT d_cluster_y;
    float* RESTRICT d_cluster_r;
    float* RESTRICT d_cluster_g;
    float* RESTRICT d_cluster_b;

    // Accumulators for Center Updating
    float* RESTRICT d_acc_x;
    float* RESTRICT d_acc_y;
    float* RESTRICT d_acc_r;
    float* RESTRICT d_acc_g;
    float* RESTRICT d_acc_b;
    int32_t* RESTRICT d_acc_count;

    // --- THESE ARE THE MISSING VARIABLES ---
    // Fast Union-Find & Connectivity Buffers 
    int32_t* RESTRICT d_grid_to_k;
    int32_t* RESTRICT d_actualK;
    int32_t* RESTRICT d_cc;
    int32_t* RESTRICT d_sizes;
    int32_t* RESTRICT d_new_labels;
};

