#pragma once

#include <cstdint>
#include <cuda_runtime.h>
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
    // --- Image-Sized Buffers ---
    float* d_r;
    float* d_g;
    float* d_b;
    int*   d_labels;    // Stores the cluster ID each pixel belongs to
    float* d_distances; // Stores the closest distance found so far

    // --- Cluster-Sized Buffers (Size depends on safe_k) ---
    float* d_cluster_x;
    float* d_cluster_y;
    float* d_cluster_r;
    float* d_cluster_g;
    float* d_cluster_b;

    // --- Accumulator Buffers (For averaging during iterations) ---
    float* d_acc_x;
    float* d_acc_y;
    float* d_acc_r;
    float* d_acc_g;
    float* d_acc_b;
    int*   d_acc_count; // How many pixels are assigned to this cluster

    // --- Safe Algorithm Parameters ---
    int safe_k;
    int step_size;
    
    // The master pointer to our single VRAM block
    void* master_arena; 
};

