#include <cstdio>
#include <cfloat>
#include <cmath> 
#include <algorithm>
#include <cuda_runtime.h>
#include "ArtMosaic_GPU.hpp"
#include "CUDA/CudaMemHandler.cuh"

// =========================================================
// 1. ARENA MANAGEMENT
// =========================================================

bool allocateGpuArena (GpuMemHandler& mem, int width, int height, int requested_k) 
{
    long totalPixels = static_cast<long>(width) * height;
    mem.safe_k = (requested_k > totalPixels) ? totalPixels : requested_k;
    
    float superPixInitVal = static_cast<float>(totalPixels) / static_cast<float>(mem.safe_k);
    mem.step_size = static_cast<int>(sqrtf(superPixInitVal));
    if (mem.step_size < 1) mem.step_size = 1;

    size_t image_pixels = static_cast<size_t>(width) * height;
    size_t cluster_count = static_cast<size_t>(mem.safe_k);

    size_t total_vram = (image_pixels * sizeof(float) * 4) + (image_pixels * sizeof(int)) + 
                        (cluster_count * sizeof(float) * 10) + (cluster_count * sizeof(int));

    cudaError_t err = cudaMalloc(&mem.master_arena, total_vram);
    if (err != cudaSuccess) return false; 

    char* ptr = (char*)mem.master_arena;
    mem.d_r = (float*)ptr; ptr += image_pixels * sizeof(float);
    mem.d_g = (float*)ptr; ptr += image_pixels * sizeof(float);
    mem.d_b = (float*)ptr; ptr += image_pixels * sizeof(float);
    mem.d_distances = (float*)ptr; ptr += image_pixels * sizeof(float);
    mem.d_labels = (int*)ptr; ptr += image_pixels * sizeof(int);

    mem.d_cluster_x = (float*)ptr; ptr += cluster_count * sizeof(float);
    mem.d_cluster_y = (float*)ptr; ptr += cluster_count * sizeof(float);
    mem.d_cluster_r = (float*)ptr; ptr += cluster_count * sizeof(float);
    mem.d_cluster_g = (float*)ptr; ptr += cluster_count * sizeof(float);
    mem.d_cluster_b = (float*)ptr; ptr += cluster_count * sizeof(float);

    mem.d_acc_x = (float*)ptr; ptr += cluster_count * sizeof(float);
    mem.d_acc_y = (float*)ptr; ptr += cluster_count * sizeof(float);
    mem.d_acc_r = (float*)ptr; ptr += cluster_count * sizeof(float);
    mem.d_acc_g = (float*)ptr; ptr += cluster_count * sizeof(float);
    mem.d_acc_b = (float*)ptr; ptr += cluster_count * sizeof(float);
    mem.d_acc_count = (int*)ptr;

    return true;
}

void freeGpuArena(GpuMemHandler& mem) { 
    if (mem.master_arena) { cudaFree(mem.master_arena); mem.master_arena = nullptr; }
}

// =========================================================
// 2. CPU-PARITY SLIC KERNELS
// =========================================================
__global__ void InterleavedToPlanarKernel
(
    const float* RESTRICT d_in,
    float* RESTRICT d_r,
    float* RESTRICT d_g,
    float* RESTRICT d_b,
    int width,
    int height,
    int pitch
)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x < width && y < height)
    {
        const float* row = d_in + (y * pitch * 4);
        int idx = y * width + x;
        d_b[idx] = row[x * 4 + 0] * 255.0f;
        d_g[idx] = row[x * 4 + 1] * 255.0f;
        d_r[idx] = row[x * 4 + 2] * 255.0f;
    }
}

__global__ void InitCentersSingleThreadKernel
(
    GpuMemHandler mem,
    int* d_grid_to_k,
    int w, 
    int h, 
    int S, 
    int nX, 
    int nY, 
    int hPW, 
    int hPH, 
    int s_half, 
    int* d_actualK
)
{
    int actualK = 0;
    for (int j = 0; j < nY; j++)
    {
        int jj = j * S + s_half + hPH;
        for (int i = 0; i < nX; i++)
        {
            int ii = i * S + s_half + hPW;
            int grid_idx = j * nX + i;
            if (ii < w && jj < h)
            {
                int planarIdx = jj * w + ii;
                mem.d_cluster_x[actualK] = (float)ii; 
                mem.d_cluster_y[actualK] = (float)jj;
                mem.d_cluster_r[actualK] = mem.d_r[planarIdx]; 
                mem.d_cluster_g[actualK] = mem.d_g[planarIdx]; 
                mem.d_cluster_b[actualK] = mem.d_b[planarIdx];
                d_grid_to_k[grid_idx] = actualK++;
            } 
            else
            {
                d_grid_to_k[grid_idx] = -1;
            }
        }
    }
    *d_actualK = actualK;
}

__global__ void InitCentersParallelKernel(GpuMemHandler mem, int* d_grid_to_k, int w, int h, int S, int nX, int nY, int hPW, int hPH, int s_half, int* d_actualK) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i >= nX || j >= nY) return; // Thread is outside the mathematical grid

    int ii = i * S + s_half + hPW;
    int jj = j * S + s_half + hPH;
    int grid_idx = j * nX + i;

    if (ii < w && jj < h) {
        // Safely claim the next available index in the compacted array
        int my_k = atomicAdd(d_actualK, 1);

        int planarIdx = jj * w + ii;
        mem.d_cluster_x[my_k] = (float)ii; 
        mem.d_cluster_y[my_k] = (float)jj;
        mem.d_cluster_r[my_k] = mem.d_r[planarIdx]; 
        mem.d_cluster_g[my_k] = mem.d_g[planarIdx]; 
        mem.d_cluster_b[my_k] = mem.d_b[planarIdx];
        
        d_grid_to_k[grid_idx] = my_k; 
    } else {
        // Mark as a hole
        d_grid_to_k[grid_idx] = -1;
    }
}


__global__ void GradientPerturbationKernel(GpuMemHandler mem, int w, int h, int actualK) {
    int k_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (k_idx < actualK) {
        int startX = __float2int_rn(mem.d_cluster_x[k_idx]);
        int startY = __float2int_rn(mem.d_cluster_y[k_idx]);
        float minGrad = FLT_MAX;
        int bestX = startX, bestY = startY;

        for (int j = -1; j <= 1; j++) {
            for (int i = -1; i <= 1; i++) {
                int cx = startX + i, cy = startY + j;
                if (cx >= 0 && cx < w && cy >= 0 && cy < h) {
                    int idx = cy * w + cx;
                    int nx = min(cx + 1, w - 1);
                    float drX = mem.d_r[cy * w + nx] - mem.d_r[idx];
                    float dgX = mem.d_g[cy * w + nx] - mem.d_g[idx];
                    float dbX = mem.d_b[cy * w + nx] - mem.d_b[idx];
                    
                    int ny = min(cy + 1, h - 1);
                    float drY = mem.d_r[ny * w + cx] - mem.d_r[idx];
                    float dgY = mem.d_g[ny * w + cx] - mem.d_g[idx];
                    float dbY = mem.d_b[ny * w + cx] - mem.d_b[idx];
                    
                    float g = (drX*drX + dgX*dgX + dbX*dbX) + (drY*drY + dgY*dgY + dbY*dbY);
                    if (g < minGrad) { minGrad = g; bestX = cx; bestY = cy; }
                }
            }
        }
        int bestIdx = bestY * w + bestX;
        mem.d_cluster_x[k_idx] = (float)bestX; mem.d_cluster_y[k_idx] = (float)bestY;
        mem.d_cluster_r[k_idx] = mem.d_r[bestIdx]; mem.d_cluster_g[k_idx] = mem.d_g[bestIdx]; mem.d_cluster_b[k_idx] = mem.d_b[bestIdx];
    }
}

__global__ void ClearDistancesKernel(GpuMemHandler mem, int count) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < count) {
        mem.d_distances[i] = FLT_MAX;
        mem.d_labels[i] = -1;
    }
}

// OPTIMIZED: HW Intrinsic L1 Caching and Early Exit
__global__ void FastAssignmentKernel(GpuMemHandler mem, const int* d_grid_to_k, int w, int h, int S, int nX, int nY, float wSpaceSq) {
    int px = blockIdx.x * blockDim.x + threadIdx.x;
    int py = blockIdx.y * blockDim.y + threadIdx.y;
    if (px >= w || py >= h) return;
    
    int pIdx = py * w + px;
    
    // __ldg forces read through L1 Texture cache
    float pr = __ldg(&mem.d_r[pIdx]);
    float pg = __ldg(&mem.d_g[pIdx]);
    float pb = __ldg(&mem.d_b[pIdx]);
    
    float min_dist = mem.d_distances[pIdx];
    int best_k = mem.d_labels[pIdx];

    int gx = px / S, gy = py / S;
    
    #pragma unroll
    for (int dy = -2; dy <= 2; dy++) {
        #pragma unroll
        for (int dx = -2; dx <= 2; dx++) {
            int cx = gx + dx, cy = gy + dy;
            if (cx >= 0 && cx < nX && cy >= 0 && cy < nY) {
                int k_idx = __ldg(&d_grid_to_k[cy * nX + cx]);
                if (k_idx != -1) {
                    float cX = __ldg(&mem.d_cluster_x[k_idx]);
                    float cY = __ldg(&mem.d_cluster_y[k_idx]);

                    // Fast hardware intrinsic rounding
                    int cX_i = __float2int_rn(cX);
                    int cY_i = __float2int_rn(cY);

                    if (px >= cX_i - S && px < cX_i + S && py >= cY_i - S && py < cY_i + S) {
                        float d_x = (float)px - cX;
                        float d_y = (float)py - cY;
                        float ds = (d_x * d_x + d_y * d_y) * wSpaceSq;

                        // MASSIVE SPEEDUP: If spatial distance alone is worse than best_total, skip memory reads!
                        if (ds >= min_dist) continue;

                        float dr = pr - __ldg(&mem.d_cluster_r[k_idx]);
                        float dg = pg - __ldg(&mem.d_cluster_g[k_idx]);
                        float db = pb - __ldg(&mem.d_cluster_b[k_idx]);
                        float dc = (dr * dr + dg * dg + db * db);

                        if (ds + dc < min_dist) { 
                            min_dist = ds + dc; 
                            best_k = k_idx; 
                        }
                    }
                }
            }
        }
    }
    mem.d_labels[pIdx] = best_k; 
    mem.d_distances[pIdx] = min_dist;
}

__global__ void ClearAccumulatorsKernel(GpuMemHandler mem, int K) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < K) {
        mem.d_acc_x[i] = 0.0f; mem.d_acc_y[i] = 0.0f;
        mem.d_acc_r[i] = 0.0f; mem.d_acc_g[i] = 0.0f; mem.d_acc_b[i] = 0.0f;
        mem.d_acc_count[i] = 0;
    }
}

// OPTIMIZED: Read-Only caching for R,G,B inputs during atomics
__global__ void AccumulateKernel(GpuMemHandler mem, int w, int h) {
    int px = blockIdx.x * blockDim.x + threadIdx.x;
    int py = blockIdx.y * blockDim.y + threadIdx.y;
    if (px >= w || py >= h) return;
    
    int pIdx = py * w + px;
    int k_idx = mem.d_labels[pIdx];
    
    if (k_idx >= 0) {
        atomicAdd(&mem.d_acc_x[k_idx], (float)px);
        atomicAdd(&mem.d_acc_y[k_idx], (float)py);
        atomicAdd(&mem.d_acc_r[k_idx], __ldg(&mem.d_r[pIdx]));
        atomicAdd(&mem.d_acc_g[k_idx], __ldg(&mem.d_g[pIdx]));
        atomicAdd(&mem.d_acc_b[k_idx], __ldg(&mem.d_b[pIdx]));
        atomicAdd(&mem.d_acc_count[k_idx], 1);
    }
}

__global__ void UpdateCentersKernel(GpuMemHandler mem, int K) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < K && mem.d_acc_count[i] > 0) {
        float inv = 1.0f / (float)mem.d_acc_count[i];
        mem.d_cluster_x[i] = mem.d_acc_x[i] * inv; mem.d_cluster_y[i] = mem.d_acc_y[i] * inv;
        mem.d_cluster_r[i] = mem.d_acc_r[i] * inv; mem.d_cluster_g[i] = mem.d_acc_g[i] * inv; mem.d_cluster_b[i] = mem.d_acc_b[i] * inv;
    }
}

// =========================================================
// 3. FAST PARALLEL UNION-FIND 
// =========================================================

__global__ void InitCCKernel(int* d_cc, int* d_sizes, int* d_new_labels, int numPixels) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numPixels) {
        d_cc[idx] = idx;
        d_sizes[idx] = 0;
        d_new_labels[idx] = -1;
    }
}

__global__ void LinkCCKernel(const int* labels, int* cc, int w, int h) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= w || y >= h) return;
    int idx = y * w + x;
    int L = labels[idx];

    auto merge = [&](int nIdx) {
        if (labels[nIdx] == L) {
            int u = idx, v = nIdx;
            while (u != v) {
                u = cc[u]; v = cc[v];
                if (u < v) { int old = atomicMin(&cc[v], u); if (old == v) break; v = old; }
                else if (v < u) { int old = atomicMin(&cc[u], v); if (old == u) break; u = old; }
            }
        }
    };
    if (x + 1 < w) merge(idx + 1);
    if (y + 1 < h) merge(idx + w);
}

__global__ void CompressCCKernel(int* cc, int numPixels) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numPixels) {
        int root = idx;
        while (root != cc[root]) {
            cc[root] = cc[cc[root]]; 
            root = cc[root];
        }
        cc[idx] = root;
    }
}

__global__ void SizeCCKernel(const int* cc, int* sizes, int numPixels) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numPixels) atomicAdd(&sizes[cc[idx]], 1);
}

__global__ void ResolveOrphansKernel(const int* labels, const int* cc, const int* sizes, int* new_labels, int min_size, int w, int h) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= w || y >= h) return;
    int idx = y * w + x;
    int root = cc[idx];

    if (sizes[root] <= min_size) {
        int L = labels[idx];
        int candidate = -1;
        if (x > 0 && labels[idx - 1] != L) candidate = labels[idx - 1];
        else if (y > 0 && labels[idx - w] != L) candidate = labels[idx - w];
        else if (x + 1 < w && labels[idx + 1] != L) candidate = labels[idx + 1];
        else if (y + 1 < h && labels[idx + w] != L) candidate = labels[idx + w];

        if (candidate != -1) atomicMax(&new_labels[root], candidate);
    }
}

__global__ void ApplyResolvedLabelsKernel(int* labels, const int* cc, const int* sizes, const int* new_labels, int min_size, int numPixels) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numPixels) {
        int root = cc[idx];
        if (sizes[root] <= min_size) {
            int nl = new_labels[root];
            if (nl != -1) labels[idx] = nl;
        }
    }
}

// =========================================================
// 4. FINAL RENDER
// =========================================================

__global__ void RenderMosaicKernel(const float* d_in, GpuMemHandler mem, float* d_out, int w, int h, int sP, int dP, int K) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= w || y >= h) return;

    int idx = y * w + x;
    int lVal = mem.d_labels[idx];
    float* out_row = d_out + (y * dP * 4);
    const float* in_row = d_in + (y * sP * 4);

    bool isBorder = false;
    if (x + 1 < w && mem.d_labels[idx + 1] != lVal) isBorder = true;
    else if (y + 1 < h && mem.d_labels[idx + w] != lVal) isBorder = true;

    if (isBorder) {
        out_row[x*4+0] = 0.5f; out_row[x*4+1] = 0.5f; out_row[x*4+2] = 0.5f;
    } else if (lVal >= 0 && lVal < K) {
        out_row[x*4+0] = mem.d_cluster_b[lVal] / 255.0f;
        out_row[x*4+1] = mem.d_cluster_g[lVal] / 255.0f;
        out_row[x*4+2] = mem.d_cluster_r[lVal] / 255.0f;
    }
    out_row[x*4+3] = in_row[x*4+3]; 
}

// =========================================================
// 5. MAIN DISPATCHER
// =========================================================

CUDA_KERNEL_CALL
void ImageLabMosaic_CUDA
(
    const float* RESTRICT inBuffer,
    float* RESTRICT outBuffer,
    int srcPitch, 
    int dstPitch, 
    int width, 
    int height, 
    int cellsNumber, 
    int frameCount, 
    cudaStream_t stream
)
{
    GpuMemHandler mem{};
    if(!allocateGpuArena(mem, width, height, cellsNumber)) return;

    const int numPixels = width * height;
    int S = mem.step_size;
    int nX = width / S; int nY = height / S;
    int hPW = std::max(0, width - S * nX) / 2; 
    int hPH = std::max(0, height - S * nY) / 2;

    dim3 pThreads(32, 16); dim3 pBlocks((width + 31) / 32, (height + 15) / 16);
    int pBlocks1D = (numPixels + 255) / 256;

    int *d_grid_to_k, *d_actualK, *d_cc, *d_sizes, *d_new_labels;
    cudaMalloc(&d_grid_to_k, nX * nY * sizeof(int));
    cudaMalloc(&d_actualK, sizeof(int));
    cudaMalloc(&d_cc, numPixels * sizeof(int));
    cudaMalloc(&d_sizes, numPixels * sizeof(int));
    cudaMalloc(&d_new_labels, numPixels * sizeof(int));

    InterleavedToPlanarKernel<<<pBlocks, pThreads, 0, stream>>>(inBuffer, mem.d_r, mem.d_g, mem.d_b, width, height, srcPitch);
    
    // Calculate grid size specifically for the nX * nY cluster grid
    dim3 kBlocks((nX + 31) / 32, (nY + 15) / 16);

    // Run fully parallel
    InitCentersParallelKernel<<<kBlocks, pThreads, 0, stream>>>(mem, d_grid_to_k, width, height, S, nX, nY, hPW, hPH, S/2, d_actualK);
    
    int host_actualK = 0;
    cudaMemcpyAsync(&host_actualK, d_actualK, sizeof(int), cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);

    int cBlocks = (host_actualK + 255) / 256;
    GradientPerturbationKernel<<<cBlocks, 256, 0, stream>>>(mem, width, height, host_actualK);

    float wSpaceSq = (40.0f / (float)S) * (40.0f / (float)S);
    
    // --- 1. FAST ASSIGNMENT ---
    for (int i = 0; i < 10; i++)
    {
        ClearDistancesKernel <<<pBlocks1D, 256, 0, stream>>>(mem, numPixels);
        FastAssignmentKernel <<<pBlocks, pThreads, 0, stream>>>(mem, d_grid_to_k, width, height, S, nX, nY, wSpaceSq);
        
        ClearAccumulatorsKernel <<<cBlocks, 256, 0, stream>>>(mem, host_actualK);
        AccumulateKernel <<<pBlocks, pThreads, 0, stream>>>(mem, width, height);
        UpdateCentersKernel <<<cBlocks, 256, 0, stream>>>(mem, host_actualK);
    }

    // --- 2. FAST UNION-FIND CONNECTIVITY ---
    int MIN_SIZE = (numPixels / host_actualK) >> 2;

    for(int pass = 0; pass < 3; pass++)
    {
        InitCCKernel <<<pBlocks1D, 256, 0, stream>>>(d_cc, d_sizes, d_new_labels, numPixels);
        
        LinkCCKernel <<<pBlocks, pThreads, 0, stream>>>(mem.d_labels, d_cc, width, height);
        
        for (int c = 0; c < 3; c++) 
            CompressCCKernel <<<pBlocks1D, 256, 0, stream>>>(d_cc, numPixels);
        
        SizeCCKernel<<<pBlocks1D, 256, 0, stream>>>(d_cc, d_sizes, numPixels);
        ResolveOrphansKernel <<<pBlocks, pThreads, 0, stream>>>(mem.d_labels, d_cc, d_sizes, d_new_labels, MIN_SIZE, width, height);
        ApplyResolvedLabelsKernel<<<pBlocks1D, 256, 0, stream>>>(mem.d_labels, d_cc, d_sizes, d_new_labels, MIN_SIZE, numPixels);
    }

    // --- 3. FINAL RE-AVERAGE ---
    ClearAccumulatorsKernel<<<cBlocks, 256, 0, stream>>>(mem, host_actualK);
    AccumulateKernel <<<pBlocks, pThreads, 0, stream>>>(mem, width, height);
    UpdateCentersKernel <<<cBlocks, 256, 0, stream>>>(mem, host_actualK);

    RenderMosaicKernel<<<pBlocks, pThreads, 0, stream>>>(inBuffer, mem, outBuffer, width, height, srcPitch, dstPitch, host_actualK);
    
    cudaStreamSynchronize(stream);
    cudaFree(d_grid_to_k); cudaFree(d_actualK); 
    cudaFree(d_cc); cudaFree(d_sizes); cudaFree(d_new_labels);
    freeGpuArena(mem);
}