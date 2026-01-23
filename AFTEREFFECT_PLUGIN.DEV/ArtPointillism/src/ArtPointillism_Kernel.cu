#include "ArtPointillism_GPU.hpp"
#include "CompileTimeUtils.hpp"
#include "ImageLabCUDA.hpp"
#include "ArtPointillismKernel.cuh"
#include <algorithm>
#include <cmath>

// Static Context (Singleton style for simplicity)
static GpuContext g_gpuCtx;

__device__ inline uint32_t hash_coords(uint32_t x, uint32_t y, uint32_t seed) noexcept
{
    uint32_t h = x;
    h ^= y;
    h ^= seed;
    h ^= h >> 16;
    h *= 0x85ebca6b;
    h ^= h >> 13;
    h *= 0xc2b2ae35;
    h ^= h >> 16;
    return h;
}

// Returns float [0.0, 1.0)
__device__ inline float random_float(int x, int y, int seed) noexcept
{
    uint32_t h = hash_coords(x, y, seed);
    // Convert int bits to float [0, 1) standard trick
    return (h & 0x00FFFFFF) * (1.0f / 16777216.0f);
}


// Helper: RGB to Lab (Inline Device Function)
__device__ float3 rgb_to_lab (float3 rgb)
{
    // 1. Linearize (Assuming input is roughly sRGB-ish gamma, but Adobe 32f is often linear)
    // For 32f Adobe buffers, data is usually Linear or Rec709 Scene. 
    // We will assume Linear input for the 32f path to save pow().

    // 2. RGB -> XYZ
    float r = rgb.x; float g = rgb.y; float b = rgb.z;
    float X = 0.4124564f * r + 0.3575761f * g + 0.1804375f * b;
    float Y = 0.2126729f * r + 0.7151522f * g + 0.0721750f * b;
    float Z = 0.0193339f * r + 0.1191920f * g + 0.9503041f * b;

    // 3. XYZ -> Lab
    // Normalize to D65
    X /= 0.95047f; Z /= 1.08883f;

    // f(t) approx
    auto f = [](float t) noexcept
    {
        return (t > 0.008856f) ? std::cbrt(t) : (7.787f * t + 16.0f / 116.0f);
    };

    float fx = f(X); float fy = f(Y); float fz = f(Z);

    return make_float3
    (
        116.0f * fy - 16.0f,     // L
        500.0f * (fx - fy),      // a
        200.0f * (fy - fz)       // b
    );
}


// Pack data into int4 (16 bytes)
__device__ inline int4 pack_jfa(int id, float x, float y, float distSq) noexcept
{
    return make_int4(
        id,
        __float_as_int(x),
        __float_as_int(y),
        __float_as_int(distSq)
    );
}

// Unpack ID
__device__ inline int unpack_id (int4 c) noexcept
{ 
    return c.x;
}

// Unpack Position
__device__ inline float2 unpack_pos(int4 c) noexcept
{
    return make_float2(__int_as_float(c.y), __int_as_float(c.z));
}


// THE KERNEL
__global__ void k_Preprocess_Fused
(
    const float* RESTRICT srcPtr, // Adobe Buffer (Raw float pointer)
    DensityInfo* RESTRICT dstMap, // Output Density
    int width,
    int height,
    int srcPitchBytes,                // Stride in Bytes
    float edgeSensitivity             // 0.0 to 1.0
)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height)
        return;

    // 1. Read Center Pixel (Luma) for Structure
    // Adobe 32f is usually B G R A (float4)
    // Pitch math: row_ptr = base + y * pitch_bytes
    const float4* row_ptr = (const float4*)((const char*)srcPtr + y * srcPitchBytes);
    float4 px = row_ptr[x];

    // Convert to Lab
    float3 rgb = make_float3(px.z, px.y, px.x); // BGR -> RGB
    float3 lab = rgb_to_lab(rgb);
    float L_center = lab.x / 100.0f; // Normalize 0..1
    float L_inv = 1.0f - L_center;

    // 2. Sobel Edge Detection (On-the-fly)
    // We read neighbors directly from Global Memory. L2 Cache handles the reuse.
    // For Phase 1 on 1060, this is plenty fast.
    float Gx = 0.0f;
    float Gy = 0.0f;

    // Simple 3x3 loop
    // Kernel Gx: -1 0 1 ...
    for (int dy = -1; dy <= 1; dy++)
    {
        int ny = std::min(std::max(y + dy, 0), height - 1);
        const float4* nrow_ptr = (const float4*)((const char*)srcPtr + ny * srcPitchBytes);

        for (int dx = -1; dx <= 1; dx++)
        {
            int nx = std::min(std::max(x + dx, 0), width - 1);

            // Read neighbor
            float4 n_px = nrow_ptr[nx];
            // Fast Luma approx for Sobel is sufficient (Green channel or Avg)
            // Or full Lab conversion? Full Lab conversion 9 times is expensive.
            // Optimization: Use Green channel or simple avg for Edge Detection.
            float n_luma = (n_px.x + n_px.y + n_px.z) * 0.3333f;

            // Apply Sobel Gx
            if (dx != 0)
            {
                float weight = (dy == 0) ? 2.0f : 1.0f;
                Gx += (dx * weight * n_luma);
            }
            // Apply Sobel Gy
            if (dy != 0)
            {
                float weight = (dx == 0) ? 2.0f : 1.0f;
                Gy += (dy * weight * n_luma);
            }
        }
    }

    float edge_mag = sqrtf(Gx*Gx + Gy*Gy);

    // 3. Mix & Normalize
    // Sensitivity: 0 = Luma Only, 1 = Edge Only
    float w_edge = edgeSensitivity; // Already normalized 0..1 by host
    float w_luma = 1.0f - w_edge;

    float final_density = (L_inv * w_luma) + (edge_mag * w_edge);

    // Apply "Lifted Floor" logic (from our CPU lessons)
    // Prevent zero-density holes
    final_density = fmaxf(final_density, 0.1f);
    final_density = fminf(final_density, 1.0f);

    // 4. Calculate Orientation (for Van Gogh Flow)
    // Angle perpendicular to gradient
    float angle = atan2f(Gy, Gx) + 1.570796f; // + 90 deg

                                              // 5. Store
    int idx = y * width + x;
    dstMap[idx] = make_float2(final_density, angle);
}


// --- KERNEL: PHASE 2 SEEDING (Warp-Aggregated) ---
__global__ void k_Seeding_WarpAggregated
(
    const DensityInfo* RESTRICT densityMap,
    GPUDot*            RESTRICT dotBuffer,
    int*               RESTRICT globalCounter, // Points to d_counters[0]
    int width,
    int height,
    float probabilityScale,
    int randomSeed,
    int maxDots
)
{
    // 1. Calculate Coordinates
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;

    bool spawn_dot = false;
    float my_x = 0.0f;
    float my_y = 0.0f;

    // 2. Determine Spawning (Thread Local)
    if (x < width && y < height)
    {
        int tid = y * width + x;
        // Read Density (internal buffer is tightly packed)
        float dens = densityMap[tid].x;

        // Probability Logic
        float chance = dens * probabilityScale;
        float roll = random_float(x, y, randomSeed);

        if (roll < chance)
        {
            spawn_dot = true;
            // Jitter within pixel (0.0 to 1.0 offset)
            // Use different seeds/offsets for X and Y jitter to avoid diagonal bias
            float jx = random_float(x + 10000, y, randomSeed);
            float jy = random_float(x, y + 10000, randomSeed);
            my_x = (float)x + jx;
            my_y = (float)y + jy;
        }
    }

    // 3. Warp Aggregation (Optimize Atomics)
    // Coalesce up to 32 atomicAdds into 1 per warp
    unsigned int mask = __activemask();
    unsigned int want_mask = __ballot_sync(mask, spawn_dot);

    // Check if anyone in the warp wants to write
    if (want_mask != 0)
    {
        int lane_id = threadIdx.x % 32;
        int leader_idx = __ffs(want_mask) - 1; // Find First Set
        int pop_count = __popc(want_mask);    // Total dots in this warp
        unsigned int lower_mask = (1U << lane_id) - 1;
        int local_rank = __popc(want_mask & lower_mask); // My index within the warp block

        int warp_base_offset = 0;

        // Leader allocates space for the whole group
        if (lane_id == leader_idx) {
            warp_base_offset = atomicAdd(globalCounter, pop_count);
        }

        // Broadcast offset back to everyone
        warp_base_offset = __shfl_sync(mask, warp_base_offset, leader_idx);

        // 4. Write to Global Memory
        if (spawn_dot)
        {
            int my_global_idx = warp_base_offset + local_rank;

            if (my_global_idx < maxDots)
            {
                // Direct struct write (16-byte aligned store)
                GPUDot d;
                d.pos_x = my_x;
                d.pos_y = my_y;
                d.padding1 = 0;
                d.padding2 = 0;

                dotBuffer[my_global_idx] = d;
            }
        }
    }
    return;
}


// 1. Clear Grid to "Empty" (-1)
__global__ void k_JFA_Clear
(
    JFACell* RESTRICT grid,
    int num_pixels
)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_pixels)
    {
        // ID = -1, Pos = 0,0, Dist = Infinite
        grid[idx] = pack_jfa(-1, -10000.0f, -10000.0f, 1.0e20f);
    }
}

// 2. Dots write themselves to the grid
__global__ void k_JFA_Splat
(
    const GPUDot* RESTRICT dots,
    const int*    RESTRICT counterPtr, // d_counters[0] contains count
    JFACell*      RESTRICT grid,
    int width,
    int height
)
{
    const int id = blockIdx.x * blockDim.x + threadIdx.x;
    int count = counterPtr[0];

    if (id < count)
    {
        // Read Dot Info
        GPUDot d = dots[id];
        int px = (int)(d.pos_x + 0.5f);
        int py = (int)(d.pos_y + 0.5f);

        // Bounds check
        if (px >= 0 && px < width && py >= 0 && py < height)
        {
            int grid_idx = py * width + px;
            // Write Seed Info: ID, X, Y, Dist=0
            grid[grid_idx] = pack_jfa(id, d.pos_x, d.pos_y, 0.0f);
        }
    }
}

// --- KERNEL: Geometric Refinement (Jump Flooding Algorithm) ---
__global__ void k_JFA_Step
(
    const JFACell* RESTRICT src,
    JFACell*       RESTRICT dst,
    int width, int height,
    int step_len
)
{
    // 2D Thread ID
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height)
        return;

    int self_idx = y * width + x;

    // 1. Load Current Best (Self) from Source
    int4 best_cell = src[self_idx];

    // Unpack distance. If ID is -1, treat dist as Infinite
    float best_dist = (best_cell.x == -1) ? 1.0e20f : __int_as_float(best_cell.w);
    float2 my_pos = make_float2((float)x, (float)y);

    // 2. Iterate 8 Neighbors + Center
    // We unroll manually or loop. Loop is fine for JFA.
    for (int dy = -1; dy <= 1; ++dy)
    {
        for (int dx = -1; dx <= 1; ++dx)
        {
            if (dx == 0 && dy == 0) continue; // Skip self (already loaded)

            int nx = x + dx * step_len;
            int ny = y + dy * step_len;

            // Boundary Check
            if (nx >= 0 && nx < width && ny >= 0 && ny < height)
            {
                int n_idx = ny * width + nx;

                // Read Neighbor
                int4 n_cell = src[n_idx];

                // Does neighbor have a valid seed?
                if (n_cell.x != -1)
                {
                    float2 seed_pos = unpack_pos(n_cell);

                    // Calculate Dist from ME to NEIGHBOR'S SEED
                    float2 diff = make_float2(my_pos.x - seed_pos.x, my_pos.y - seed_pos.y);
                    float distSq = diff.x*diff.x + diff.y*diff.y;

                    // Compare & Swap
                    if (distSq < best_dist)
                    {
                        best_dist = distSq;
                        best_cell = n_cell;
                        // Update cached distance in w component
                        best_cell.w = __float_as_int(best_dist);
                    }
                }
            }
        }
    }

    // 3. Write Best to Dest
    dst[self_idx] = best_cell;
}



CUDA_KERNEL_CALL
void ArtPointillism_CUDA
(
    const float* RESTRICT inBuffer, // source (input) buffer
    float* RESTRICT outBuffer,      // destination (output) buffer
    int srcPitch,                   // source buffer pitch in pixels 
    int dstPitch,                   // destination buffer pitch in pixels
    int is16f,                      // is fp16 or fp32 format
    int width,                      // horizontal image size in pixels
    int height,                     // vertical image size in lines
    const PontillismControls* algoGpuParams, // algorithm controls
    cudaStream_t stream
)
{
    // 1. Manage VRAM
    g_gpuCtx.CheckAndReallocate (width, height);

    // 2. Clear Atomic Counters
    // We need to reset the dot count to 0 for this frame
    cudaMemsetAsync (g_gpuCtx.d_counters, 0, sizeof(int), stream);

    // Normalize slider
    const float edge_sens = static_cast<float>(algoGpuParams->EdgeSensitivity) / 100.0f;

    dim3 blockDim(16, 32, 1);
    dim3 gridDim((width + blockDim.x - 1) / blockDim.x, (height + blockDim.y - 1) / blockDim.y, 1);

    k_Preprocess_Fused <<< gridDim, blockDim, 0, stream >>>
    (
        inBuffer,
        g_gpuCtx.d_densityMap,
        width, height,
        srcPitch,
        edge_sens
    );

    constexpr float base_prob = 0.025f;

    // Map Slider (0..100)
    float multiplier = 1.0f;
    if (algoGpuParams->DotDencity < 50)
    {
        multiplier = 0.1f + (static_cast<float>(algoGpuParams->DotDencity) / 50.0f) * 0.9f;
    }
    else
    {
        multiplier = 1.0f + ((static_cast<float>(algoGpuParams->DotDencity) - 50.0f) / 50.0f) * 3.0f;
    }

    float final_prob_scale = base_prob * multiplier;

    // Block dimensions tailored for occupancy on Pascal
    dim3 blockP2(32, 16, 1); // 512 threads per block
    dim3 gridP2((width + blockP2.x - 1) / blockP2.x, (height + blockP2.y - 1) / blockP2.y, 1);

    k_Seeding_WarpAggregated <<< gridP2, blockP2, 0, stream >>>
    (
        g_gpuCtx.d_densityMap,
        g_gpuCtx.d_dots,
        g_gpuCtx.d_counters, // Counter at index 0 holds the total count
        width,
        height,
        final_prob_scale,
        algoGpuParams->RandomSeed,
        MAX_GPU_DOTS
    );

    // --- PHASE 3: GEOMETRIC REFINEMENT (JFA) ---
    // 1. Calculate max step
    int max_dim = (width > height) ? width : height;
    int step = 1;
    while (step < max_dim) step <<= 1;
    step >>= 1;

    // 2. Grid for Full Screen Processing
    dim3 block2D(32, 16, 1); // 512 threads
    dim3 grid2D((width + block2D.x - 1) / block2D.x, (height + block2D.y - 1) / block2D.y, 1);

    // 3. Grid for Dots (1D)
    int max_dots = MAX_GPU_DOTS; // Or read back value if needed, but MAX is safe
    dim3 blockDot(256, 1, 1);
    dim3 gridDot((max_dots + 255) / 256, 1, 1);

    // 4. Initialize Buffer A (Ping)
    k_JFA_Clear <<< grid2D, block2D, 0, stream >>>
    (
        g_gpuCtx.d_jfaPing, width * height
    );

    // 5. Splat Seeds into Buffer A
    k_JFA_Splat <<< gridDot, blockDot, 0, stream >>>
    (
        g_gpuCtx.d_dots,
        g_gpuCtx.d_counters,
        g_gpuCtx.d_jfaPing,
        width,
        height
    );

    // 6. The Loop
    JFACell* src = g_gpuCtx.d_jfaPing;
    JFACell* dst = g_gpuCtx.d_jfaPong;

    while (step >= 1)
    {
        k_JFA_Step <<<grid2D, block2D, 0, stream >>>
        (
            src, dst,
            width, height,
            step
        );

        // Swap Pointers
        JFACell* tmp = src; src = dst; dst = tmp;
        step >>= 1;
    }

    // RESULT: 'src' now points to the final valid Voronoi Map.


    return;
}
