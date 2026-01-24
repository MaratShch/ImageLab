#include "ArtPointillism_GPU.hpp"
#include "CompileTimeUtils.hpp"
#include "ImageLabCUDA.hpp"
#include "ArtPointillismKernel.cuh"
#include "PainterFactory.hpp"
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


__device__ float3 lab_to_rgb_linear (float3 lab) noexcept
{
    float L = lab.x;
    float a = lab.y;
    float b = lab.z;

    // --- 1. Lab -> XYZ ---
    // Constants
    const float D65_Xn = 0.95047f;
    const float D65_Yn = 1.00000f;
    const float D65_Zn = 1.08883f;
    const float delta = 6.0f / 29.0f;
    const float epsilon = 0.008856f; // delta^3

    float fy = (L + 16.0f) / 116.0f;
    float fx = fy + (a / 500.0f);
    float fz = fy - (b / 200.0f);

    float fx3 = fx * fx * fx;
    float fz3 = fz * fz * fz;

    float X = (fx3 > epsilon) ? fx3 : ((fx - 16.0f / 116.0f) / 7.787f);
    float Y = (L > 8.0f) ? (fy * fy * fy) : (L / 903.3f);
    float Z = (fz3 > epsilon) ? fz3 : ((fz - 16.0f / 116.0f) / 7.787f);

    X *= D65_Xn;
    Y *= D65_Yn;
    Z *= D65_Zn;

    // --- 2. XYZ -> Linear RGB (sRGB Matrix) ---
    float R = 3.2404542f * X - 1.5371385f * Y - 0.4985314f * Z;
    float G = -0.9692660f * X + 1.8760108f * Y + 0.0415560f * Z;
    float B = 0.0556434f * X - 0.2040259f * Y + 1.0572252f * Z;

    // Saturate (Clamp to 0..1 to prevent weird highlights)
    R = fminf(fmaxf(R, 0.0f), 1.0f);
    G = fminf(fmaxf(G, 0.0f), 1.0f);
    B = fminf(fmaxf(B, 0.0f), 1.0f);

    return make_float3(R, G, B);
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


__global__ void k_Integrate_Colors_Atomic
(
    const JFACell* __restrict__ jfaMap, // From Phase 3
    const float4*  __restrict__ srcLab, // Input Image (Pre-converted to Lab in Phase 1? Or read RGB and convert?)
                                        // Better: Read Original Input (float4 BGRA) and convert to Lab on fly to save VRAM.
                                        // Let's assume we read the original Input Buffer passed to Render.
    const float*   __restrict__ srcInputRaw, // Adobe Input Buffer
    int            srcPitchBytes,
    DotColorAccumulator* __restrict__ dotAccum, // Output: Sums
    int width, int height
)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    int idx = y * width + x;

    // 1. Read JFA to find owner
    int dot_id = unpack_id(jfaMap[idx]);

    // Safety check
    if (dot_id != -1) {
        // 2. Read Source Pixel
        const float4* row_ptr = (const float4*)((const char*)srcInputRaw + y * srcPitchBytes);
        float4 px = row_ptr[x];

        // Convert RGB -> Lab (Inline helper we defined in Phase 1)
        float3 lab = rgb_to_lab(make_float3(px.z, px.y, px.x)); // BGRA -> RGB

                                                                // 3. Atomic Accumulation
                                                                // Note: CUDA doesn't have atomicAdd for float4 or float3 structs.
                                                                // We must do component-wise atomics.
        atomicAdd(&dotAccum[dot_id].x, lab.x);
        atomicAdd(&dotAccum[dot_id].y, lab.y);
        atomicAdd(&dotAccum[dot_id].z, lab.z);
        atomicAdd(&dotAccum[dot_id].w, 1.0f); // Count
    }
    return;
}


__global__ void k_Decompose_Attributes
(
    const DotColorAccumulator* RESTRICT dotAccum,
    const GPUDot*              RESTRICT dotPos,
    const DensityInfo*         RESTRICT densityMap, // For Gradient/Orientation
    const float4*              RESTRICT palette,    // Loaded in GpuContext
    DotRenderInfo*             RESTRICT dotInfo,
    int*                       RESTRICT counterPtr, // d_counters[0]
    int width, int height,
    int paletteSize,
    float vibrancy,
    int colorMode,  // 0=Scientific, 1=Expressive
    int strokeShape // 2=Ellipse (Van Gogh)
) {
    const int id = blockIdx.x * blockDim.x + threadIdx.x;
    const int count = counterPtr[0];

    if (id >= count) return;

    // 1. Get Average Color
    float4 acc = dotAccum[id];
    if (acc.w < 1.0f)
    {
        // Dead dot (no pixels). Set to dummy.
        dotInfo[id] = { 0, 0, 0.0f, 0.0f };
        return;
    }

    float inv_n = 1.0f / acc.w;
    float3 target = make_float3(acc.x * inv_n, acc.y * inv_n, acc.z * inv_n);

    // 2. Apply Color Mode (Vibrancy)
    // (Ported from your C++ logic)
    float chroma = sqrtf(target.y * target.y + target.z * target.z);
    float boost = 1.0f + (vibrancy / 100.0f);

    if (colorMode == 1)
    { // Expressive
        boost *= 1.5f;
        // Gray Killer Logic
        if (chroma < 10.0f && chroma > 0.5f)
        {
            float fake = 20.0f + (vibrancy * 0.2f);
            float scale = fake / chroma;
            target.y *= scale; target.z *= scale;
        }
    }
    target.y *= boost; target.z *= boost;

    // 3. Palette Matching (Brute Force 24-32 is instant on GPU)
    int p1 = 0; float d1 = 1.0e20f;
    int p2 = 0; float d2 = 1.0e20f;

    for (int i = 0; i < paletteSize; ++i)
    {
        float4 pal = palette[i]; // .w is padding
        float dL = target.x - pal.x;
        float da = target.y - pal.y;
        float db = target.z - pal.z;
        float dist = dL*dL + da*da + db*db;

        if (dist < d1)
        {
            d2 = d1; p2 = p1;
            d1 = dist; p1 = i;
        }
        else if (dist < d2)
        {
            d2 = dist; p2 = i;
        }
    }

    float rd1 = sqrtf(d1);
    float rd2 = sqrtf(d2);
    float ratio = (rd1 + rd2 < 0.001f) ? 1.0f : (rd2 / (rd1 + rd2));

    // 4. Orientation (Van Gogh Flow)
    float angle = 0.0f;
    if (strokeShape == 2)
    { // Oriented Ellipse
        GPUDot d = dotPos[id];
        int px = min(max((int)d.pos_x, 1), width - 2);
        int py = min(max((int)d.pos_y, 1), height - 2);

        // Read Gradient from Density Map
        // (Assuming you stored Angle in .y of DensityInfo in Phase 1?
        // If so, we just read it directly! No need to recalc Sobel.)
        int mapIdx = py * width + px;
        angle = densityMap[mapIdx].y; // Phase 1 stored this!
    }
    else if (strokeShape == 1)
    {   // Mosaic
        // Random jitter for squares (using hash)
        angle = (random_float(id, 0, 1234) - 0.5f) * 0.5f; // +/- ~15 deg
    }

    // 5. Store
    DotRenderInfo info;
    info.colorIndex1 = p1;
    info.colorIndex2 = p2;
    info.ratio = ratio;
    info.orientation = angle;
    dotInfo[id] = info;

    return;
}


__global__ void k_Render_Final_Gather
(
    const JFACell*       RESTRICT jfaMap,
    const GPUDot*        RESTRICT dots,
    const DotRenderInfo* RESTRICT dotInfo,
    const float4*        RESTRICT palette,
    const float*         RESTRICT srcInputRaw, // For Blending
    float*               RESTRICT dstOutputRaw, // Adobe Output
    int width,
    int height,
    int srcPitchBytes,
    int dstPitchBytes,
    float dotSizeSlider, // 0..100
    int strokeShape,     // 0=Circle, 1=Square, 2=Ellipse
    int backgroundMode,
    float opacity        // 0..100
)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;

    const int idx = y * width + x;

    // 1. Get Owner
    int dot_id = unpack_id(jfaMap[idx]);

    float3 final_lab;
    float alpha_out = 1.0f; // Default opaque

    bool painted = false;

    if (dot_id != -1)
    {
        // 2. Get Dot Geometry
        GPUDot d = dots[dot_id];
        DotRenderInfo info = dotInfo[dot_id];

        float dx = (float)x - d.pos_x;
        float dy = (float)y - d.pos_y;

        // 3. Determine Radius/Size
        // (Simplified logic: Base size 3.0 + slider)
        // Ideally, pass pre-calculated "Base Radius" derived from dot count
        float base_r = 3.0f + (dotSizeSlider / 20.0f); // Tuning needed here to match CPU

        bool inside = false;

        // 4. Shape Check
        if (strokeShape == 0)
        { // Circle
            if ((dx*dx + dy*dy) <= (base_r * base_r)) inside = true;
        }
        else
        {
            // Rotate
            float cos_a = cosf(info.orientation);
            float sin_a = sinf(info.orientation);
            float u = dx * cos_a + dy * sin_a;
            float v = -dx * sin_a + dy * cos_a;

            if (strokeShape == 1)
            { // Square
                inside = (fabsf(u) < base_r && fabsf(v) < base_r);
            }
            else if (strokeShape == 2)
            { // Ellipse
                float a_axis = base_r * 1.5f; // Long
                float b_axis = base_r * 0.4f; // Short
                if (((u*u) / (a_axis*a_axis) + (v*v) / (b_axis*b_axis)) <= 1.0f) inside = true;
            }
        }

        // 5. Coloring
        if (inside)
        {
            // Stochastic Mix (Deterministic per pixel based on ID/Pos)
            float roll = random_float(x, y, dot_id);
            int c_idx = (roll < info.ratio) ? info.colorIndex1 : info.colorIndex2;

            float4 pal_col = palette[c_idx];
            final_lab = make_float3(pal_col.x, pal_col.y, pal_col.z);
            painted = true;
        }
    }

    // 6. Background Handling
    if (!painted)
    {
        if (backgroundMode == 1)
        { // White
            final_lab = make_float3(100.0f, 0.0f, 0.0f);
        }
        else if (backgroundMode == 2)
        { // Source (Use input Lab)
                                        // We need to read input again
            const float4* row_in = (const float4*)((const char*)srcInputRaw + y * srcPitchBytes);
            float4 px = row_in[x];
            final_lab = rgb_to_lab(make_float3(px.z, px.y, px.x));
        }
        else if (backgroundMode == 3)
        { // Transparent
            final_lab = make_float3(0, 0, 0);
            alpha_out = 0.0f;
        }
        else
        { // Canvas (Cream)
            final_lab = make_float3(96.0f, 2.0f, 8.0f);
        }
    }

    // 7. Lab -> RGB Conversion
    // (Standard D65 conversion logic here - reusing your known math)
    float3 rgb_linear = lab_to_rgb_linear(final_lab); // Implement this device helper

                                                      // 8. Output to Adobe Buffer
                                                      // Adobe 32f is usually B, G, R, A (Linear or Gamma? Pr assumes Linear usually for 32f)
                                                      // We assume Linear output.

                                                      // Blending with Original (Opacity Slider)
    if (opacity > 0.0f)
    {
        const float4* row_in = (const float4*)((const char*)srcInputRaw + y * srcPitchBytes);
        float4 src_px = row_in[x];
        float3 src_rgb = make_float3(src_px.z, src_px.y, src_px.x); // BGR->RGB

        float factor = opacity / 100.0f;
        rgb_linear.x = rgb_linear.x * (1.0f - factor) + src_rgb.x * factor;
        rgb_linear.y = rgb_linear.y * (1.0f - factor) + src_rgb.y * factor;
        rgb_linear.z = rgb_linear.z * (1.0f - factor) + src_rgb.z * factor;
    }

    float4 out_px;
    out_px.x = rgb_linear.z; // B
    out_px.y = rgb_linear.y; // G
    out_px.z = rgb_linear.x; // R
    out_px.w = alpha_out;    // A

    float4* row_out = (float4*)((char*)dstOutputRaw + y * dstPitchBytes);
    row_out[x] = out_px;
}


__global__ void k_Debug_SolidRed
(
    float* RESTRICT dst,
    int width,
    int height,
    int dstPitchBytes
)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    // Calculate pointer to this pixel
    float4* row = (float4*)((char*)dst + y * dstPitchBytes);

    // Write Solid RED
    // Adobe 32f is usually B, G, R, A
    row[x] = make_float4(0.0f, 0.0f, 1.0f, 1.0f);
    return;
}


__global__ void k_Debug_Grid
(
    float* __restrict__ dst,
    int width, int height,
    int dstPitchBytes
)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    // Calculate pointer
    float4* row = (float4*)((char*)dst + y * dstPitchBytes);

    float r = 0.0f;
    // Draw Grid every 50 pixels
    if ((x % 50) == 0 || (y % 50) == 0)
    {
        r = 1.0f; // Red Lines
    }

    row[x] = make_float4(0.0f, 0.0f, r, 1.0f);
}


__global__ void k_Debug_Passthrough
(
    const float* __restrict__ src,
    float*       __restrict__ dst,
    int width, int height,
    int srcPitchBytes, int dstPitchBytes
)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;

    // Read Source using srcPitch
    float4 px = *(const float4*)((const char*)src + y * srcPitchBytes + x * sizeof(float4));

    // Write Dest using dstPitch
    float4* out = (float4*)((char*)dst + y * dstPitchBytes + x * sizeof(float4));

    *out = px;
}




CUDA_KERNEL_CALL
void ArtPointillism_CUDA
(
    const float* RESTRICT inBuffer,
    float* RESTRICT outBuffer,
    int srcPitch, // Input in PIXELS
    int dstPitch, // Input in PIXELS
    int is16f,
    int width,
    int height,
    const PontillismControls* algoGpuParams,
    cudaStream_t stream
)
{
    // 1. Calculate Bytes for Kernels
    const int srcPitchBytes = srcPitch * sizeof(float4);
    const int dstPitchBytes = dstPitch * sizeof(float4);

    // Manage VRAM
    g_gpuCtx.CheckAndReallocate (width, height);

    // Clear Counters
    cudaMemsetAsync (g_gpuCtx.d_counters, 0, sizeof(int), stream);

    // --- PALETTE ---
    IPainter* painter = GetPainterRegistry(algoGpuParams->PainterStyle);
    RenderContext cpu_ctx;
    painter->SetupContext(cpu_ctx);

    g_gpuCtx.UpdatePaletteFromPlanar
    (
        cpu_ctx.pal_L, 
        cpu_ctx.pal_a, 
        cpu_ctx.pal_b,
        cpu_ctx.palette_size,
        static_cast<int>(algoGpuParams->PainterStyle),
        stream
    );

    // --- PHASE 1: PREPROCESS ---
    const float edge_sens = static_cast<float>(algoGpuParams->EdgeSensitivity) / 100.0f;
    dim3 blockDim(16, 32, 1);
    dim3 gridDim((width + blockDim.x - 1) / blockDim.x, (height + blockDim.y - 1) / blockDim.y, 1);

    k_Preprocess_Fused <<< gridDim, blockDim, 0, stream >>>
    (
        inBuffer,
        g_gpuCtx.d_densityMap,
        width,
        height,
        srcPitchBytes,
        edge_sens
    );

    // --- PHASE 2: SEEDING ---
    constexpr float base_prob = 0.025f;
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

    dim3 blockP2(32, 16, 1);
    dim3 gridP2((width + blockP2.x - 1) / blockP2.x, (height + blockP2.y - 1) / blockP2.y, 1);

    k_Seeding_WarpAggregated <<< gridP2, blockP2, 0, stream >>>
    (
        g_gpuCtx.d_densityMap,
        g_gpuCtx.d_dots,
        g_gpuCtx.d_counters,
        width,
        height,
        final_prob_scale,
        algoGpuParams->RandomSeed,
        MAX_GPU_DOTS
    );

    // --- PHASE 3: JFA ---
    int max_dim = (width > height) ? width : height;
    int step = 1;
    while (step < max_dim) step <<= 1;
    step >>= 1;

    int max_dots = MAX_GPU_DOTS;
    dim3 blockDot(256, 1, 1);
    dim3 gridDot((max_dots + 255) / 256, 1, 1);

    k_JFA_Clear <<< gridDim, blockDim, 0, stream >>>
    (
        g_gpuCtx.d_jfaPing, width * height
    );

    k_JFA_Splat <<< gridDot, blockDot, 0, stream >>>
    (
        g_gpuCtx.d_dots,
        g_gpuCtx.d_counters,
        g_gpuCtx.d_jfaPing,
        width, height
    );

    JFACell* src = g_gpuCtx.d_jfaPing;
    JFACell* dst = g_gpuCtx.d_jfaPong;

    while (step >= 1)
    {
        k_JFA_Step <<< gridDim, blockDim, 0, stream >>>
        (
            src, dst, width, height, step
        );

        // Swap Pointers
        JFACell* tmp = src; src = dst; dst = tmp;
        step >>= 1;
    }

    // IMPORTANT: 'src' now points to the buffer containing the final valid map.
    // It could be Ping or Pong depending on the loop count.
    JFACell* final_voronoi_map = src;

    // --- PHASE 4: RENDERING ---

    // 1. Clear Color Accumulators
    cudaMemsetAsync(g_gpuCtx.d_dotColors, 0, MAX_GPU_DOTS * sizeof(DotColorAccumulator), stream);

    // 2. Integrate (Uses calculated Bytes Pitch and Correct Map)
    k_Integrate_Colors_Atomic <<< gridDim, blockDim, 0, stream >>>
    (
        final_voronoi_map,
        (const float4*)inBuffer,
        inBuffer,
        srcPitchBytes, 
        g_gpuCtx.d_dotColors,
        width, height
    );

    // 3. Decompose
    int shape_id = 0;
    if (algoGpuParams->Shape == StrokeShape::ART_POINTILLISM_SHAPE_SQUARE) shape_id = 1;
    if (algoGpuParams->Shape == StrokeShape::ART_POINTILLISM_SHAPE_ELLIPSE) shape_id = 2;

    int mode_id = 0;
    if (algoGpuParams->PainterStyle == ArtPointillismPainter::ART_POINTILLISM_PAINTER_VAN_GOGH ||
        algoGpuParams->PainterStyle == ArtPointillismPainter::ART_POINTILLISM_PAINTER_MATISSE)
    {
        mode_id = 1;
    }

    k_Decompose_Attributes <<< gridDot, blockDot, 0, stream >>>
    (
        g_gpuCtx.d_dotColors,
        g_gpuCtx.d_dots,
        g_gpuCtx.d_densityMap,
        g_gpuCtx.d_palette,
        g_gpuCtx.d_dotInfo,
        g_gpuCtx.d_counters,
        width,
        height,
        32,
        static_cast<float>(algoGpuParams->Vibrancy),
        mode_id,
        shape_id
    );

    // 4. Final Gather (Uses calculated Bytes Pitch)
    const float dot_size_val = static_cast<float>(algoGpuParams->DotSize);
    const float opacity_val = static_cast<float>(algoGpuParams->Opacity);

    k_Render_Final_Gather <<< gridDim, blockDim, 0, stream >>>
    (
        final_voronoi_map, // CORRECT: Use the result of JFA
        g_gpuCtx.d_dots,
        g_gpuCtx.d_dotInfo,
        g_gpuCtx.d_palette,
        inBuffer,
        outBuffer,
        width, height,
        srcPitchBytes,
        dstPitchBytes,
        dot_size_val,
        shape_id,
        static_cast<int>(algoGpuParams->Background),
        opacity_val
     );

    // Optional sync for debugging, remove for release
    cudaDeviceSynchronize();

    return;
}