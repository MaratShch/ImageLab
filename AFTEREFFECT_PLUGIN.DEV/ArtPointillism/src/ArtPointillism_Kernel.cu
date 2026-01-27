#include "ArtPointillism_GPU.hpp"
#include "CompileTimeUtils.hpp"
#include "ImageLabCUDA.hpp"
#include "ArtPointillismKernel.cuh"
#include "PainterFactory.hpp"
#include <algorithm>
#include <cmath>


// Static Context (Singleton style for simplicity)
static GpuContext g_gpuCtx;


// High-Quality PCG Hash (Stateless)
// Significantly better distribution for seeding than Murmur.
__device__ inline uint32_t pcg_hash (uint32_t input) noexcept
{
    const uint32_t state = input * 747796405u + 2891336453u;
    const uint32_t word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    return (word >> 22u) ^ word;
}

// Generate deterministic float [0.0, 1.0)
__device__ inline float random_float (int x, int y, int seed) noexcept
{
    // Mix coordinates to break grid patterns
    const uint32_t coord_hash = static_cast<uint32_t>(x) * 131u + static_cast<uint32_t>(y) * 65521u;
    const uint32_t h = pcg_hash(coord_hash ^ static_cast<uint32_t>(seed));
    return static_cast<float>(h) * 2.3283064e-10f; // 1.0 / 2^32
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
    const float w_edge = edgeSensitivity; // Already normalized 0..1 by host
    const float w_luma = 1.0f - w_edge;

    float final_density = (L_inv * w_luma) + (edge_mag * w_edge);

    // Apply "Lifted Floor" logic (from our CPU lessons)
    // Prevent zero-density holes
    final_density = fmaxf(final_density, 0.25f);
    final_density = fminf(final_density, 1.0f);

    // 4. Calculate Orientation (for Van Gogh Flow)
    // Angle perpendicular to gradient
    float angle = atan2f(Gy, Gx) + 1.570796f; // + 90 deg

    // 5. Store
    int idx = y * width + x;
    dstMap[idx] = make_float2(final_density, angle);

    return;
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


__global__ void k_JFA_Clear
(
    JFACell* RESTRICT grid,
    int width, 
    int height
)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x >= width || y >= height) return;

    const int idx = y * width + x;

    // ID = -1, Pos = Off-screen, Dist = Infinite
    grid[idx] = pack_jfa(-1, -10000.0f, -10000.0f, 1.0e20f);

    return;
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
    const JFACell* RESTRICT jfaMap,
    const float*   RESTRICT srcInputRaw, // Adobe Input Buffer (Raw Float Ptr)
    int            srcPitchBytes,            // Pitch in Bytes
    DotColorAccumulator* RESTRICT dotAccum, // Output: Sums
    int width, 
    int height
)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    const int idx = y * width + x;

    // 1. Read JFA to find owner
    // If we are on a pixel that wasn't claimed, skip.
    int dot_id = unpack_id(jfaMap[idx]);

    if (dot_id != -1)
    {
        // 2. Read Source Pixel (Correct Pitch Math)
        const float4* row_ptr = (const float4*)((const char*)srcInputRaw + y * srcPitchBytes);
        float4 px = row_ptr[x];

        // 3. Convert BGRA -> Lab
        // Adobe 32f is B,G,R,A. Helper expects R,G,B.
        float3 rgb = make_float3(px.z, px.y, px.x);
        float3 lab = rgb_to_lab(rgb);

        // 4. Atomic Accumulation
        // (dotAccum is tightly packed struct of floats)
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
        int px = std::min(std::max((int)d.pos_x, 1), width - 2);
        int py = std::min(std::max((int)d.pos_y, 1), height - 2);

        // Read Gradient from Density Map
        // (Assuming you stored Angle in .y of DensityInfo in Phase 1?
        // If so, we just read it directly! No need to recalc Sobel.)
        const int mapIdx = py * width + px;
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


// 0 = Normal
// 1 = Show Voronoi Cells (Random colors based on Dot ID)
// 2 = Show Geometry (Red = Inside Dot, Blue = Outside/Background)
// 3 = Show Palette Index (Grayscale based on Color Index)

__global__ void k_Render_Final_Gather(
    const JFACell*       RESTRICT jfaMap,
    const GPUDot*        RESTRICT dots,
    const DotRenderInfo* RESTRICT dotInfo,
    const float4*        RESTRICT palette,
    const float*         RESTRICT srcInputRaw,
    float*               RESTRICT dstOutputRaw,
    int width,
    int height,
    int srcPitchBytes,
    int dstPitchBytes,
    float computedRadius,
    int strokeShape,
    bool longStroke,
    int backgroundMode,
    float opacity
) {
    // 1. Thread Coords
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;

    const int idx = y * width + x;

    // --- STEP A: READ SOURCE (To capture Original Alpha) ---
    // Adobe 32f buffer is float4 (B, G, R, A) or similar
    const float4* row_in = (const float4*)((const char*)srcInputRaw + y * srcPitchBytes);
    float4 src_px = row_in[x];

    // CAPTURE ORIGINAL ALPHA (Preserve strict pass-through by default)
    float final_alpha = src_px.w;

    // Convert source BGR -> RGB -> Lab (Used later for 'Source Background' mode or Blending)
    float3 src_rgb = make_float3(src_px.z, src_px.y, src_px.x);

    // --- STEP B: DETERMINE OWNERSHIP ---
    int dot_id = unpack_id(jfaMap[idx]);

    float3 final_lab; // We work in Lab space
    bool painted = false;

    if (dot_id != -1)
    {
        GPUDot d = dots[dot_id];
        DotRenderInfo info = dotInfo[dot_id];

        float dx = (float)x - d.pos_x;
        float dy = (float)y - d.pos_y;
        float base_r = computedRadius;

        bool inside = false;

        // Geometry Checks
        if (strokeShape == UnderlyingType(StrokeShape::ART_POINTILLISM_SHAPE_CIRCLE))
        { // Circle
            if ((dx*dx + dy*dy) <= (base_r * base_r)) inside = true;
        }
        else
        {
            float cos_a = cosf(info.orientation);
            float sin_a = sinf(info.orientation);
            float u = dx * cos_a + dy * sin_a;
            float v = -dx * sin_a + dy * cos_a;

            if (strokeShape == UnderlyingType(StrokeShape::ART_POINTILLISM_SHAPE_SQUARE))
            { // Square
                inside = (fabsf(u) < base_r && fabsf(v) < base_r);
            }
            else if (strokeShape == UnderlyingType(StrokeShape::ART_POINTILLISM_SHAPE_ELLIPSE))
            { // Ellipse (Van Gogh)
                float a_axis, b_axis;
                if (true == longStroke)
                {
                    a_axis = base_r * 3.0f;
                    b_axis = base_r * 0.45f;
                }
                else
                {
                    a_axis = base_r * 1.7f;
                    b_axis = base_r * 0.5f;
                }
                if (((u*u) / (a_axis*a_axis) + (v*v) / (b_axis*b_axis)) <= 1.0f)
                    inside = true;
            }
        }

        // Coloring
        if (inside)
        {
            // FIX: Coherent Coloring (Randomness based on Dot ID, not Pixel X,Y)
            // 0x5EE1 is a magic seed to scramble the bits
            float roll = random_float(dot_id, dot_id, 0x5EE1);

            int c_idx = (roll < info.ratio) ? info.colorIndex1 : info.colorIndex2;
            c_idx = std::max(0, std::min(c_idx, 31)); // Safety clamp

            float4 pal_col = palette[c_idx];
            final_lab = make_float3(pal_col.x, pal_col.y, pal_col.z);
            painted = true;
        }
}

    // --- STEP C: BACKGROUND LOGIC ---
    if (!painted)
    {
        if (backgroundMode == 1)
        { // White
            final_lab = make_float3(100.0f, 0.0f, 0.0f);
        }
        else if (backgroundMode == 2)
        { // Source Image
          // Convert pre-loaded Source RGB to Lab on the fly
            final_lab = rgb_to_lab(src_rgb);
        }
        else if (backgroundMode == 3)
        { // Transparent
          // VISIBLE VOID: L=0, Alpha = 0.
            final_lab = make_float3(0, 0, 0);
            final_alpha = 0.0f; // <--- The ONLY case where we explicitly modify alpha to 0
        }
        else
        { // Canvas (Cream) - Default
            final_lab = make_float3(96.0f, 2.0f, 8.0f);
        }
    }

    // --- STEP D: OUTPUT CONVERSION ---
    // 1. Lab -> Linear RGB
    float3 rgb_linear = lab_to_rgb_linear(final_lab);

    // 2. Opacity Blend (Mixing generated art back with Original Source)
    if (opacity > 0.0f)
    {
        float factor = opacity / 100.0f;
        // Mix Processed RGB with Original Source RGB (loaded in Step A)
        rgb_linear.x = rgb_linear.x * (1.0f - factor) + src_rgb.x * factor;
        rgb_linear.y = rgb_linear.y * (1.0f - factor) + src_rgb.y * factor;
        rgb_linear.z = rgb_linear.z * (1.0f - factor) + src_rgb.z * factor;
    }

    // 3. Write Output
    // Re-pack into BGR format, PRESERVING 'final_alpha'
    float4 out_px;
    out_px.x = rgb_linear.z; // B
    out_px.y = rgb_linear.y; // G
    out_px.z = rgb_linear.x; // R
    out_px.w = final_alpha;  // A (Either Source A, or 0.0 if transparent mode)

    float4* row_out = (float4*)((char*)dstOutputRaw + y * dstPitchBytes);
    row_out[x] = out_px;

    return;
}


#include "ArtPointillismKernelDBG.cuh"


CUDA_KERNEL_CALL
void ArtPointillism_CUDA
(
    const float* RESTRICT inBuffer, // source (input) buffer
    float* RESTRICT outBuffer,      // destination (output) buffer
    int srcPitch,                   // source buffer pitch in PIXELS 
    int dstPitch,                   // destination buffer pitch in PIXELS
    int width,                      // horizontal image size in pixels
    int height,                     // vertical image size in lines
    const PontillismControls* algoGpuParams, // algorithm controls
    int frameCounter,
    cudaStream_t stream
)
{
    // 0. Read Algorithm Control Parameters
    const ArtPointillismPainter PainterStyle    = algoGpuParams->PainterStyle;
    const int32_t               DotDencity      = algoGpuParams->DotDencity;
    const int32_t               DotSize         = algoGpuParams->DotSize;
    const int32_t               EdgeSensitivity = algoGpuParams->EdgeSensitivity;
    const int32_t               Vibrancy        = algoGpuParams->Vibrancy;
    const BackgroundArt         Background      = algoGpuParams->Background;
    const int32_t               Opacity         = algoGpuParams->Opacity;
    const int32_t               RandomSeed      = algoGpuParams->RandomSeed;

    // -------------------------------------------------------------------------
    // 1. SETUP & MEMORY
    // -------------------------------------------------------------------------

    // Convert Pixel Pitch to Byte Pitch for pointer arithmetic in kernels
    const int srcPitchBytes = srcPitch * sizeof(float4);
    const int dstPitchBytes = dstPitch * sizeof(float4);

    // Manage VRAM (Lazy Reallocation)
    g_gpuCtx.CheckAndReallocate(width, height);

    // Clear Atomic Counters (Reset dot count to 0)
    cudaMemsetAsync(g_gpuCtx.d_counters, 0, sizeof(int), stream);

    // -------------------------------------------------------------------------
    // 2. PALETTE MANAGEMENT
    // -------------------------------------------------------------------------

    // Retrieve the CPU Painter Strategy
    IPainter* painter = GetPainterRegistry(PainterStyle);

    // Extract Data
    RenderContext cpu_ctx;
    painter->SetupContext(cpu_ctx);

    // Upload to GPU (Only happens if style changed)
    g_gpuCtx.UpdatePaletteFromPlanar
    (
        cpu_ctx.pal_L,
        cpu_ctx.pal_a,
        cpu_ctx.pal_b,
        cpu_ctx.palette_size,
        static_cast<int>(PainterStyle),
        stream
    );

    // -------------------------------------------------------------------------
    // 3. PHASE 1: PREPROCESS (Density & Structure)
    // -------------------------------------------------------------------------

    const float edge_sens = static_cast<float>(EdgeSensitivity) / 100.0f;

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

    // -------------------------------------------------------------------------
    // 4. MATH: DOT BUDGET & RADIUS CALCULATION (TUNED)
    // -------------------------------------------------------------------------

    // A. BOOST THE BASE DENSITY
    constexpr float base_dots_per_block = 350.0f;

    // B. Calculate Multiplier (Same as before)
    float density_val = static_cast<float>(DotDencity);
    float count_multiplier = 1.0f;
    if (density_val < 50.0f)
    {
        count_multiplier = 0.1f + (density_val / 50.0f) * 0.9f;
    }
    else
    {
        count_multiplier = 1.0f + ((density_val - 50.0f) / 50.0f) * 3.0f;
    }

    // Probability for Seeding
    constexpr float base_prob = base_dots_per_block / 10000.0f;
    float final_prob_scale = base_prob * count_multiplier;

    // C. Calculate Dynamic Radius
    const float area_factor = (static_cast<float>(width * height)) / 10000.0f;
    const float total_expected_dots = std::min(static_cast<float>(MAX_GPU_DOTS), area_factor * base_dots_per_block * count_multiplier);
    const float avg_spacing = std::sqrt((static_cast<float>(width * height)) / total_expected_dots);

    // D. BOOST THE OVERLAP
    // We want dots to comfortably overlap neighbors to hide the canvas.
    // Old Start: 0.6. New Start: 0.85.
    float size_val = static_cast<float>(DotSize);
    float overlap_factor = 0.85f + (size_val / 100.0f) * 1.5f;

    // Final Radius
    // Remove the 0.8 reduction. Let them be full size.
    float computedRadius = avg_spacing * overlap_factor;
    if (computedRadius < 2.5f) computedRadius = 2.5f;

    // -------------------------------------------------------------------------
    // 5. PHASE 2: SEEDING (Warp Aggregated)
    // -------------------------------------------------------------------------
    const int time_variant_seed = RandomSeed + frameCounter;

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
        time_variant_seed,
        MAX_GPU_DOTS
    );

    // -------------------------------------------------------------------------
    // 6. PHASE 3: GEOMETRIC REFINEMENT (JFA)
    // -------------------------------------------------------------------------

    const int max_dim = (width > height) ? width : height;
    int step = 1;
    while (step < max_dim) step <<= 1;
    step >>= 1;

    int max_dots_alloc = MAX_GPU_DOTS;
    dim3 blockDot(256, 1, 1);
    dim3 gridDot((max_dots_alloc + 255) / 256, 1, 1);

    // Init Grid
    k_JFA_Clear <<< gridDim, blockDim, 0, stream >>> (g_gpuCtx.d_jfaPing, width, height);

    // Splat Seeds
    k_JFA_Splat <<< gridDot, blockDot, 0, stream >>>
    (
        g_gpuCtx.d_dots,
        g_gpuCtx.d_counters,
        g_gpuCtx.d_jfaPing,
        width,
        height
    );

    // Ping-Pong Loop
    JFACell* src = g_gpuCtx.d_jfaPing;
    JFACell* dst = g_gpuCtx.d_jfaPong;

    while (step >= 1)
    {
        k_JFA_Step <<< gridDim, blockDim, 0, stream >>> (src, dst, width, height, step);

        // Swap
        JFACell* tmp = src; src = dst; dst = tmp;
        step >>= 1;
    }

    // Capture the final valid map
    JFACell* final_voronoi_map = src;

    // -------------------------------------------------------------------------
    // 7. PHASE 4: ARTISTIC RENDERING
    // -------------------------------------------------------------------------

    // 4.A: Integrate Colors
    cudaMemsetAsync(g_gpuCtx.d_dotColors, 0, MAX_GPU_DOTS * sizeof(DotColorAccumulator), stream);

    k_Integrate_Colors_Atomic <<< gridDim, blockDim, 0, stream >>>
    (
        final_voronoi_map,    // Map
        inBuffer,             // Raw Input Pointer
        srcPitchBytes,        // Correct Byte Pitch
        g_gpuCtx.d_dotColors, // Accumulator
        width,
        height
    );

    // 4.B: Decompose & Attributes
    const int shape_id = static_cast<int>(cpu_ctx.shape);
    const int mode_id  = static_cast<int>(cpu_ctx.color_mode);

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
        static_cast<float>(Vibrancy),
        mode_id,
        shape_id
    );

    // 4.C: Final Gather (Drawing)
    const float opacity_val = static_cast<float>(Opacity);

    k_Render_Final_Gather <<< gridDim, blockDim, 0, stream >>>
    (
        final_voronoi_map,
        g_gpuCtx.d_dots,
        g_gpuCtx.d_dotInfo,
        g_gpuCtx.d_palette,
        inBuffer,
        outBuffer,
        width,
        height,
        srcPitchBytes,
        dstPitchBytes,
        computedRadius,
        shape_id,
        (PainterStyle == ArtPointillismPainter::ART_POINTILLISM_PAINTER_VAN_GOGH),
        static_cast<int>(Background),
        opacity_val
    );

    // Wait for stream to finish (optional for debug, remove for max async performance)
    cudaDeviceSynchronize();

    return;
}