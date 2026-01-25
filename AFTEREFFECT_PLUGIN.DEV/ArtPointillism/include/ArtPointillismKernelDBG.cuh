/////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////// DBG FUNCTIONS ////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////
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


// --- DEBUG KERNEL: Visualize Density Map ---
__global__ void k_Preprocess_Fused_DBG
(
    const float* RESTRICT srcPtr,      // Adobe Input
    DensityInfo* RESTRICT internalMap, // Internal Density Map
    float*       RESTRICT debugOutput, // Adobe Output (for visualization)
    int width,
    int height,
    int srcPitchBytes,
    int dstPitchBytes,                     // Needed to write to screen
    float edgeSensitivity
)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    // 1. Read Input (Standard Logic)
    const float4* row_in = (const float4*)((const char*)srcPtr + y * srcPitchBytes);
    float4 px = row_in[x];

    // 2. Convert to Lab
    // (Inline simplified conversion or call helper if defined)
    float r = px.z; float g = px.y; float b = px.x;
    // Standard Linear RGB -> Luma approx for speed in debug, or full Lab
    // Let's use the exact same logic as your production kernel:
    float X = 0.4124564f * r + 0.3575761f * g + 0.1804375f * b;
    float Y = 0.2126729f * r + 0.7151522f * g + 0.0721750f * b;
    float Z = 0.0193339f * r + 0.1191920f * g + 0.9503041f * b;

    // Normalize Y
    float yn = Y; // Assuming Y is 0..1 from linear RGB

                  // Lab f(t) for L only
    float fy = (yn > 0.008856f) ? cbrtf(yn) : (7.787f * yn + 16.0f / 116.0f);
    float L = (116.0f * fy) - 16.0f;
    float L_norm = L / 100.0f;
    float L_inv = 1.0f - L_norm;

    // 3. Sobel Edge Detection
    float Gx = 0.0f;
    float Gy = 0.0f;

    for (int dy = -1; dy <= 1; dy++) {
        int ny = min(max(y + dy, 0), height - 1);
        const float4* nrow = (const float4*)((const char*)srcPtr + ny * srcPitchBytes);

        for (int dx = -1; dx <= 1; dx++) {
            int nx = min(max(x + dx, 0), width - 1);
            float4 n_px = nrow[nx];
            // Simple Luma for edge check: 0.21R + 0.71G + 0.07B
            float n_luma = 0.2126f*n_px.z + 0.7152f*n_px.y + 0.0722f*n_px.x;

            if (dx != 0) {
                float w = (dy == 0) ? 2.0f : 1.0f;
                Gx += (dx * w * n_luma);
            }
            if (dy != 0) {
                float w = (dx == 0) ? 2.0f : 1.0f;
                Gy += (dy * w * n_luma);
            }
        }
    }

    float edge = sqrtf(Gx*Gx + Gy*Gy);

    // 4. Mix
    float w_edge = edgeSensitivity;
    float w_luma = 1.0f - w_edge;
    float density = (L_inv * w_luma) + (edge * w_edge);

    // 5. Store Internal (Standard Behavior)
    int idx = y * width + x;
    internalMap[idx].x = density;
    internalMap[idx].y = 0.0f; // Angle dummy

                               // --- DEBUG OUTPUT ---
                               // Write the Density directly to the screen as Gray.
                               // High Density (1.0) = White Pixel.
                               // Low Density (0.0) = Black Pixel.

    float4* row_out = (float4*)((char*)debugOutput + y * dstPitchBytes);
    row_out[x] = make_float4(density, density, density, 1.0f); // B, G, R, A

    return;
}

// --- DEBUG KERNEL 1: VISUALIZE DENSITY MAP (Phase 1 Check) ---
// Expected: A Grayscale, inverted version of your image.
// Failure:  Static noise, scratches, or pure black/white.
__global__ void k_Debug_ShowDensity
(
    const DensityInfo* RESTRICT densityMap,
    float*             RESTRICT dst,
    int width, int height,
    int dstPitchBytes
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;

    // Read Density (0.0 to 1.0)
    int idx = y * width + x;
    float d = densityMap[idx].x;

    // Write as Grayscale Green
    float4* row = (float4*)((char*)dst + y * dstPitchBytes);
    row[x] = make_float4(d, d, d, 1.0f);
}

// --- DEBUG KERNEL 2: VISUALIZE VORONOI MAP (Phase 2 & 3 Check) ---
// Expected: A "Stained Glass" / Mosaic look. Large colored cells.
// Failure:  Tiny pixel-sized noise (JFA failed), or Black (No seeds generated).
__global__ void k_Debug_ShowJFA
(
    const JFACell* RESTRICT jfaMap,
    float*         RESTRICT dst,
    int width, int height,
    int dstPitchBytes
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;

    int idx = y * width + x;
    int id = unpack_id(jfaMap[idx]);

    float4 color;
    if (id == -1) {
        color = make_float4(0, 0, 0, 1); // Black if no owner
    }
    else {
        // Hash the ID to get a random color
        float r = random_float(id, 0, 0);
        float g = random_float(id, 1, 0);
        float b = random_float(id, 2, 0);
        color = make_float4(r, g, b, 1.0f);
    }

    float4* row = (float4*)((char*)dst + y * dstPitchBytes);
    row[x] = color;
}


// --- DEBUG KERNEL: Visualize Decomposition Results ---
__global__ void k_Debug_ShowDecomposition
(
    const JFACell*       RESTRICT jfaMap,   // To find owner
    const DotRenderInfo* RESTRICT dotInfo,  // The result of Decompose
    const float4*        RESTRICT palette,  // To see the actual color
    float*               RESTRICT dst,      // Output
    int width, int height,
    int dstPitchBytes
)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    const int idx = y * width + x;

    // 1. Find Owner
    const int dot_id = unpack_id(jfaMap[idx]);

    float3 final_rgb;

    if (dot_id != -1)
    {
        // 2. Read Decomposition Result
        DotRenderInfo info = dotInfo[dot_id];

        // 3. Get the Primary Selected Palette Color
        int pal_idx = info.colorIndex1;

        // Safety Check
        if (pal_idx >= 0 && pal_idx < 32)
        { // 32 is max palette size
            float4 lab = palette[pal_idx];
            float3 lab3 = make_float3(lab.x, lab.y, lab.z);

            // Convert Lab Palette -> RGB for display
            final_rgb = lab_to_rgb_linear(lab3);
        }
        else
        {
            // Error: Invalid Index -> Magenta
            final_rgb = make_float3(1.0f, 0.0f, 1.0f);
        }
    }
    else
    {
        // No owner -> Black
        final_rgb = make_float3(0.0f, 0.0f, 0.0f);
    }

    // 4. Write Output
    float4* row = (float4*)((char*)dst + y * dstPitchBytes);

    // Output BGRA 1.0 Alpha
    row[x] = make_float4(final_rgb.z, final_rgb.y, final_rgb.x, 1.0f);

    return;
}

/////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////
