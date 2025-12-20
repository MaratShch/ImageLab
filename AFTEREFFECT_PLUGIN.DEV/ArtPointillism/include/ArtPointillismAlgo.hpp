#ifndef __IMAGE_LAB_ART_POINTILISM_ALGORITHM_DEFINITIONS__
#define __IMAGE_LAB_ART_POINTILISM_ALGORITHM_DEFINITIONS__

#include <algorithm>
#include <cstdint>
#include "CommonAuxPixFormat.hpp"
#include "ArtPointillismColorConvert.hpp"
#include "ArtPointillismPainters.hpp"
#include "FastAriphmetics.hpp"


inline void CIELab_LumaInvert (const fCIELabPix* RESTRICT pSrc, float* RESTRICT pLumaDst, A_long sizeX, A_long sizeY) noexcept
{
    // Luma result: 1.0 (White) becomes 0.0 (No Dots)
    //              0.0 (Black) becomes 1.0 (Max Dots)

    const ptrdiff_t lumaSize = static_cast<ptrdiff_t>(sizeX * sizeY);
    for (ptrdiff_t i = 0; i < lumaSize; i++)
        pLumaDst[i] = 1.0f - pSrc[i].L * 0.010f;
    return;
}


inline void LumaEdgeDetection
(
    const float* RESTRICT pSrc,
          float* RESTRICT pDst,
    A_long sizeX,
    A_long sizeY
) noexcept
{
    // Safety Check: If the image is too small for a 3x3 kernel, 
    // simply copy the original data to the destination and return.
    if (sizeX < 3 || sizeY < 3)
    {
        std::memcpy (pDst, pSrc, sizeX * sizeY * sizeof(float));
        return;
    }

    const ptrdiff_t stride = sizeX; // Assuming stride equals width (no padding)
    const size_t row_size_bytes = static_cast<size_t>(sizeX) * sizeof(float);

    // ---------------------------------------------------------
    // 1. TOP BORDER HANDLING
    // ---------------------------------------------------------
    // Copy the entire first row directly from source to destination.
    // std::memcpy is highly optimized (often uses AVX/SSE instructions).
    std::memcpy(pDst, pSrc, row_size_bytes);

    // ---------------------------------------------------------
    // 2. CENTRAL PROCESSING (Sliding Window Strategy)
    // ---------------------------------------------------------
    // Initialize sliding pointers to the first three rows (0, 1, 2)
    const float* row_top = pSrc;                // Row Y-1
    const float* row_mid = pSrc + stride;       // Row Y
    const float* row_bot = pSrc + stride * 2;   // Row Y+1

    float* row_dst = pDst + stride;             // Destination Row Y

    const ptrdiff_t lastLine = static_cast<ptrdiff_t>(sizeY - 1);
    const ptrdiff_t lastPixl = static_cast<ptrdiff_t>(sizeX - 1);

    for (ptrdiff_t y = 1; y < lastLine; ++y)
    {
        // --- [LEFT BORDER] ---
        // Copy the leftmost pixel directly from the source.
        // This is extremely fast because 'row_mid' is already hot in the L1 Cache.
        row_dst[0] = row_mid[0];

        // --- [MAIN LOOP: SOBEL FILTER] ---
        // Iterate horizontally from x=1 to x=Width-2.
        // This loop has NO branching (if statements), allowing the compiler 
        // to generate clean AVX/SIMD vectorized code.
        for (ptrdiff_t x = 1; x < lastPixl; ++x)
        {
            // Top Row
            float tl = row_top[x - 1];
            float t  = row_top[x    ];
            float tr = row_top[x + 1];

            // Middle Row (we only need left and right neighbors)
            float l = row_mid[x - 1];
            float r = row_mid[x + 1];

            // Bottom Row
            float bl = row_bot[x - 1];
            float b  = row_bot[x    ];
            float br = row_bot[x + 1];

            // Apply Sobel Kernels
            // Gx Kernel: [-1 0 1], [-2 0 2], [-1 0 1]
            float Gx = (tr + 2.f * r + br) - (tl + 2.f * l + bl);

            // Gy Kernel: [-1 -2 -1], [0 0 0], [1 2 1]
            float Gy = (bl + 2.f * b + br) - (tl + 2.f * t + tr);

            // Compute Magnitude
            row_dst[x] = std::sqrt(Gx * Gx + Gy * Gy);
        }

        // --- [RIGHT BORDER] ---
        // Copy the rightmost pixel directly from the source.
        row_dst[lastPixl] = row_mid[lastPixl];

        // Advance all pointers to the next row
        row_top += stride;
        row_mid += stride;
        row_bot += stride;
        row_dst += stride;
    }

    // ---------------------------------------------------------
    // 3. BOTTOM BORDER HANDLING
    // ---------------------------------------------------------
    // Calculate pointers to the very last row in memory
    const float* src_last_row = pSrc + lastLine * stride;
    float*       dst_last_row = pDst + lastLine * stride;

    // Copy the entire last row directly
    std::memcpy(dst_last_row, src_last_row, row_size_bytes);

    return;
}

#endif // __IMAGE_LAB_ART_POINTILISM_ALGORITHM_DEFINITIONS__