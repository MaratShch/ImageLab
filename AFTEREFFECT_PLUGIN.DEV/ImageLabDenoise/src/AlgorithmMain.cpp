#include <cstdint>
#include <iostream>
#include <random>
#include <algorithm>
#include "AlgoMemHandler.hpp"
#include "AlgorithmMain.hpp"

// NC Algorithm headers
#include "AVX2_AlgoPyramidBuilder.hpp"
#include "AVX2_AlgoNoiseOracle.hpp"
#include "AVX2_AlgoBayesFilter.hpp"

// Safely repacks the host buffer to the padded stride, then pads the edges
inline void Pad_Edges_YUV
(
    float* RESTRICT plane,
    int32_t sizeX,
    int32_t sizeY,
    int32_t padW,
    int32_t padH
) noexcept
{
    // 1. REPACK BUFFER: Expand from tight host pitch (sizeX) to internal pitch (padW)
    // Iterate backwards (bottom to top) to prevent overwriting unshifted data.
    if (padW > sizeX)
    {
        for (int32_t y = sizeY - 1; y > 0; --y)
        {
            for (int32_t x = sizeX - 1; x >= 0; --x)
            {
                plane[y * padW + x] = plane[y * sizeX + x];
            }
        }
    }

    // 2. Pad Right Edge
    for (int32_t y = 0; y < sizeY; ++y)
    {
        const float last_val = plane[y * padW + (sizeX - 1)];
        for (int32_t x = sizeX; x < padW; ++x)
        {
            plane[y * padW + x] = last_val;
        }
    }
    
    // 3. Pad Bottom Edge
    for (int32_t y = sizeY; y < padH; ++y)
    {
        for (int32_t x = 0; x < padW; ++x)
        {
            plane[y * padW + x] = plane[(sizeY - 1) * padW + x];
        }
    }
    return;
}

// Safely un-packs the padded stride back to the tight host stride
inline void Unpack_Edges_YUV
(
    float* RESTRICT plane,
    int32_t sizeX,
    int32_t sizeY,
    int32_t padW
) noexcept
{
    // Compress from internal pitch (padW) back to tight host pitch (sizeX)
    // Iterate forwards (top to bottom) to safely overwrite.
    if (padW > sizeX)
    {
        for (int32_t y = 1; y < sizeY; ++y)
        {
            for (int32_t x = 0; x < sizeX; ++x)
            {
                plane[y * sizeX + x] = plane[y * padW + x];
            }
        }
    }
}

// =========================================================
// ALGORITHM MAIN (MULTI-SCALE PYRAMID)
// =========================================================
void Algorithm_Main
(
    const MemHandler& mem,
    const int32_t sizeX,
    const int32_t sizeY,
    const AlgoControls& algoCtrl
)
{
    if (mem_handler_valid(mem))
    {
        // 0. DUPLICATE EDGES INTO PADDING (Now includes stride re-packing)
        Pad_Edges_YUV(mem.Y_planar, sizeX, sizeY, mem.padW, mem.padH);
        Pad_Edges_YUV(mem.U_planar, sizeX, sizeY, mem.padW, mem.padH);
        Pad_Edges_YUV(mem.V_planar, sizeX, sizeY, mem.padW, mem.padH);
        
        // 1. CONSTRUCT PYRAMID (Using padW / padH)
        AVX2_Build_Laplacian_Pyramid (mem, mem.padW, mem.padH);
        
        // 2. BLIND NOISE ORACLE (Using padW / padH)
        AVX2_Estimate_Noise_Covariances (mem, mem.padW, mem.padH, algoCtrl);

        // 3. MULTI-SCALE DENOISING
        const int32_t qW = mem.padW / 4, qH = mem.padH / 4;
        const int32_t hW = mem.padW / 2, hH = mem.padH / 2;
        const int32_t fW = mem.padW,     fH = mem.padH;
        

        // --- LEVEL 2: QUARTER RESOLUTION ---
        AVX2_Process_Scale_NL_Bayes (mem, mem.Y_quart, mem.U_quart, mem.V_quart, qW, qH, 0.0625f);

        AVX2_Reconstruct_Laplacian_Level (mem.Accum_Y, mem.Y_diff_half, mem.Y_half, hW, hH);
        AVX2_Reconstruct_Laplacian_Level (mem.Accum_U, mem.U_diff_half, mem.U_half, hW, hH);
        AVX2_Reconstruct_Laplacian_Level (mem.Accum_V, mem.V_diff_half, mem.V_half, hW, hH);

        // --- LEVEL 1: HALF RESOLUTION ---
        AVX2_Process_Scale_NL_Bayes (mem, mem.Y_half, mem.U_half, mem.V_half, hW, hH, 0.25f);

        AVX2_Reconstruct_Laplacian_Level (mem.Accum_Y, mem.Y_diff_full, mem.Y_planar, fW, fH);
        AVX2_Reconstruct_Laplacian_Level (mem.Accum_U, mem.U_diff_full, mem.U_planar, fW, fH);
        AVX2_Reconstruct_Laplacian_Level (mem.Accum_V, mem.V_diff_full, mem.V_planar, fW, fH);

        // --- LEVEL 0: FULL RESOLUTION ---
        AVX2_Process_Scale_NL_Bayes (mem, mem.Y_planar, mem.U_planar, mem.V_planar, fW, fH, 1.0f);
        
        // 4. RESTORE TIGHT HOST PITCH
        // The host color converter will read from Accum_*, expecting tightly packed data.
        Unpack_Edges_YUV(mem.Accum_Y, sizeX, sizeY, mem.padW);
        Unpack_Edges_YUV(mem.Accum_U, sizeX, sizeY, mem.padW);
        Unpack_Edges_YUV(mem.Accum_V, sizeX, sizeY, mem.padW);

    }
   
    return;
}