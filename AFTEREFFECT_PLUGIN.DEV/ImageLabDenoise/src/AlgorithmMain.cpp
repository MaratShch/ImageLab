#include <cstdint>
#include <iostream>
#include <random>
#include <algorithm>
#include "AlgoMemHandler.hpp"
#include "AlgorithmMain.hpp"

// NC Algorithm headers
#include "AlgoPyramidBuilder.hpp"
#include "AlgoNoiseOracle.hpp"
#include "AlgoBayesFilter.hpp"


// Safely replicates the last column and row into the padding zone
inline void Pad_Edges_YUV
(
    float* RESTRICT plane,
    int32_t sizeX,
    int32_t sizeY,
    int32_t padW,
    int32_t padH
) noexcept
{
    // Pad Right Edge
    for (int32_t y = 0; y < sizeY; ++y)
    {
        const float last_val = plane[y * padW + (sizeX - 1)];
        for (int32_t x = sizeX; x < padW; ++x)
        {
            plane[y * padW + x] = last_val;
        }
    }
    // Pad Bottom Edge
    for (int32_t y = sizeY; y < padH; ++y)
    {
        for (int32_t x = 0; x < padW; ++x)
        {
            plane[y * padW + x] = plane[(sizeY - 1) * padW + x];
        }
    }
    return;
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
        // 0. DUPLICATE EDGES INTO PADDING
        Pad_Edges_YUV(mem.Y_planar, sizeX, sizeY, mem.padW, mem.padH);
        Pad_Edges_YUV(mem.U_planar, sizeX, sizeY, mem.padW, mem.padH);
        Pad_Edges_YUV(mem.V_planar, sizeX, sizeY, mem.padW, mem.padH);
        
        // 1. CONSTRUCT PYRAMID (Using padW / padH)
        Build_Laplacian_Pyramid (mem, mem.padW, mem.padH);
        
        // 2. BLIND NOISE ORACLE (Using padW / padH)
        Estimate_Noise_Covariances (mem, mem.padW, mem.padH, algoCtrl);

        // 3. MULTI-SCALE DENOISING
        const int32_t qW = mem.padW / 4, qH = mem.padH / 4;
        const int32_t hW = mem.padW / 2, hH = mem.padH / 2;
        const int32_t fW = mem.padW,     fH = mem.padH;
        
        // --- LEVEL 2: QUARTER RESOLUTION ---
        // 2x2 averaging twice reduces noise variance by 16 (4^2)
        Process_Scale_NL_Bayes (mem, mem.Y_quart, mem.U_quart, mem.V_quart, qW, qH, 0.0625f);

        // Reconstruct Half Resolution
        // The denoised Quarter result resides in mem.Accum_*
        Reconstruct_Laplacian_Level (mem.Accum_Y, mem.Y_diff_half, mem.Y_half, hW, hH);
        Reconstruct_Laplacian_Level (mem.Accum_U, mem.U_diff_half, mem.U_half, hW, hH);
        Reconstruct_Laplacian_Level (mem.Accum_V, mem.V_diff_half, mem.V_half, hW, hH);

        // --- LEVEL 1: HALF RESOLUTION ---
        // 2x2 averaging once reduces noise variance by 4
        Process_Scale_NL_Bayes (mem, mem.Y_half, mem.U_half, mem.V_half, hW, hH, 0.25f);

        // Reconstruct Full Resolution
        Reconstruct_Laplacian_Level (mem.Accum_Y, mem.Y_diff_full, mem.Y_planar, fW, fH);
        Reconstruct_Laplacian_Level (mem.Accum_U, mem.U_diff_full, mem.U_planar, fW, fH);
        Reconstruct_Laplacian_Level (mem.Accum_V, mem.V_diff_full, mem.V_planar, fW, fH);

        Process_Scale_NL_Bayes (mem, mem.Y_planar, mem.U_planar, mem.V_planar, fW, fH, 1.0f);
        
        // NOTE: The final, pristine denoised image is now located in:
        // mem.Accum_Y, mem.Accum_U, mem.Accum_V
    }
   
    return;
}