#include <cstdint>
#include <iostream>
#include <algorithm>
#include "AlgoMemHandler.hpp"
#include "AlgorithmMain.hpp"

// NC Algorithm headers
#include "AVX2_AlgoPyramidBuilder.hpp"
#include "AVX2_AlgoNoiseOracle.hpp"
#include "AVX2_Smpl_AlgoBayesFilter.hpp"

inline void Pad_Edges_YUV
(
    float* RESTRICT plane,
    int32_t sizeX,
    int32_t sizeY,
    int32_t padW,
    int32_t padH
) noexcept
{
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

    for (int32_t y = 0; y < sizeY; ++y)
    {
        const float last_val = plane[y * padW + (sizeX - 1)];
        for (int32_t x = sizeX; x < padW; ++x)
        {
            plane[y * padW + x] = last_val;
        }
    }
    
    for (int32_t y = sizeY; y < padH; ++y)
    {
        for (int32_t x = 0; x < padW; ++x)
        {
            plane[y * padW + x] = plane[(sizeY - 1) * padW + x];
        }
    }
}

inline void Unpack_Edges_YUV
(
    float* RESTRICT plane,
    int32_t sizeX,
    int32_t sizeY,
    int32_t padW
) noexcept
{
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
        // 0. DUPLICATE EDGES INTO PADDING
        Pad_Edges_YUV(mem.Y_planar, sizeX, sizeY, mem.padW, mem.padH);
        Pad_Edges_YUV(mem.U_planar, sizeX, sizeY, mem.padW, mem.padH);
        Pad_Edges_YUV(mem.V_planar, sizeX, sizeY, mem.padW, mem.padH);
        
        // 1. CONSTRUCT PYRAMID
        AVX2_Build_Laplacian_Pyramid (mem, mem.padW, mem.padH);
        
        // 2. BLIND NOISE ORACLE
        AVX2_Estimate_Noise_Covariances (mem, mem.padW, mem.padH, algoCtrl);

        // 3. MULTI-SCALE DENOISING
        const int32_t qW = mem.padW / 4, qH = mem.padH / 4;
        const int32_t hW = mem.padW / 2, hH = mem.padH / 2;
        const int32_t fW = mem.padW,     fH = mem.padH;
        
        // --- LEVEL 2: QUARTER RESOLUTION (Coarse Noise) ---
        float mult_Q = algoCtrl.master_denoise_amount * algoCtrl.coarse_noise_reduction * 0.0625f;
        AVX2_Smpl_Process_Scale_NL_Bayes (mem, mem.Y_quart, mem.U_quart, mem.V_quart, qW, qH, mult_Q, algoCtrl);

        AVX2_Reconstruct_Laplacian_Level (mem.Accum_Y, mem.Y_diff_half, mem.Y_half, hW, hH);
        AVX2_Reconstruct_Laplacian_Level (mem.Accum_U, mem.U_diff_half, mem.U_half, hW, hH);
        AVX2_Reconstruct_Laplacian_Level (mem.Accum_V, mem.V_diff_half, mem.V_half, hW, hH);

        // --- LEVEL 1: HALF RESOLUTION (Standard Noise) ---
        float mult_H = algoCtrl.master_denoise_amount * 0.25f;
        AVX2_Smpl_Process_Scale_NL_Bayes (mem, mem.Y_half, mem.U_half, mem.V_half, hW, hH, mult_H, algoCtrl);

        AVX2_Reconstruct_Laplacian_Level (mem.Accum_Y, mem.Y_diff_full, mem.Y_planar, fW, fH);
        AVX2_Reconstruct_Laplacian_Level (mem.Accum_U, mem.U_diff_full, mem.U_planar, fW, fH);
        AVX2_Reconstruct_Laplacian_Level (mem.Accum_V, mem.V_diff_full, mem.V_planar, fW, fH);

        // --- LEVEL 0: FULL RESOLUTION (Fine Detail) ---
        float mult_F = algoCtrl.master_denoise_amount * algoCtrl.fine_detail_preservation * 1.0f;
        AVX2_Smpl_Process_Scale_NL_Bayes (mem, mem.Y_planar, mem.U_planar, mem.V_planar, fW, fH, mult_F, algoCtrl);
        
        // 4. RESTORE TIGHT HOST PITCH
        Unpack_Edges_YUV(mem.Accum_Y, sizeX, sizeY, mem.padW);
        Unpack_Edges_YUV(mem.Accum_U, sizeX, sizeY, mem.padW);
        Unpack_Edges_YUV(mem.Accum_V, sizeX, sizeY, mem.padW);
    }
}