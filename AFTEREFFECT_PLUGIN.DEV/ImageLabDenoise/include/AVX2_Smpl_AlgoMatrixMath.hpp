#pragma once
#include <cstdint>
#include "Common.hpp"

bool AVX2_Smpl_Inverse_Matrix (float* RESTRICT io_mat, const int32_t p_N);

void AVX2_Smpl_Transpose_Matrix (float* RESTRICT io_mat, const int32_t p_N);

void AVX2_Smpl_Product_Matrix
(
    float* RESTRICT o_mat, 
    const float* RESTRICT i_A, 
    const float* RESTRICT i_B, 
    const int32_t p_n, 
    const int32_t p_m, 
    const int32_t p_l
);