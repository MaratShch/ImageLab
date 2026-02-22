#pragma once

#include <cstdint>
#include <immintrin.h>
#include "Common.hpp"


// Invert a symmetric real matrix in-place: V -> Inv(V)
bool AVX2_Inverse_Matrix(double* RESTRICT io_mat, const int32_t p_N) noexcept;

// Transpose a real square matrix in-place: V -> V^T
void AVX2_Transpose_Matrix(double* RESTRICT io_mat, const int32_t p_N) noexcept;

// Compute the empirical covariance matrix from patches (float input) using AVX2
void AVX2_Covariance_Matrix
(
    const float* RESTRICT i_patches, 
    double* RESTRICT o_covMat, 
    const int32_t p_nb, 
    const int32_t p_N
) noexcept;

// Compute the empirical covariance matrix from patches (double input) using AVX2
void AVX2_Covariance_Matrix
(
    const double* RESTRICT i_patches, 
    double* RESTRICT o_covMat, 
    const int32_t p_nb, 
    const int32_t p_N
) noexcept;

// Multiply two matrices: o_mat = i_A * i_B using AVX2
void AVX2_Product_Matrix
(
    double* RESTRICT o_mat, 
    const double* RESTRICT i_A, 
    const double* RESTRICT i_B, 
    const int32_t p_n, 
    const int32_t p_m, 
    const int32_t p_l
) noexcept;