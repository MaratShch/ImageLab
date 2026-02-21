#pragma once

#include <cstdint>
#include "Common.hpp"

// Invert a symmetric real matrix in-place: V -> Inv(V)
// Returns true on success, false if the matrix is not positive definite
bool Inverse_Matrix(double* RESTRICT io_mat, const int32_t p_N);

// Transpose a real square matrix in-place: V -> V^T
void Transpose_Matrix(double* RESTRICT io_mat, const int32_t p_N);

// Compute the empirical covariance matrix from patches (float input)
void Covariance_Matrix
(
    const float* RESTRICT i_patches, 
    double* RESTRICT o_covMat, 
    const int32_t p_nb, 
    const int32_t p_N
);

// Compute the empirical covariance matrix from patches (double input)
void Covariance_Matrix
(
    const double* RESTRICT i_patches, 
    double* RESTRICT o_covMat, 
    const int32_t p_nb, 
    const int32_t p_N
);

// Multiply two matrices: o_mat = i_A * i_B
void Product_Matrix
(
    double* RESTRICT o_mat, 
    const double* RESTRICT i_A, 
    const double* RESTRICT i_B, 
    const int32_t p_n, 
    const int32_t p_m, 
    const int32_t p_l
);