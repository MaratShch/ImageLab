#include <immintrin.h>
#include <cmath>
#include "Common.hpp"
#include "AefxDevPatch.hpp"
#include "PaintMemHandler.hpp"

void compute_initial_tensors_fused_AVX2
(
    const float* RESTRICT im,
    float* RESTRICT A,
    float* RESTRICT B,
    float* RESTRICT C,
    const A_long width,
    const A_long height
) noexcept;

void linear_gradient_gray_AVX2
(
    const float* RESTRICT im,
    float* RESTRICT pGx,
    float* RESTRICT pGy,
    const A_long sizeX,
    const A_long sizeY
) noexcept;

void structure_tensors0_AVX2
(
    const float* RESTRICT gx,
    const float* RESTRICT gy,
    const A_long sizeX, 
    const A_long sizeY,
    float* RESTRICT a,
    float* RESTRICT b,
    float* RESTRICT c
) noexcept;

void generate_gaussian_kernel(float* kernel, const A_long radius, const float sigma) noexcept;

void convolution_AVX2
(
    const float* RESTRICT imIn,
    const float* RESTRICT kernel,
    float* RESTRICT imOutX, // Temporary horizontal buffer (tmpBlur)
    float* RESTRICT imOut,  // Final smoothed output
    const A_long sizeX,
    const A_long sizeY,
    const A_long radius
) noexcept;

void smooth_structure_tensors_AVX2
(
    const float* RESTRICT A,
    const float* RESTRICT B,
    const float* RESTRICT C,
    const float sigma, 
    const A_long sizeX,
    const A_long sizeY,
    float* RESTRICT A_reg,
    float* RESTRICT B_reg,
    float* RESTRICT C_reg,
    float* RESTRICT tmpBlur // Passed from MemHandler
) noexcept;

void diagonalize_structure_tensors_AVX2
(
    const float* RESTRICT A,
    const float* RESTRICT B,
    const float* RESTRICT C,
    const A_long sizeX,
    const A_long sizeY,
    float* RESTRICT Lambda1,
    float* RESTRICT Lambda2,
    float* RESTRICT Eigvect2_x,
    float* RESTRICT Eigvect2_y
) noexcept;

void morpho_open_optimized
(
    const float* RESTRICT imIn,
    float* RESTRICT imOut,
    const float* RESTRICT pLogW,
    const A_long* RESTRICT I,
    const A_long* RESTRICT J,
    const A_long iter,
    const A_long nonZeros,
    const A_long width,
    const A_long height,
    const MemHandler& memHndl
) noexcept;

void morpho_close_optimized
(
    const float* RESTRICT imIn,
    float* RESTRICT imOut,
    const float* RESTRICT pLogW,
    const A_long* RESTRICT I,
    const A_long* RESTRICT J,
    const A_long iter,
    const A_long nonZeros,
    const A_long width,
    const A_long height,
    const MemHandler& memHndl
) noexcept;

void morpho_asf_optimized
(
    const float* RESTRICT imIn,
    float* RESTRICT imOut,
    const float* RESTRICT pLogW,
    const A_long* RESTRICT I,
    const A_long* RESTRICT J,
    const A_long iter,
    const A_long nonZeros,
    const A_long width,
    const A_long height,
    const MemHandler& memHndl
) noexcept;



