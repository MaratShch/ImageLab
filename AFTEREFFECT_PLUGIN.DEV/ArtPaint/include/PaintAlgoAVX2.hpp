#include <immintrin.h>
#include <cmath>
#include "Common.hpp"
#include "AefxDevPatch.hpp"
#include "PaintMemHandler.hpp"

// ==================================================================================
// --- AVX2 TENSOR PIPELINE ---
// ==================================================================================

void compute_initial_tensors_fused_AVX2
(
    const float* RESTRICT im,
    float* RESTRICT A,
    float* RESTRICT B,
    float* RESTRICT C,
    const A_long width,
    const A_long height
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
    float* RESTRICT tmpBlur
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


void morpho_open
(
    float* RESTRICT imInOut, 
    const A_long* RESTRICT I,
    const A_long* RESTRICT J,
    const A_long iter,
    const A_long nonZeros,
    const A_long width,
    const A_long height,
    const MemHandler& memHndl
) noexcept;

void morpho_close
(
    float* RESTRICT imInOut,
    const A_long* RESTRICT I,
    const A_long* RESTRICT J,
    const A_long iter,
    const A_long nonZeros,
    const A_long width,
    const A_long height,
    const MemHandler& memHndl
) noexcept;

void morpho_asf
(
    float* RESTRICT imInOut,
    const A_long* RESTRICT I,
    const A_long* RESTRICT J,
    const A_long iter,
    const A_long nonZeros,
    const A_long width,
    const A_long height,
    const MemHandler& memHndl
) noexcept;