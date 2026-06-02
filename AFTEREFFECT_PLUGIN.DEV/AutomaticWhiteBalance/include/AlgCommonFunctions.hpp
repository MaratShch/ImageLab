#ifndef __IMAGE_LAB_AWB_ALGO_COMMON_FUNCTIONS__
#define __IMAGE_LAB_AWB_ALGO_COMMON_FUNCTIONS__

#include <cstdint>
#include "Common.hpp"
#include "AlgoControl.hpp"
#include "CommonAuxPixFormat.hpp"   // dRGB

// -----------------------------------------------------------------------------
// AWB scalar kernels (reference implementation).
//
// Written to be auto-vectorization friendly (contiguous planar access, RESTRICT,
// branchless hot loops). The AVX2 and CUDA variants will later replace these
// definitions behind the SAME declarations.
// -----------------------------------------------------------------------------

struct GrayEstimate
{
    dRGB    sum;     // sum of linear RGB over selected (gray) pixels
    int64_t count;   // number of selected pixels
};

// Detect gray pixels (orthonormal opponent metric) and accumulate their linear RGB.
// 'step' subsamples the scan (1 = every pixel). Planar, contiguous.
GrayEstimate collect_gray_estimate
(
    const float* RESTRICT R,
    const float* RESTRICT G,
    const float* RESTRICT B,
    const int64_t total,
    const float   threshold,
    const int32_t step
) noexcept;

// Build the full 3x3 linear-RGB correction matrix (chromatic adaptation).
// Identity when the estimate is empty/degenerate. Runs once per frame.
void build_correction_matrix_linear
(
    const GrayEstimate& est,
    const AlgoControls& ctrl,
    float M[9]
) noexcept;

// Apply the full 3x3 matrix over planar buffers. Negatives floored at 0;
// no upper clamp (scene-linear / HDR safe).
void apply_correction
(
    const float* RESTRICT sR, const float* RESTRICT sG, const float* RESTRICT sB,
    float*       RESTRICT dR, float*       RESTRICT dG, float*       RESTRICT dB,
    const int64_t total,
    const float   M[9]
) noexcept;

// rg-chromaticity of the estimate (iteration stop criterion). Tiny -> inline.
inline void estimate_chromaticity (const GrayEstimate& g, float& x, float& y) noexcept
{
    const double s = g.sum.R + g.sum.G + g.sum.B;
    if (g.count <= 0 || s <= 0.0) { x = 0.f; y = 0.f; return; }
    x = static_cast<float>(g.sum.R / s);
    y = static_cast<float>(g.sum.G / s);
}

#endif // __IMAGE_LAB_AWB_ALGO_COMMON_FUNCTIONS__
