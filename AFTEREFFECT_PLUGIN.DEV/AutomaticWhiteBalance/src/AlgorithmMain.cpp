#include <cstring>
#include <utility>
#include <algorithm>

#include "AlgorithmMain.hpp"
#include "AlgCommonFunctions.hpp"   // GrayEstimate + scalar kernels

#ifndef RESTRICT
#define RESTRICT __restrict
#endif

namespace
{
    // orchestration constants
    constexpr int32_t gMaxIter  = 16;
    constexpr int32_t gStatStep = 1;        // raise (e.g. 4) to subsample the estimate pass (Stage 4.4)
    constexpr float   gConvEps2 = 1.0e-8f;  // illuminant-chromaticity convergence (squared)
}

// =============================================================================
// Pure AWB core. All buffers come from memHandler (planar linear RGB_32f).
// Estimate + correct; optional iterative refinement. Result always lands in
// memHandler.output.
// =============================================================================
void Algorithm_Main
(
    const MemHandler& memHandler,
    const int32_t sizeX,
    const int32_t sizeY,
    const AlgoControls& algoCtrl
) noexcept
{
    if (!mem_handler_valid(memHandler) || sizeX <= 0 || sizeY <= 0)
        return;

    const int64_t total     = static_cast<int64_t>(sizeX) * static_cast<int64_t>(sizeY);
    const int32_t iterCnt   = std::max(1, std::min(gMaxIter, algoCtrl.sliderIterCnt));
    const float   threshold = static_cast<float>(algoCtrl.sliderThreshold) * 0.01f;
    // NOTE: threshold feeds the ORTHONORMAL metric sqrt(C1^2+C2^2)/Y, which has a
    // different scale than the old (|U|+|V|)/Y. Recalibrate the default slider.

    const float* RESTRICT iR = memHandler.input.R;
    const float* RESTRICT iG = memHandler.input.G;
    const float* RESTRICT iB = memHandler.input.B;

    float* RESTRICT oR = memHandler.output.R;
    float* RESTRICT oG = memHandler.output.G;
    float* RESTRICT oB = memHandler.output.B;

    // ---- pass 0: estimate from source, write balanced result to output ------
    GrayEstimate est = collect_gray_estimate(iR, iG, iB, total, threshold, gStatStep);

    float M[9];
    build_correction_matrix_linear(est, algoCtrl, M);
    apply_correction(iR, iG, iB, oR, oG, oB, total, M);

    // single-shot path (recommended default)
    const bool haveScratch = (nullptr != memHandler.scratch.R &&
                              nullptr != memHandler.scratch.G &&
                              nullptr != memHandler.scratch.B);
    if (iterCnt <= 1 || !haveScratch)
        return;

    // ---- optional iterative refinement -------------------------------------
    // Re-estimate on the corrected image; stop once the illuminant chromaticity
    // stops moving. Planes ping-pong between output and scratch; final result is
    // guaranteed to end up in output.
    float px, py;
    estimate_chromaticity(est, px, py);

    RGBPlanes cur = memHandler.output;     // holds the latest corrected image
    RGBPlanes nxt = memHandler.scratch;

    for (int32_t k = 1; k < iterCnt; ++k)
    {
        const GrayEstimate ek =
            collect_gray_estimate(cur.R, cur.G, cur.B, total, threshold, gStatStep);

        float cx, cy;
        estimate_chromaticity(ek, cx, cy);
        const float du = cx - px;
        const float dv = cy - py;
        if ((du * du + dv * dv) < gConvEps2)
            break;                                  // converged; 'cur' holds result
        px = cx; py = cy;

        build_correction_matrix_linear(ek, algoCtrl, M);
        apply_correction(cur.R, cur.G, cur.B, nxt.R, nxt.G, nxt.B, total, M);
        std::swap(cur, nxt);
    }

    if (cur.R != memHandler.output.R)
    {
        const size_t bytes = static_cast<size_t>(total) * sizeof(float);
        std::memcpy(memHandler.output.R, cur.R, bytes);
        std::memcpy(memHandler.output.G, cur.G, bytes);
        std::memcpy(memHandler.output.B, cur.B, bytes);
    }

    return;
}
