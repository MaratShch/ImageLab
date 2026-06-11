#ifndef __IMAGE_LAB_PCA_AWB_CORRECTION__
#define __IMAGE_LAB_PCA_AWB_CORRECTION__

#include <cstdint>
#include "AlgCommonEnums.hpp"   // eILLUMINATE, eChromaticAdaptation

// ---------------------------------------------------------------------------
// Illuminant estimate: summed linear RGB over the selected pixels + the count.
// Only the chromaticity (direction) is used downstream, so the PCA path passes
// a unit direction with count = 1; a gray-world fallback could pass a real sum.
// (Self-contained -- intentionally NOT the gray-point project's GrayEstimate.)
// ---------------------------------------------------------------------------
struct AwbEstimate
{
    double  sumR;
    double  sumG;
    double  sumB;
    int64_t count;
};

// Build the 3x3 linear-RGB chromatic-adaptation matrix that maps the estimated
// illuminant onto the target white (illuminate) using the chosen CAT model
// (chromatic). Writes identity when the estimate is degenerate (count <= 0).
// Reads only these two enums -- no AlgoControls dependency.
void build_correction_matrix_linear
(
    const AwbEstimate&         est,
    const eILLUMINATE          illuminate,
    const eChromaticAdaptation chromatic,
    float                      M[9]
) noexcept;

// Apply the 3x3 matrix over planar linear-RGB buffers (scalar). Negatives are
// floored at 0; no upper clamp (scene-linear / HDR safe). In-place is fine.
void apply_correction
(
    const float* sR, const float* sG, const float* sB,
    float*       dR, float*       dG, float*       dB,
    const int64_t total,
    const float   M[9]
) noexcept;

#endif // __IMAGE_LAB_PCA_AWB_CORRECTION__
