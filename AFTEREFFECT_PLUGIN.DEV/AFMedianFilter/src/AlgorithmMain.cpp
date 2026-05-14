#include "AlgorithmMain.hpp"


// ============================================================================
// ProcessSingleChannel
//
// Runs the AFMF padding + ping-pong iteration loop on one image plane.
// Used by Algorithm_Main for both luma-only and RGB modes:
//   - luma mode: called once with (proc_Y, scratch, out_Y).
//   - RGB mode:  called three times with (proc_<R/G/B>, scratch, out_<R/G/B>).
//
// The scratch buffer is shared across channels in RGB mode -- channels are
// processed sequentially, so each call freely overwrites whatever the
// previous channel left in scratch.
// ============================================================================
static inline void ProcessSingleChannel
(
    float*               sourcePlane,
    float*               scratchPlane,
    float*               outputPlane,
    const int32_t        sizeX,
    const int32_t        sizeY,
    const int32_t        strideElements,
    const AlgoControls&  ctrl
)
{
    // Ping-pong setup: pre-pick the first target so the FINAL iteration always
    // ends up writing to outputPlane regardless of the iteration count parity.
    float* currentSource = sourcePlane;
    float* currentTarget = (ctrl.iterations % 2 != 0) ? outputPlane : scratchPlane;

    for (int32_t i = 0; i < ctrl.iterations; ++i)
    {
        // Pad the current source. Even when the source is the previous pass's
        // output (or scratch), they all carry the invisible halo allocated
        // by the arena, so this is always safe.
        ApplyMirroredPadding
        (
            currentSource,
            sizeX,
            sizeY,
            strideElements,
            ctrl.radius
        );

        // Belt-and-suspenders: force the absolute final iteration to write
        // straight to outputPlane in case earlier swap arithmetic ever drifts.
        if (i == ctrl.iterations - 1)
        {
            currentTarget = outputPlane;
        }

        ProcessImage_Scalar
        (
            const_cast<const float*>(currentSource),
            currentTarget,
            sizeX,
            sizeY,
            strideElements,
            ctrl
        );

        // Swap pointers for the next pass: this pass's output becomes
        // next pass's input, and we flip to whichever buffer we're not on.
        currentSource = currentTarget;
        currentTarget = (currentSource == scratchPlane) ? outputPlane : scratchPlane;
    }
}


// ============================================================================
// ComputeNoiseMap_Luma
//
// Replaces out_Y with a centred Y-channel residual; zeros chroma so the final
// BGR output renders as a clean grayscale noise map.
//
//   out_Y[x] = clamp((proc_Y[x] - out_Y[x]) + 128, 0, 255)
//   proc_U[x] = proc_V[x] = 0
// ============================================================================
static inline void ComputeNoiseMap_Luma
(
    const MemHandler& memHandler,
    const int32_t     sizeX,
    const int32_t     sizeY
)
{
    constexpr float NEUTRAL_LUMA = 128.0f;
    const     int32_t stride     = memHandler.strideY_Elements;

    for (int32_t y = 0; y < sizeY; ++y)
    {
        const float* RESTRICT origRow     = memHandler.proc_Y + y * stride;
              float* RESTRICT denoisedRow = memHandler.out_Y  + y * stride;
              float* RESTRICT uRow        = memHandler.proc_U + y * stride;
              float* RESTRICT vRow        = memHandler.proc_V + y * stride;

        for (int32_t x = 0; x < sizeX; ++x)
        {
            const float diff   = origRow[x] - denoisedRow[x];
            float       mapped = NEUTRAL_LUMA + diff;

            if (mapped <   0.0f) mapped =   0.0f;
            if (mapped > 255.0f) mapped = 255.0f;

            denoisedRow[x] = mapped;

            // Zero chroma so the visualisation is grayscale.
            uRow[x] = 0.0f;
            vRow[x] = 0.0f;
        }
    }
}


// ============================================================================
// ComputeNoiseMap_RGB
//
// Per-channel centred residual: each output channel independently holds the
// signed correction applied to that channel, biased to mid-grey.
//
//   out_R[x] = clamp((proc_R[x] - out_R[x]) + 128, 0, 255)
//   out_G[x] = clamp((proc_G[x] - out_G[x]) + 128, 0, 255)
//   out_B[x] = clamp((proc_B[x] - out_B[x]) + 128, 0, 255)
//
// The colour of a residual indicates which channel(s) were corrected:
//   - neutral grey       -> no correction on that pixel
//   - reddish/greenish/bluish tint -> correction on R / G / B respectively
//   - white-ish          -> all three channels corrected (typical for SPN)
// ============================================================================
static inline void ComputeNoiseMap_RGB
(
    const MemHandler& memHandler,
    const int32_t     sizeX,
    const int32_t     sizeY
)
{
    constexpr float NEUTRAL = 128.0f;
    const     int32_t stride = memHandler.strideY_Elements;

    for (int32_t y = 0; y < sizeY; ++y)
    {
        // In RGB mode the YUV-named planes carry R/G/B inputs and outputs.
        const float* RESTRICT origR = memHandler.proc_Y + y * stride;
        const float* RESTRICT origG = memHandler.proc_U + y * stride;
        const float* RESTRICT origB = memHandler.proc_V + y * stride;
              float* RESTRICT outR  = memHandler.out_Y  + y * stride;
              float* RESTRICT outG  = memHandler.out_U  + y * stride;
              float* RESTRICT outB  = memHandler.out_V  + y * stride;

        for (int32_t x = 0; x < sizeX; ++x)
        {
            float dR = NEUTRAL + (origR[x] - outR[x]);
            float dG = NEUTRAL + (origG[x] - outG[x]);
            float dB = NEUTRAL + (origB[x] - outB[x]);

            if (dR <   0.0f) dR =   0.0f;
            if (dR > 255.0f) dR = 255.0f;
            if (dG <   0.0f) dG =   0.0f;
            if (dG > 255.0f) dG = 255.0f;
            if (dB <   0.0f) dB =   0.0f;
            if (dB > 255.0f) dB = 255.0f;

            outR[x] = dR;
            outG[x] = dG;
            outB[x] = dB;
        }
    }
}


// ============================================================================
// Algorithm_Main
//
// Dispatches between luma-only and RGB processing based on ctrl.inputType.
//
// Luma mode:
//   - Process Y plane only (existing behaviour).
//   - Optionally compute Y-residual noise map (zero chroma).
//
// RGB mode:
//   - Process R, G, B planes independently using the same per-channel helper.
//   - Each channel decides its own bestRadius / corruption status -- a salt
//     impulse on R alone gets cleaned without touching G or B.
//   - Optionally compute per-channel residual noise map.
//
// MemHandler requirements for RGB mode:
//   - proc_Y / proc_U / proc_V hold R / G / B inputs (allocator side decides
//     to use these as RGB planes when ctrl.inputType == AFMF_INPUT_ALL_RGB).
//   - out_Y  / out_U  / out_V  receive R / G / B outputs. *** out_U and out_V
//     must exist as fields in MemHandler -- this is the only struct change
//     required to enable RGB mode. ***
//   - scratch_Y is shared across all three channel passes (sequential
//     processing makes any single scratch sufficient).
// ============================================================================
void Algorithm_Main
(
    const MemHandler&    memHandler,
    const int32_t        sizeX,
    const int32_t        sizeY,
    const AlgoControls&  algoCtrl
)
{
    const int32_t stride = memHandler.strideY_Elements;

        // ===== RGB MODE =====
        // Three independent passes; scratch_Y is shared safely because the
        // channels are processed strictly in sequence.
        ProcessSingleChannel  // R channel
        (
            memHandler.proc_Y,
            memHandler.scratch_Y,
            memHandler.out_Y,
            sizeX, sizeY, stride, algoCtrl
        );
        ProcessSingleChannel  // G channel
        (
            memHandler.proc_U,
            memHandler.scratch_Y,
            memHandler.out_U,
            sizeX, sizeY, stride, algoCtrl
        );
        ProcessSingleChannel  // B channel
        (
            memHandler.proc_V,
            memHandler.scratch_Y,
            memHandler.out_V,
            sizeX, sizeY, stride, algoCtrl
        );

        // Optional per-channel noise-map post-processing.
        if (algoCtrl.outputType == AFMF_Output::AFMF_OUTPUT_NOISE_MAP)
        {
            ComputeNoiseMap_RGB(memHandler, sizeX, sizeY);
        }

    return;
}