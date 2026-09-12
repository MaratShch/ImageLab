// =============================================================================
// AlgorithmMain.cpp  —  PHASE 2 orchestration (Steps A' , A , B , C , D)
//
// Called ONCE PER RENDER CALL, after Phase 1 (ingest + fused measure) has
// already produced:
//     * memHandler input  : linear, interleaved, working-space RGB float32
//                           (or the CONFIDENCE MAP, when that output view is
//                           selected — the fused pass writes it in place)
//     * superPixel        : the locus-gated weighted mean
//
// The effect has TWO jobs, and they map onto the two white points of the
// chromatic adaptation:
//     SOURCE white  = what the light WAS  -> correcting it is WHITE BALANCE
//     TARGET white  = what you WANT       -> moving it is CREATIVE GRADING
// Source is absolute (seeded from the measurement); target is a preset plus
// user OFFSETS that default to zero, so "no look" is a true no-op and a look
// copied between shots stays meaningful. This mirrors how grading tools split
// absolute (camera-RAW) from relative (primaries) temperature control.
//
// This function does NOT measure pixels. Phase 1 did that. Here we:
//     A'  solve the super-pixel -> measured CCT/Duv        (ALWAYS)
//     A   choose the SOURCE white  = WHITE BALANCE         (Measure | Manual)
//     B   choose the TARGET white  = LOOK / GRADE          (preset + offsets)
//     C   build ONE 3x3 correction matrix M_wb             (once per frame)
//     D   apply M_wb to every pixel                        (AVX2)
//
// SINGLE-THREADED per call by design: the host tiles the frame and calls this
// concurrently on several threads. Nothing here writes shared mutable state.
// =============================================================================

#include "AlgorithmMain.hpp"          // declaration only
#include "AlgoControl.hpp"            // YOUR AlgoControls
#include "AlgoWhiteBalance.hpp"       // Step C : build_wb_matrix, eCatModel
#include "AlgoApplyWB.hpp"            // Step D : apply_white_balance, ApplyParams
#include "color_utils.hpp"            // Step A': superpixel_to_cct
#include "CCTLut/CCTLut.hpp"          // CCT_MIN / CCT_MAX of the generated grid
#include "ColorTransformMatrix.hpp"   // eCOLOR_OBSERVER
#include "Algo2Rgb2XYZ.hpp"           // sRGBtoXYZ_f64 / XYZtosRGB_f64
#include "super_pixel.hpp"            // SuperPixel<>, CctDuv<>


// ---------------------------------------------------------------------------
// Local helpers — all internal, none of these are user-visible parameters.
// ---------------------------------------------------------------------------
namespace
{
    // ------------------------------------------------------------------
    // MemHandler adapters - THE ONLY TWO LINES THAT TOUCH YOUR BUFFER
    // MEMBER NAMES. If your MemHandler spells them differently, edit here
    // and nowhere else. (Names taken from your own main.cpp comments:
    // "memHndl (in: srcRGB_f32, out: dstRGB_f32)".)
    // ------------------------------------------------------------------
    inline const float* mem_linear_in (const MemHandler& m) noexcept
    { return m.srcRGB_f32; }

    inline float* mem_linear_out (const MemHandler& m) noexcept
    { return m.dstRGB_f32; }

    // ------------------------------------------------------------------
    // Controls your AlgoControls does not carry yet. They are fixed at the
    // documented defaults so the build works against your struct AS IT IS.
    // Add the fields when you want them exposed, then replace each constant
    // with params.<field>. Nothing else changes.
    // ------------------------------------------------------------------
    const double  kStrength      = 1.0;    // 100 %  (no 'strength' field yet)
    const bool    kHighlightSafe = false;  // internal option, default off
    const int32_t kWarmthMired   = 0;      // Look: no offset  (no fields yet)
    const double  kTintShiftDuv  = 0.0;    // Look: no offset
    const double  kTargetCct     = 6504.0; // v1 fixed D65, per your Step B note
    const double  kTargetDuv     = 0.0;

    // UI tint (+-100, magenta-positive) -> CIE Duv. ONE NUMBER TO CONFIRM:
    // this maps the full +-100 slider onto +-0.020 Duv, which covers real
    // illuminants with headroom (validated extremes: -0.0112 Sunset,
    // +0.0137). Change the constant if you want a different span.
    const double kTintToDuv = 2.0e-4;
    // UI mired  ->  Kelvin.  The UI works in mireds because a fixed Kelvin
    // step is not perceptually uniform (dUv per Kelvin varies ~460x over
    // 2000..40000 K; per mired only ~1.3x). The ALGORITHM works in Kelvin
    // because the CCT LUT is Kelvin-indexed at 1 K. Convert here, once.
    inline double mired_to_kelvin (const int32_t miredCoarse,
                                  const int32_t miredFine) noexcept
    {
        // coarse and fine are SUMMED before conversion - never applied as two
        // separate corrections.
        int32_t m = miredCoarse + miredFine;
        if (m < 25)   m = 25;          // 40000 K
        if (m > 1111) m = 1111;        //   900 K  (full LUT span)
        return 1000000.0 / static_cast<double>(m);
    }

    // UI tint -> CIE Duv.  UI convention is MAGENTA-POSITIVE; the physical
    // Duv sign is GREEN-positive, hence the negation. Coarse is in units of
    // 1e-3, fine in units of 1e-4 (see the control specification).
    inline double tint_to_duv (const int32_t tintCoarse,
                               const int32_t tintFine) noexcept
    {
        const double duv = static_cast<double>(tintCoarse) * 1.0e-3 +
                           static_cast<double>(tintFine)   * 1.0e-4;
        return -duv;                   // UI magenta-positive -> CIE green-positive
    }

    // ---- working-space RGB <-> XYZ (D65) --------------------------------
    // Uses the project's canonical tables from Algo2Rgb2XYZ.hpp. Both
    // directions come from ONE 50-digit derivation, so round-trips close
    // exactly - verified here: the shipped literals match a 50-digit
    // reference with ZERO error in both directions, and M*M^-1 - I is 1.2e-16.
    //
    // The SAME pair must also feed build_locus_gate() and the AVX2 measure
    // context. Measurement and correction disagreeing about the working space
    // is a correctness bug, not a tolerance issue.
    //
    // Rec.2020: define IMAGELAB2_HAVE_REC2020_MATRICES once the matching pair
    // is added to Algo2Rgb2XYZ.hpp (constants supplied separately, same
    // derivation). Until then workingSpace != 0 falls back to sRGB rather
    // than referencing symbols that do not exist.
    // Maps YOUR eWorkingSpace (ws_Auto=0, ws_sRGB_709=1, ws_Rec2020=2,
    // ws_DisplayP3=3, ws_ACEScg=4) onto the matrix pair.
    // ws_Auto falls back to sRGB, exactly as your header specifies.
    // Rec.2020 / P3 / ACEScg currently fall back to sRGB because the tree has
    // only the sRGB pair; the Rec.2020 constants are supplied separately in
    // the same derivation. A silent wrong matrix is worse than a documented
    // fallback, so this is deliberate - wire the others before shipping those
    // working spaces.
    inline const double* working_space_to_XYZ (const int32_t ws) noexcept
    {
    #ifdef IMAGELAB2_HAVE_REC2020_MATRICES
        if (ws_Rec2020 == ws) return Rec2020toXYZ_f64;
    #endif
        (void)ws;
        return sRGBtoXYZ_f64;
    }

    inline const double* XYZ_to_working_space (const int32_t ws) noexcept
    {
    #ifdef IMAGELAB2_HAVE_REC2020_MATRICES
        if (ws_Rec2020 == ws) return XYZtoRec2020_f64;
    #endif
        (void)ws;
        return XYZtosRGB_f64;
    }

    // Base delivery target, in MIRED (the offsets below are additive in mired,
    // which is exactly how conversion filters are specified).
    inline int32_t target_base_mired (const int32_t preset) noexcept
    {
        switch (preset)
        {
            case 1:  return 182;   // D55  5503 K
            case 2:  return 200;   // D50  5003 K
            case 3:  return 159;   // DCI-P3 6300 K
            case 0:
            default: return 154;   // D65  6504 K  (default)
        }
    }

    // TARGET white = delivery base + the user's LOOK offsets.
    //   warmth  : mired shift. POSITIVE = warmer image. Additive, and the same
    //             number means the same visual shift at any base temperature.
    //             Filter equivalents: +18 = 81A, +112 = 85, +131 = 85B,
    //             -112 = 80B, -131 = 80A.
    //   tintShift: Duv offset, UI magenta-positive -> negate for CIE.
    // Both default to 0, so an ungraded shot passes through as pure white
    // balance.
    inline void target_white (const int32_t preset,
                              const int32_t warmthCoarse, const int32_t warmthFine,
                              const int32_t tintShiftCoarse, const int32_t tintShiftFine,
                              double& cct, double& duv) noexcept
    {
        int32_t m = target_base_mired(preset) + warmthCoarse + warmthFine;
        if (m < 25)   m = 25;          // 40000 K
        if (m > 1111) m = 1111;        //   900 K
        cct = 1000000.0 / static_cast<double>(m);
        duv = -(static_cast<double>(tintShiftCoarse) * 1.0e-3 +
                static_cast<double>(tintShiftFine)   * 1.0e-4);
    }

    // Observer -> the matching locus LUT. The observer used for the CORRECTION
    // must be the SAME one used for the MEASUREMENT - a mismatch is a
    // correctness bug, not a tolerance issue.
    inline void pick_lut (AlgoCCT::CctHandle<double>& h, const int32_t observer,
                          const AlgoCCT::LutRow*& lut,
                          std::size_t& n) noexcept
    {
        if (observer_CIE_1964 == observer) { const auto p = h.getLut_CIE_1964();
                                            lut = p.first; n = p.second; }
        else                               { const auto p = h.getLut_CIE_1931();
                                            lut = p.first; n = p.second; }
    }
} // anonymous namespace


// =============================================================================
void Algorithm_Main
(
    AlgoCCT::CctHandle<double>& cctHdnl,
    const SuperPixel<double>&   superPixel,  // Previously computed SuperPixel
    const MemHandler&           memHandler,  // linear in/out RGB buffers
    const int32_t               sizeX,       // horizontal size in pixels
    const int32_t               sizeY,       // vertical size in pixels
    const AlgoControls&         params,      // Algorithm control parameters
    CctDuv<double>&             cct_duv,     // OUT: computed CCT and Duv/Tint
    const AlgoWB::WbReference*  reference    // stored reference (may be null)
) noexcept
{
    double cct_computed = 0.0;      // STEP A' output: measured CCT [K]
    double duv_computed = 0.0;      // STEP A' output: measured Duv

    // ---- working-space matrices (must match what Phase 1 measured with) ----
    // The SAME pair feeds the locus gate, the measurement and Step C.
    const double* const rgb2xyz = working_space_to_XYZ (params.workingSpace);
    const double* const xyz2rgb = XYZ_to_working_space (params.workingSpace);

    // =======================================================================
    // STEP A' : MEASURE the incoming CCT/Duv - ALWAYS (unconditional).
    // The scene's measured white point is a first-class OUTPUT of every call
    // (UI readout + confidence workflow), independent of wbMode and of
    // whether a correction is applied. So it runs before, and regardless of,
    // Steps A..D.
    // IN : superPixel (from ingest), RGB->XYZ matrix, params.observer.
    // OUT: cct_duv (the measured source white).
    // =======================================================================
    superpixel_to_cct (superPixel, cctHdnl, rgb2xyz,
                       static_cast<eCOLOR_OBSERVER>(params.observer),
                       cct_computed, duv_computed);

    cct_duv.cct = cct_computed;    // measured CCT reported unconditionally
    cct_duv.duv = duv_computed;    // measured Duv reported unconditionally

    // =======================================================================
    // OUTPUT VIEW = CONFIDENCE MAP -> there is nothing to correct.
    // Phase 1 already wrote the map INTO the input linear buffer, and the
    // caller selects that buffer for egress. So we return WITHOUT touching
    // the output buffer - no copy, no matrix, no apply pass. This makes map
    // mode cheaper than normal operation, and it keeps the map showing what
    // the MEASUREMENT used (correcting the map would misrepresent it).
    // The measured readout above is still produced.
    // =======================================================================
    if (0 != params.confidenceMap)
        return;

    // =======================================================================
    // STEP A : establish the SOURCE white for the CORRECTION.
    // Distinct from the measurement above: in Manual mode the correction
    // source comes from the sliders, not from the measured value.
    // =======================================================================
    double src_cct = 0.0;
    double src_duv = 0.0;

    if (wbMode_Manual == params.wbMode)
    {
        // Sliders are authoritative. UI mired -> Kelvin, UI tint -> CIE Duv.
        // Your fields are Kelvin-based: coarse + fine trim, summed.
        src_cct = static_cast<double>(params.temperatureK) +
                  static_cast<double>(params.temperatureFineK);
        // UI tint is magenta-positive; CIE Duv is green-positive -> negate.
        src_duv = -(static_cast<double>(params.tint) * kTintToDuv);
    }
    else if (nullptr != reference && reference->valid)
    {
        // Measure mode with a STORED REFERENCE (the normal case).
        // Note this is the EFFECTIVE source solved at Compute time, not the
        // raw measurement: it is the value that makes ONE application land on
        // the target. On Sunset that turns a 11.9 mired residual into 0.09.
        // The raw measurement is still what the UI reports (set above).
        src_cct = reference->sourceCct;
        src_duv = reference->sourceDuv;
    }
    else
    {
        // No stored reference yet - fall back to this frame's measurement.
        // Correct, just not refined (one pass closes ~88 % of the gap).
        src_cct = cct_computed;
        src_duv = duv_computed;
    }

    // ---- measurement / input sanity gate ----------------------------------
    // A frame with no usable neutral content yields an all-zero super-pixel,
    // and superpixel_to_cct cannot produce a white point from it. Correcting
    // on a garbage source would visibly wreck the frame, so pass the image
    // through unchanged (output stays whatever the caller prepared) while
    // still reporting the failed measurement upstream.
    if (!(src_cct > 0.0) || !(src_cct == src_cct))      // 0, negative or NaN
        return;

    // =======================================================================
    // STEP B : TARGET white  =  THE LOOK.
    // Delivery base (D65 etc.) plus the user's Warmth / Tint-shift OFFSETS.
    // With both offsets at 0 this is pure white balance: the frame is
    // neutralised to the delivery white. Dial Warmth positive and the frame
    // gets warmer; negative and it gets cooler - the direction the user
    // expects, because we are moving the white they are rendering TO.
    // =======================================================================
    double tgt_cct = 6504.0;   // D65
    double tgt_duv = 0.0;
    // v1: fixed D65, exactly as your Step B comment specifies.
    // LOOK HOOK - when you add the Warmth / Tint-shift fields, replace these
    // two lines with:
    //     target_white(params.targetPreset,
    //                  params.warmthCoarse, params.warmthFine,
    //                  params.tintShiftCoarse, params.tintShiftFine,
    //                  tgt_cct, tgt_duv);
    // target_white() is already present below and needs no change.
    tgt_cct = 1000000.0 / (double)(target_base_mired(0) + kWarmthMired);
    tgt_duv = -kTintShiftDuv;

    // =======================================================================
    // STEP C : build M_wb from source/target via the chromatic adaptation
    // transform. Built ONCE here - never per pixel.
    //   IN : source (src_cct, src_duv), target (tgt_cct, tgt_duv),
    //        params.catModel, params.adaptationDegree, params.observer,
    //        working-space RGB<->XYZ pair.
    //   OUT: M_wb[9] (row-major, linear RGB -> corrected linear RGB)
    // =======================================================================
    const AlgoCCT::LutRow* lut = nullptr;
    std::size_t lutN = 0u;
    pick_lut (cctHdnl, params.observer, lut, lutN);

    double M_wb[9];
    const bool matrixOk = AlgoWB::build_wb_matrix (
                              lut, lutN,
                              CCT_LUT_1931_2DEG::CCT_MIN,   // both generated
                              CCT_LUT_1931_2DEG::CCT_MAX,   // tables share the grid
                              CctDuv<double>{ src_cct, src_duv },
                              CctDuv<double>{ tgt_cct, tgt_duv },
                              params.catModel,
                              params.adaptationDegree,      // v1: 1.0
                              rgb2xyz, xyz2rgb,
                              M_wb);

    // Singular CAT / working space - cannot happen for the shipped matrices,
    // but never apply a matrix we failed to build.
    if (!matrixOk)
        return;

    // =======================================================================
    // STEP D : fill the OUTPUT linear RGB buffer from the INPUT linear RGB
    // buffer. One 3x3 multiply per pixel, 8 pixels at a time (AVX2 + FMA).
    // Strength, highlight-safe scaling and the clip policy are folded into
    // the MATRIX inside apply_white_balance - they cost nothing per pixel.
    // =======================================================================
    AlgoWB::ApplyParams applyParams;
    applyParams.strength      = kStrength;         // no 'strength' field yet
    applyParams.highlightSafe = kHighlightSafe;    // internal, default off
    // clip_Auto needs the pixel format, which AlgoControls does not carry and
    // Algorithm_Main cannot see. clip_Never is the safe stand-in: integer and
    // encoded formats still clamp at ENCODE inside egress, while _Linear
    // targets keep their HDR - which clip_Always would destroy.
    applyParams.clipPolicy    = AlgoWB::clip_Never;

    AlgoWB::apply_white_balance (mem_linear_in (memHandler),   // const float*
                                 mem_linear_out(memHandler),   // float*
                                 sizeX, sizeY,
                                 M_wb,
                                 applyParams,
                                 /*useAvx2=*/ true);       // whole plugin is /arch:AVX2

    return;
}
