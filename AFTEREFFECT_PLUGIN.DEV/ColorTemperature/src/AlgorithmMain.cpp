// =============================================================================
// AlgorithmMain.cpp  -  PHASE 2 orchestration (Steps A', A, B, C, D)
//                       + the SETUP and COMPUTE entry points the host calls.
//
// The effect has TWO jobs, mapped onto the two white points of the chromatic
// adaptation:
//     SOURCE white = what the light WAS   -> correcting it is WHITE BALANCE
//     TARGET white = what you WANT        -> moving it is GRADING (the Look)
// Source is absolute (mireds, seeded from the measurement); target is a
// delivery preset plus Look OFFSETS that default to zero.
//
// Three entry points, by when they run:
//     Algorithm_BuildMeasureSetup()  once, and again whenever observer,
//                                    working space or gate band change
//     Algorithm_ComputeReference()   when the user presses Compute
//     Algorithm_Main()               every render call
//
// Every AlgoControls field is read here (or is documented as consumed by the
// setup / compute entry point). Controls are validated ONCE per call on a
// local copy with normalize_algo_controls(), so invalid host data never
// reaches the maths.
//
// SINGLE-THREADED per call by design: the host tiles the frame and calls this
// concurrently. Nothing here writes shared mutable state.
// =============================================================================

#include "AlgorithmMain.hpp"          // declarations
#include "AlgoControl.hpp"            // AlgoControls, AlgoCtl:: limits
#include "AlgoWhiteBalance.hpp"       // Step C : build_wb_matrix, eCatModel
#include "AlgoApplyWB.hpp"            // Step D : apply_white_balance
#include "AlgoReference.hpp"          // WbReference, confidence, refinement
#include "color_utils.hpp"            // Step A': superpixel_to_cct
#include "CCTLut/CCTLut.hpp"                 // CCT_MIN / CCT_MAX of the generated grid
#include "ColorTransformMatrix.hpp"   // eCOLOR_OBSERVER
#include "Algo2Rgb2XYZ.hpp"           // sRGBtoXYZ_f64 / XYZtosRGB_f64
#include "super_pixel.hpp"            // SuperPixel<>, CctDuv<>

// The UI enums in AlgoControl.hpp mirror enums owned by the algorithm
// headers. Pin the correspondence so a renumbering cannot silently remap.
static_assert(static_cast<int32_t>(clipCtl_Auto)   == static_cast<int32_t>(AlgoWB::clip_Auto),   "clip enum drift");
static_assert(static_cast<int32_t>(clipCtl_Always) == static_cast<int32_t>(AlgoWB::clip_Always), "clip enum drift");
static_assert(static_cast<int32_t>(clipCtl_Never)  == static_cast<int32_t>(AlgoWB::clip_Never),  "clip enum drift");
static_assert(static_cast<int32_t>(cat_Bradford)   == static_cast<int32_t>(AlgoWB::cat_Bradford), "CAT enum drift");
static_assert(static_cast<int32_t>(cat_CAT16)      == static_cast<int32_t>(AlgoWB::cat_CAT16),    "CAT enum drift");
static_assert(static_cast<int32_t>(cat_CAT02)      == static_cast<int32_t>(AlgoWB::cat_CAT02),    "CAT enum drift");
static_assert(static_cast<int32_t>(cat_VonKries)   == static_cast<int32_t>(AlgoWB::cat_VonKries), "CAT enum drift");
static_assert(static_cast<int32_t>(obs_CIE_1931_2deg)  == static_cast<int32_t>(observer_CIE_1931), "observer enum drift");
static_assert(static_cast<int32_t>(obs_CIE_1964_10deg) == static_cast<int32_t>(observer_CIE_1964), "observer enum drift");


namespace
{
    // ------------------------------------------------------------------
    // MemHandler adapters - the ONLY two lines touching buffer member names.
    // ------------------------------------------------------------------
    inline const float* mem_linear_in (const MemHandler& m) noexcept
    { return m.srcRGB_f32; }

    inline float* mem_linear_out (const MemHandler& m) noexcept
    { return m.dstRGB_f32; }

    // ------------------------------------------------------------------
    // Delivery TARGETS - the real white points, not Planckian approximations.
    //
    // CORRECTED in this revision. The previous table used ROUNDED mireds
    // (154/182/200/159) and Duv = 0 for every entry. But the standard whites
    // are NOT on the Planckian locus: D65/D55/D50 sit at Duv ~ +0.0032 and the
    // DCI-P3 white at +0.0134. sRGB's own white (1,1,1) IS D65, so "Render for
    // D65" with Duv 0 would push a perfectly neutral image 0.0032 Duv toward
    // magenta.
    //
    // Values: CCT/Duv of each standard chromaticity, computed by THIS
    // project's CCT engine (Ohno 2013 on the 1-K CIE-1931 LUT) so that
    // build_wb_matrix() reproduces the chromaticity on round-trip.
    // Independent check (colour-science): agrees to 0.03 K and 1e-6 Duv.
    // PRECISION: the chromaticities are published to 4 decimals
    // (e.g. D65 x = 0.3127, y = 0.3290), i.e. about +-1 K / +-3e-5 Duv of
    // physical uncertainty. The stored digits are enough for a sub-1e-5 uv
    // round-trip; more would be false precision.
    //   D65    x 0.3127 y 0.3290   D55    x 0.3324 y 0.3474
    //   D50    x 0.3457 y 0.3585   DCI-P3 x 0.3140 y 0.3510
    // ------------------------------------------------------------------
    struct TargetWhite { double cct; double duv; };
    constexpr TargetWhite kTargetD65   { 6504.3449, 0.00320720 };
    constexpr TargetWhite kTargetD55   { 5503.6515, 0.00324810 };
    constexpr TargetWhite kTargetD50   { 5000.7264, 0.00319328 };
    constexpr TargetWhite kTargetDCIP3 { 6302.7974, 0.01343665 };

    inline TargetWhite target_base (const int32_t preset) noexcept
    {
        switch (preset)
        {
            case target_D55:   return kTargetD55;
            case target_D50:   return kTargetD50;
            case target_DCIP3: return kTargetDCIP3;
            case target_D65:
            default:           return kTargetD65;
        }
    }

    // SOURCE white from the controls (Manual mode).
    // Temperature: coarse + fine in mireds -> Kelvin via the SAME helper the UI
    // uses for its readout (control_source_kelvin), so they cannot disagree.
    // Tint: UI is magenta-positive, CIE Duv is green-positive -> negate.
    inline void source_from_controls (const AlgoControls& c,
                                      double& cct, double& duv) noexcept
    {
        cct = control_source_kelvin(c);
        duv = -((static_cast<double>(c.tint) + static_cast<double>(c.tintFine))
                * AlgoCtl::kTintToDuv);
    }

    // TARGET white = delivery base + the LOOK offsets.
    //   warmth : mired SHIFT added to the base mireds. Positive -> higher mired
    //            -> lower K -> warmer image. Additive and base-independent.
    //   tint shift : same UI units as tint, magenta-positive -> negate.
    // With all offsets at 0 this is the exact delivery white (pure WB).
    inline void target_from_controls (const AlgoControls& c, const bool withLook,
                                      double& cct, double& duv) noexcept
    {
        const TargetWhite b = target_base(c.targetPreset);
        double m = kelvin_to_mired(b.cct);
        duv = b.duv;
        if (withLook)
        {
            m   += static_cast<double>(c.warmthMired) + static_cast<double>(c.warmthFineMired);
            duv -= (static_cast<double>(c.tintShift) + static_cast<double>(c.tintShiftFine))
                   * AlgoCtl::kTintToDuv;
        }
        // keep inside the LUT span (900 .. 40000 K)
        const double mLo = kelvin_to_mired(CCT_LUT_1931_2DEG::CCT_MAX);
        const double mHi = kelvin_to_mired(CCT_LUT_1931_2DEG::CCT_MIN);
        if (m < mLo) m = mLo;
        if (m > mHi) m = mHi;
        cct = mired_to_kelvin(m);
    }

    // Working space -> matrix pair. Maps YOUR eWorkingSpace.
    // ws_Auto falls back to sRGB as specified. Rec.2020 is enabled by
    // IMAGELAB2_HAVE_REC2020_MATRICES once the pair is in Algo2Rgb2XYZ.hpp.
    // Display P3 / ACEScg have no matrices in the tree yet and fall back to
    // sRGB: a documented fallback is preferable to a silently wrong matrix.
    // The SAME pair must build the gate (Algorithm_BuildMeasureSetup does).
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

    // Observer -> locus LUT. Must match the measurement.
    inline void pick_lut (const AlgoCCT::CctHandle<double>& h, const int32_t observer,
                          const AlgoCCT::LutRow*& lut, std::size_t& n) noexcept
    {
        if (obs_CIE_1964_10deg == observer) { const auto p = h.getLut_CIE_1964();
                                              lut = p.first; n = p.second; }
        else                                { const auto p = h.getLut_CIE_1931();
                                              lut = p.first; n = p.second; }
    }

    // clipCtl_Auto: clamp unless the output format is _Linear. With an
    // unknown format we cannot know, so we do not clamp - safe, because
    // integer and encoded formats clamp at ENCODE inside egress anyway, while
    // _Linear keeps its HDR (which an unconditional clamp would destroy).
    inline bool resolve_clip_from_controls (const AlgoControls& c) noexcept
    {
        if (clipCtl_Always == c.clipPolicy) return true;
        if (clipCtl_Never  == c.clipPolicy) return false;
        if (c.pixelFormat < 0)              return false;      // unknown
        return AlgoWB::resolve_clip(AlgoWB::clip_Auto,
                   static_cast<AlgoPrIngest::ePrPixelFormat>(c.pixelFormat));
    }
} // anonymous namespace


// =============================================================================
// SETUP - build the gate (scalar ingest) and the AVX2 measure context from the
// controls that change WHAT IS MEASURED: observer, working space, gate band.
// Call once, and again whenever Algorithm_SetupIsStale() says so.
// =============================================================================
bool Algorithm_BuildMeasureSetup (const AlgoCCT::CctHandle<double>& cctHdnl,
                                  const AlgoControls& params,
                                  AlgoMeasureSetup& setup) noexcept
{
    AlgoControls p = params;
    normalize_algo_controls(p);

    const AlgoCCT::LutRow* lut = nullptr;
    std::size_t lutN = 0u;
    pick_lut(cctHdnl, p.observer, lut, lutN);
    if (nullptr == lut || lutN < 2u) { setup.valid = false; return false; }

    double full = static_cast<double>(p.gateDuvFull);
    double zero = static_cast<double>(p.gateDuvZero);
    AlgoWB::sanitize_gate_band(full, zero);

    AlgoPrIngest::build_locus_gate(lut, lutN, working_space_to_XYZ(p.workingSpace),
                                   full, zero, setup.gate);
#ifdef __AVX2__
    AlgoPrIngest::avx2::build_measure_ctx(setup.gate, setup.ctx);
#endif
    setup.observer     = p.observer;
    setup.workingSpace = p.workingSpace;
    setup.gateFull     = p.gateDuvFull;
    setup.gateZero     = p.gateDuvZero;
    setup.valid        = true;
    return true;
}

bool Algorithm_SetupIsStale (const AlgoMeasureSetup& setup,
                             const AlgoControls& params) noexcept
{
    if (!setup.valid) return true;
    AlgoControls p = params;
    normalize_algo_controls(p);
    return setup.observer     != p.observer     ||
           setup.workingSpace != p.workingSpace ||
           setup.gateFull     != p.gateDuvFull  ||
           setup.gateZero     != p.gateDuvZero;
}

bool Algorithm_ReferenceIsStale (const AlgoWB::WbReference& ref,
                                 const AlgoControls& params) noexcept
{
    if (!ref.valid) return true;
    AlgoControls p = params;
    normalize_algo_controls(p);
    // A reference measured under a different observer, working space or gate
    // band is a DIFFERENT MEASUREMENT. (A change of target preset is only
    // advisory: the refinement targeted the old base, so re-Compute is
    // recommended but the reference is still about the same light.)
    return ref.builtObserver     != p.observer     ||
           ref.builtWorkingSpace != p.workingSpace ||
           ref.builtGateFull     != p.gateDuvFull  ||
           ref.builtGateZero     != p.gateDuvZero;
}


#ifdef __AVX2__
// =============================================================================
// COMPUTE - what the "Compute" button (measureNow) runs.
//   1. solve the super-pixel -> measured CCT/Duv (observer-matched)
//   2. classify confidence with the USER'S thresholds; refuse unless allowed
//   3. refine: solve for the effective source that lands ONE apply on the
//      NEUTRAL target (delivery base, WITHOUT the Look offsets - the reference
//      describes the light; the Look is applied on top every frame, so
//      changing Warmth later does not invalidate it)
//   4. record the setup it was measured under
// linearImage MUST be the normal linear image (ingest with confidenceMap
// OFF): refining against the confidence map would be meaningless.
// Returns false -> keep the previously stored reference.
// =============================================================================
bool Algorithm_ComputeReference (AlgoCCT::CctHandle<double>& cctHdnl,
                                 const AlgoMeasureSetup&     setup,
                                 const SuperPixel<double>&   superPixel,
                                 const double                keptFraction,
                                 const float*                linearImage,
                                 const int32_t               sizeX,
                                 const int32_t               sizeY,
                                 const AlgoControls&         params,
                                 AlgoWB::WbReference&        ref) noexcept
{
    if (!setup.valid || nullptr == linearImage || sizeX <= 0 || sizeY <= 0)
        return false;

    AlgoControls p = params;
    normalize_algo_controls(p);

    const double* const rgb2xyz = working_space_to_XYZ(p.workingSpace);
    const double* const xyz2rgb = XYZ_to_working_space(p.workingSpace);
    const eCOLOR_OBSERVER obs   = static_cast<eCOLOR_OBSERVER>(p.observer);

    double mCct = 0.0, mDuv = 0.0;
    superpixel_to_cct(superPixel, cctHdnl, rgb2xyz, obs, mCct, mDuv);

    const AlgoWB::ConfidenceThresholds thr = AlgoWB::make_thresholds(
        static_cast<double>(p.confHighPct), static_cast<double>(p.confLowPct),
        static_cast<double>(p.confDuvCaution), 0 != p.acceptLowConf);

    double tCct = 0.0, tDuv = 0.0;
    target_from_controls(p, /*withLook=*/false, tCct, tDuv);

    const AlgoCCT::LutRow* lut = nullptr;
    std::size_t lutN = 0u;
    pick_lut(cctHdnl, p.observer, lut, lutN);

    constexpr int32_t kMaxRefineRounds = 4;
    AlgoWB::WbReference out;
    const bool ok = AlgoWB::compute_reference(
        linearImage, sizeX, sizeY, setup.ctx, cctHdnl,
        lut, lutN, CCT_LUT_1931_2DEG::CCT_MIN, CCT_LUT_1931_2DEG::CCT_MAX,
        mCct, mDuv, keptFraction, tCct, tDuv,
        p.catModel, static_cast<double>(p.adaptationDegree),
        rgb2xyz, xyz2rgb, obs, thr, kMaxRefineRounds, out);
    if (!ok) return false;                     // caller keeps the old reference

    out.builtObserver     = p.observer;
    out.builtWorkingSpace = p.workingSpace;
    out.builtGateFull     = p.gateDuvFull;
    out.builtGateZero     = p.gateDuvZero;
    out.builtTargetPreset = p.targetPreset;
    ref = out;
    return true;
}
#endif // __AVX2__


// =============================================================================
// RENDER - every frame.
// =============================================================================
void Algorithm_Main
(
    AlgoCCT::CctHandle<double>& cctHdnl,
    const SuperPixel<double>&   superPixel,  // Previously computed SuperPixel
    const MemHandler&           memHandler,  // linear in/out RGB buffers
    const int32_t               sizeX,       // horizontal size in pixels
    const int32_t               sizeY,       // vertical size in pixels
    const AlgoControls&         params,      // Algorithm control parameters
    CctDuv<double>&             cct_duv,     // OUT: measured CCT and Duv
    const AlgoWB::WbReference*  reference    // stored reference (may be null)
) noexcept
{
    // Validate ONCE, on a local copy: invalid host data never reaches maths.
    AlgoControls p = params;
    normalize_algo_controls(p);

    double cct_computed = 0.0;      // STEP A' output: measured CCT [K]
    double duv_computed = 0.0;      // STEP A' output: measured Duv

    // ---- working-space matrices: the SAME pair the gate was built with ----
    const double* const rgb2xyz = working_space_to_XYZ(p.workingSpace);
    const double* const xyz2rgb = XYZ_to_working_space(p.workingSpace);

    // =======================================================================
    // STEP A' : MEASURE the incoming CCT/Duv - ALWAYS (unconditional).
    // The measured white is a first-class OUTPUT of every call (UI readout),
    // independent of wbMode and of whether a correction is applied.
    // IN : superPixel (from ingest), RGB->XYZ matrix, observer.
    // OUT: cct_duv (the measured source white).
    // =======================================================================
    superpixel_to_cct(superPixel, cctHdnl, rgb2xyz,
                      static_cast<eCOLOR_OBSERVER>(p.observer),
                      cct_computed, duv_computed);
    cct_duv.cct = cct_computed;
    cct_duv.duv = duv_computed;

    // =======================================================================
    // OUTPUT = CONFIDENCE MAP -> nothing to correct. Phase 1 wrote the map
    // into the INPUT buffer and the caller selects it for egress
    // (main.cpp: confidenceMap ? srcRGB_f32 : dstRGB_f32). Return without
    // touching the output buffer. The readout above is still produced.
    // =======================================================================
    if (0 != p.confidenceMap)
        return;

    // =======================================================================
    // STEP A : SOURCE white  =  WHITE BALANCE.
    //   Manual  : the sliders (mireds + tint) are authoritative.
    //   Measure : the stored reference if it is valid AND was measured under
    //             the current observer / working space / gate band; otherwise
    //             this frame's own measurement (correct, just not refined).
    // =======================================================================
    double src_cct = 0.0;
    double src_duv = 0.0;

    if (wbMode_Manual == p.wbMode)
    {
        source_from_controls(p, src_cct, src_duv);
    }
    else if (nullptr != reference && !Algorithm_ReferenceIsStale(*reference, p))
    {
        src_cct = reference->sourceCct;    // refined at Compute time
        src_duv = reference->sourceDuv;
    }
    else
    {
        src_cct = cct_computed;
        src_duv = duv_computed;
    }

    // No usable white point (e.g. no neutral content): pass the frame through
    // unchanged rather than correcting from garbage.
    if (!(src_cct > 0.0) || !std::isfinite(src_cct) || !std::isfinite(src_duv))
        return;

    // =======================================================================
    // STEP B : TARGET white  =  THE LOOK.
    // Delivery base plus Warmth / Tint-shift offsets. All offsets at 0 ->
    // exact delivery white (pure white balance). Positive Warmth -> warmer.
    // =======================================================================
    double tgt_cct = 0.0;
    double tgt_duv = 0.0;
    target_from_controls(p, /*withLook=*/true, tgt_cct, tgt_duv);

    // =======================================================================
    // STEP C : M_wb from source/target via the chromatic adaptation
    // transform. Built ONCE here - never per pixel.
    // =======================================================================
    const AlgoCCT::LutRow* lut = nullptr;
    std::size_t lutN = 0u;
    pick_lut(cctHdnl, p.observer, lut, lutN);

    double M_wb[9];
    const bool matrixOk = AlgoWB::build_wb_matrix(
                              lut, lutN,
                              CCT_LUT_1931_2DEG::CCT_MIN,   // both generated
                              CCT_LUT_1931_2DEG::CCT_MAX,   // tables share the grid
                              CctDuv<double>{ src_cct, src_duv },
                              CctDuv<double>{ tgt_cct, tgt_duv },
                              p.catModel,
                              static_cast<double>(p.adaptationDegree),
                              rgb2xyz, xyz2rgb,
                              M_wb);
    if (!matrixOk)
        return;

    // =======================================================================
    // STEP D : OUTPUT linear RGB from INPUT linear RGB. One 3x3 per pixel,
    // 8 at a time (AVX2 + FMA). Strength, highlight-safe and the clip policy
    // are folded into the MATRIX / a predicate - zero per-pixel cost.
    // =======================================================================
    AlgoWB::ApplyParams ap;
    ap.strength      = static_cast<double>(p.strength) * 0.01;   // % -> 0..1
    ap.highlightSafe = (0 != p.highlightSafe);

    double Mfinal[9];
    bool   unusedClip = false;
    AlgoWB::prepare_apply(M_wb, ap, Mfinal, unusedClip);
    const bool doClip = resolve_clip_from_controls(p);

#ifdef __AVX2__
    AlgoWB::apply_wb_avx2  (mem_linear_in(memHandler), mem_linear_out(memHandler),
                            sizeX, sizeY, Mfinal, doClip);
#else
    AlgoWB::apply_wb_scalar(mem_linear_in(memHandler), mem_linear_out(memHandler),
                            sizeX, sizeY, Mfinal, doClip);
#endif
    return;
}
