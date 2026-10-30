#ifndef __IMAGELAB2_ALGO_CONTROL_HPP__
#define __IMAGELAB2_ALGO_CONTROL_HPP__

// =============================================================================
// AlgoControl.hpp - user-facing controls for the CCT white-balance algorithm.
//
// This struct is the SINGLE bundle of decisions the user makes; the host UI
// (After Effects parameters in v1, a custom panel later) writes into it, and
// the algorithm reads it. Each field notes the built-in AE SDK control that
// backs it, its range, and its default. getAlgoControlsDefault() returns the
// neutral starting state (measure mode, no manual offset, correctness
// settings on their safe defaults, confidence map OFF).
//
// TIERS (see design notes): the algorithm itself only strictly needs the
// target white point (temperature + tint) and the correctness settings
// (working space / observer / CAT). The rest describe HOW the target is
// obtained (measure vs manual) and WHAT the output shows (image vs map).
// =============================================================================

// REVISION NOTE (extension, not replacement):
//   * Color temperature is now controlled in MIREDS (10^6 / K). The former
//     Kelvin fields temperatureK / temperatureFineK are REPLACED by
//     temperatureMired / temperatureFineMired - keeping both would create two
//     parallel controls that could disagree. Kelvin is DERIVED for display by
//     control_source_kelvin() from the same stored value, so the UI and the
//     algorithm can never show different temperatures.
//     IMPACT: AE slider registration changes (range/precision below), and a
//     project saved with Kelvin values needs a one-time conversion
//     (mired = 1e6 / K) - see kelvin_to_mired().
//   * Added: tintFine, the Look group (warmth / tint shift / target preset),
//     strength, clip policy, confidence policy, locus-gate band, and the
//     host-filled pixelFormat. All existing fields, enums and semantics are
//     otherwise unchanged.
//   * Every numeric limit and default is a constexpr in AlgoCtl:: below, so
//     the host's parameter registration and normalize_algo_controls() use ONE
//     source of truth.
//   * FIXED: getAlgoControlsDefault() set confidenceMap = 1, which rendered
//     the confidence map instead of the corrected image (the field's own
//     documentation says default 0 / OFF).

#include <cstdint>
#include <cmath>

    // --- enumerations backing the popup (PF_ADD_POPUP) controls -------------

    // How the target white point is chosen.
    //   Measure : the auto-measured scene white point drives the result;
    //             the temperature/tint fields report what was measured.
    //   Manual  : the user dials temperature/tint directly; no measurement.
    enum eWbMode : int32_t
    {
        wbMode_Measure = 0,
        wbMode_Manual  = 1
    };

    // Working / RGB color space of the incoming buffer. CORRECTNESS-CRITICAL:
    // selects the RGB<->XYZ matrix AND the gate/luma. Prefer auto-detect from
    // the host's color management; expose as an override popup.
    enum eWorkingSpace : int32_t
    {
        ws_Auto      = 0,   // detect from host (falls back to sRGB if unknown)
        ws_sRGB_709  = 1,   // sRGB / Rec.709 primaries (HD / SDR) - safe default
        ws_Rec2020   = 2,   // Rec.2020 (HDR / wide gamut)
        ws_DisplayP3 = 3,   // Display P3
        ws_ACEScg    = 4    // ACEScg (AP1)
    };

    // CIE standard observer - selects the locus table. 2 deg is the CCT
    // convention and matches every display/scope; 10 deg is a surface-color
    // convention (Advanced, rarely changed).
    enum eObserver : int32_t
    {
        obs_CIE_1931_2deg  = 0,   // default
        obs_CIE_1964_10deg = 1
    };

    // Chromatic adaptation transform used to build the correction. Bradford
    // matches the Adobe/ICC ecosystem (closest agreement with Lumetri);
    // CAT16 is the most current; the rest are for completeness (Advanced).
    enum eCatModel : int32_t
    {
        cat_Bradford = 0,   // default
        cat_CAT16    = 1,
        cat_CAT02    = 2,
        cat_VonKries = 3
    };

    // Delivery white the LOOK offsets are measured from (the neutral target).
    // Technical choice, not creative: creative shifts are the Warmth offsets.
    enum eTargetPreset : int32_t
    {
        target_D65   = 0,   // 6504 K - sRGB / Rec.709 delivery - default
        target_D55   = 1,   // 5503 K - daylight film convention
        target_D50   = 2,   // 5003 K - print / prepress viewing
        target_DCIP3 = 3    // 6300 K - theatrical
    };

    // When corrected linear values are clamped to [0,1].
    //   Auto   : clamp UNLESS the output format is _Linear (HDR passes).
    //   Always / Never : forced. Values match AlgoWB::eClipPolicy (checked by
    //   static_assert in AlgorithmMain.cpp).
    enum eClipPolicyCtl : int32_t
    {
        clipCtl_Auto   = 0,   // default
        clipCtl_Always = 1,
        clipCtl_Never  = 2
    };


    // ------------------------------------------------------------------------
    // AlgoCtl - compile-time limits and defaults for EVERY control.
    // ONE source of truth: the host's PF_ADD_* registration should use these,
    // and normalize_algo_controls() clamps to exactly these.
    // Types match the fields they bound (float sliders -> float constants).
    // ------------------------------------------------------------------------
    namespace AlgoCtl
    {
        // ---- mired <-> Kelvin -------------------------------------------
        // Exact by definition: mired = 1e6 / K. 1e6 is exactly representable
        // in both float and double.
        constexpr double kMiredScale  = 1.0e6;
        constexpr float  kMiredScaleF = 1.0e6f;

        // Source temperature (coarse), mireds.
        //   min 25 mired   = 40000 K  (cool end of the CCT LUT)
        //   max 1000 mired =  1000 K  (warm limit of the previous Kelvin
        //                               control - preserved)
        //   default 1e6/6500 = 153.846... mired = 6500 K (previous default)
        // Step 1 mired: one mired is ~3.4e-4 in CIE-1960 uv, below the visual
        // threshold at EVERY temperature (a Kelvin step is not - dUv per
        // Kelvin varies ~460x across 2000..40000 K, per mired ~1.3x).
        constexpr float kTempMiredMin     = 25.0f;
        constexpr float kTempMiredMax     = 1000.0f;
        constexpr float kTempMiredDefault = 1.0e6f / 6500.0f;
        constexpr float kTempMiredStep    = 1.0f;

        // Source temperature (fine), mireds, added to the coarse value BEFORE
        // conversion. +-10 mired at 0.1 bridges ten coarse detents each way
        // at a tenth of the coarse resolution.
        constexpr float kTempFineMiredMin     = -10.0f;
        constexpr float kTempFineMiredMax     =  10.0f;
        constexpr float kTempFineMiredDefault =   0.0f;
        constexpr float kTempFineMiredStep    =   0.1f;

        // Tint UI units (existing +-100 convention, magenta-positive).
        // ONE UI unit = kTintToDuv in CIE Duv, so +-100 -> +-0.020 Duv,
        // covering real illuminants with headroom (validated extremes:
        // -0.0112 Sunset, +0.0137). Coarse and fine share this unit.
        constexpr double kTintToDuv        = 2.0e-4;
        constexpr float  kTintMin          = -100.0f;
        constexpr float  kTintMax          =  100.0f;
        constexpr float  kTintDefault      =    0.0f;
        constexpr float  kTintFineMin      =  -10.0f;
        constexpr float  kTintFineMax      =   10.0f;
        constexpr float  kTintFineDefault  =    0.0f;
        constexpr float  kTintFineStep     =    0.1f;

        // LOOK: warmth = MIRED SHIFT of the target (like a conversion filter:
        // +18 = 81A, +112 = 85, +131 = 85B, -131 = 80A). Positive = warmer.
        constexpr float kWarmthMin         = -150.0f;
        constexpr float kWarmthMax         =  150.0f;
        constexpr float kWarmthDefault     =    0.0f;
        constexpr float kWarmthFineMin     =  -10.0f;
        constexpr float kWarmthFineMax     =   10.0f;
        constexpr float kWarmthFineDefault =    0.0f;
        // Tint shift uses the tint UI units above.

        constexpr float kTintStep          =    1.0f;
        constexpr float kWarmthStep        =    1.0f;
        constexpr float kWarmthFineStep    =    0.1f;

        // Strength, percent. 0 = bypass, 100 = full correction.
        constexpr float kStrengthStep    =   0.1f;
        constexpr float kStrengthMin     =   0.0f;
        constexpr float kStrengthMax     = 100.0f;
        constexpr float kStrengthDefault = 100.0f;

        // Degree of adaptation (existing field).
        constexpr float kAdaptMin     = 0.0f;
        constexpr float kAdaptMax     = 1.0f;
        constexpr float kAdaptDefault = 1.0f;
        constexpr float kAdaptStep    = 0.01f;

        // Confidence policy (classification only - never touches pixels).
        // PLACEHOLDERS: calibrate against the real image batch.
        constexpr float kConfHighPctDefault  = 15.0f;
        constexpr float kConfLowPctDefault   =  2.0f;
        constexpr float kConfPctMin          =  0.0f;
        constexpr float kConfPctMax          = 100.0f;
        constexpr float kConfDuvCautionMin     = 0.001f;
        constexpr float kConfDuvCautionMax     = 0.080f;
        constexpr float kConfDuvCautionDefault = 0.010f;
        constexpr float kConfPctStep           = 0.1f;
        constexpr float kDuvStep               = 0.001f;

        // Locus-gate band (|Duv|). MOVES THE MEASUREMENT - see gateDuvFull.
        constexpr float kGateFullMin     = 0.001f;
        constexpr float kGateFullMax     = 0.100f;
        constexpr float kGateFullDefault = 0.010f;
        constexpr float kGateZeroMin     = 0.001f;
        constexpr float kGateZeroMax     = 0.200f;
        constexpr float kGateZeroDefault = 0.020f;
        // If zero <= full after clamping, zero is set to full * this factor.
        constexpr float kGateMinTaperRatio = 1.5f;

        // pixelFormat sentinel: host has not told us the format.
        constexpr int32_t kPixelFormatUnknown = -1;
    } // namespace AlgoCtl


    // ------------------------------------------------------------------------
    // AlgoControls - the full control bundle.
    //
    // Ranges/defaults are the tuned, validated values from the engine work;
    // the comment on each field names the built-in AE SDK control to use.
    // ------------------------------------------------------------------------
    struct AlgoControls
    {
        // === TIER 1: the core measure -> adjust -> apply loop ================

        // Eyedropper location for "measure the white point HERE" (in Measure
        // mode). In layer pixel coords; map through downsample before use.
        //   SDK: PF_ADD_POINT           range: layer extent   default: center
        // These fields are not usable yet.
        // (The measurement is a locus-gated WHOLE-FRAME super-pixel; there is
        //  no point-sampling path in the algorithm. No code reads these two
        //  fields. An eyedropper is new work, not wiring.)
        // [UI] Point control (PF_ADD_POINT), X   MIN 0.0  MAX 1.0  DEF 0.5   - not usable yet
        float    pickX;                  // normalized 0..1 of layer width
        // [UI] Point control (PF_ADD_POINT), Y   MIN 0.0  MAX 1.0  DEF 0.5   - not usable yet
        float    pickY;                  // normalized 0..1 of layer height

        // Explicit "run the measurement now" trigger (so the heavy super-pixel
        // pass runs on demand, not every parameter tick). Momentary.
        //   SDK: PF_ADD_BUTTON          range: n/a            default: 0
        // Algorithm side: Algorithm_ComputeReference() (AlgorithmMain.hpp) is
        // what the button handler calls; it measures, classifies confidence,
        // refines and returns the WbReference to store in sequence data.
        // [UI] Button (PF_ADD_BUTTON)            momentary 0/1    DEF 0     - label "Compute"
        int32_t  measureNow;             // set by the button handler, then cleared

        // SOURCE color temperature, COARSE, in MIREDS (10^6 / K). Primary
        // white-balance adjustment. In Measure mode the measured value is shown
        // (converted to mireds); in Manual mode the user drives it.
        //   SDK: PF_ADD_FLOAT_SLIDERX   range: 25 .. 1000 mired  default: 153.85
        //        (40000 .. 1000 K)                               (= 6500 K)
        //   step 1 mired. Display the Kelvin equivalent next to the slider via
        //   control_source_kelvin() - it is DERIVED from this field, never
        //   stored separately.
        //   Why mireds: equal mired steps are equal perceptual steps across the
        //   whole range (dUv per mired varies only ~1.3x from 2000 to
        //   40000 K, per Kelvin ~460x). REPLACES the former temperatureK.
        // [UI] Float slider   MIN 25  MAX 1000  DEF 153.846 (= 6500 K)  STEP 1 mired  - show Kelvin beside it
        float    temperatureMired;

        // SOURCE color temperature, FINE trim, mireds. Zero-centred, ADDED to
        // the coarse value before conversion (never applied as a second
        // correction). REPLACES the former temperatureFineK.
        //   SDK: PF_ADD_FLOAT_SLIDERX   range: -10 .. +10 mired  default: 0
        //   step 0.1 mired - a tenth of a coarse detent, ~3.4e-5 in uv.
        // [UI] Float slider   MIN -10  MAX +10  DEF 0  STEP 0.1 mired
        float    temperatureFineMired;

        // Tint on the green<->magenta axis. UI convention: POSITIVE = magenta,
        // negative = green (matches Lightroom/Resolve); the engine inverts the
        // sign to CIE Duv internally.
        //   SDK: PF_ADD_FLOAT_SLIDERX   range: -100 .. +100   default: 0
        //   One UI unit = AlgoCtl::kTintToDuv (2e-4) Duv -> +-0.020 Duv.
        // [UI] Float slider   MIN -100  MAX +100  DEF 0  STEP 1   (magenta +, green -)
        float    tint;

        // Fine tint trim, same UI units as tint, ADDED to it before conversion.
        //   SDK: PF_ADD_FLOAT_SLIDERX   range: -10 .. +10     default: 0
        //   step 0.1 -> 2e-5 Duv.
        // [UI] Float slider   MIN -10  MAX +10  DEF 0  STEP 0.1
        float    tintFine;

        // === TIER 2: mode & output =========================================

        // Measure vs Manual (see eWbMode).
        //   SDK: PF_ADD_POPUP           range: enum           default: Measure
        // [UI] Popup list-box (PF_ADD_POPUP)   Measure | Manual   DEF Measure
        int32_t  wbMode;

        // Output selector kept as the ORIGINAL field: when non-zero the target
        // buffer receives the CONFIDENCE MAP (kept pixels as-is, excluded
        // pixels black) instead of the corrected image.
        //   SDK: PF_ADD_CHECKBOX        range: 0/1            default: 0 (OFF)
        // [UI] Check-box (PF_ADD_CHECKBOX)   MIN 0  MAX 1  DEF 0 (OFF = corrected image)
        int32_t  confidenceMap;

        // === TIER 3: correctness settings (usually left on defaults) ========

        // Working / RGB space of the buffer (see eWorkingSpace). Auto-detect
        // preferred; this is the highest-impact correctness switch.
        //   SDK: PF_ADD_POPUP           range: enum           default: Auto
        // [UI] Popup list-box   Auto | sRGB/Rec.709 | Rec.2020 | Display P3 | ACEScg   DEF Auto
        int32_t  workingSpace;

        // CIE observer (see eObserver). Advanced.
        //   SDK: PF_ADD_POPUP           range: enum           default: 2 deg
        // [UI] Popup list-box   CIE 1931 2 deg | CIE 1964 10 deg   DEF CIE 1931 2 deg
        int32_t  observer;

        // Chromatic adaptation model (see eCatModel). Advanced.
        //   SDK: PF_ADD_POPUP           range: enum           default: Bradford
        // [UI] Popup list-box   Bradford | CAT16 | CAT02 | Von Kries   DEF Bradford
        int32_t  catModel;

        // Degree of adaptation D for CAT02/CAT16 (0 = none, 1 = full). Advanced;
        // fixed at 1.0 for v1. Kept in the struct so exposing it later needs no
        // signature change.
        //   SDK: PF_ADD_FLOAT_SLIDERX   range: 0.0 .. 1.0     default: 1.0
        // [UI] Float slider   MIN 0.0  MAX 1.0  DEF 1.0  STEP 0.01   (Advanced)
        float    adaptationDegree;

        // === TIER 4: LOOK - the creative warm/cool control =================
        // These move the TARGET white (what the image is rendered TO), while
        // the temperature/tint fields above move the SOURCE (what the light
        // WAS). All are OFFSETS defaulting to 0, so "no look" is an exact
        // no-op and a look copied to another shot keeps its meaning.

        // Warmth, mired SHIFT of the target. POSITIVE = WARMER image.
        //   SDK: PF_ADD_FLOAT_SLIDERX   range: -150 .. +150   default: 0
        // [UI] Float slider   MIN -150  MAX +150  DEF 0  STEP 1 mired   (+ = warmer)
        float    warmthMired;

        // Warmth fine trim, mireds, added to warmthMired.
        //   SDK: PF_ADD_FLOAT_SLIDERX   range: -10 .. +10     default: 0
        // [UI] Float slider   MIN -10  MAX +10  DEF 0  STEP 0.1 mired
        float    warmthFineMired;

        // Tint shift of the target, tint UI units (magenta-positive).
        //   SDK: PF_ADD_FLOAT_SLIDERX   range: -100 .. +100   default: 0
        // [UI] Float slider   MIN -100  MAX +100  DEF 0  STEP 1   (magenta +, green -)
        float    tintShift;

        // Tint shift fine trim, tint UI units.
        //   SDK: PF_ADD_FLOAT_SLIDERX   range: -10 .. +10     default: 0
        // [UI] Float slider   MIN -10  MAX +10  DEF 0  STEP 0.1
        float    tintShiftFine;

        // Delivery base the Look offsets are measured from (eTargetPreset).
        //   SDK: PF_ADD_POPUP           range: enum           default: D65
        // [UI] Popup list-box   D65 | D55 | D50 | DCI-P3   DEF D65
        int32_t  targetPreset;

        // === TIER 5: rendering ==============================================

        // Strength, percent: blends the correction MATRIX toward identity.
        // 0 = exact bypass, 100 = full correction. Zero per-pixel cost.
        //   SDK: PF_ADD_FLOAT_SLIDERX   range: 0 .. 100       default: 100
        // [UI] Float slider   MIN 0  MAX 100  DEF 100  STEP 0.1 (percent)
        float    strength;

        // Clip policy (eClipPolicyCtl).
        //   SDK: PF_ADD_POPUP           range: enum           default: Auto
        // [UI] Popup list-box   Auto | Always | Never   DEF Auto
        int32_t  clipPolicy;

        // Internal option: scale the matrix so no channel is amplified
        // (prevents NEW clipping, darkens instead). Not a recommended user
        // control - Strength covers the need. 0 = off.
        // [UI] Check-box (internal; do NOT expose)   MIN 0  MAX 1  DEF 0
        int32_t  highlightSafe;

        // === TIER 6: confidence policy - VISIBLE and USER-CONTROLLABLE =====
        // These only CLASSIFY the measurement (HIGH / LOW / NONE) and decide
        // whether Compute may accept a weak frame. They never touch pixels.

        //   SDK: PF_ADD_FLOAT_SLIDERX   range: 0 .. 100 %     default: 15
        // [UI] Float slider   MIN 0  MAX 100  DEF 15  STEP 0.1 (percent)
        float    confHighPct;
        //   SDK: PF_ADD_FLOAT_SLIDERX   range: 0 .. 100 %     default: 2
        // [UI] Float slider   MIN 0  MAX 100  DEF 2  STEP 0.1 (percent, <= confHighPct)
        float    confLowPct;
        //   SDK: PF_ADD_FLOAT_SLIDERX   range: 0.001 .. 0.080 default: 0.010
        // [UI] Float slider   MIN 0.001  MAX 0.080  DEF 0.010  STEP 0.001
        float    confDuvCaution;
        //   SDK: PF_ADD_CHECKBOX        range: 0/1            default: 0
        // [UI] Check-box   MIN 0  MAX 1  DEF 0
        int32_t  acceptLowConf;

        // === TIER 7: locus-gate band - MOVES THE MEASUREMENT ================
        // Unlike the confidence thresholds, these change WHICH PIXELS ARE
        // COUNTED, so the reported CCT itself changes. Consumed at SETUP by
        // Algorithm_BuildMeasureSetup(); changing either requires rebuilding
        // the gate and a fresh Compute.
        //   SDK: PF_ADD_FLOAT_SLIDERX   range: 0.001 .. 0.100 default: 0.010
        // [UI] Float slider   MIN 0.001  MAX 0.100  DEF 0.010  STEP 0.001   (Advanced)
        float    gateDuvFull;
        //   SDK: PF_ADD_FLOAT_SLIDERX   range: 0.001 .. 0.200 default: 0.020
        // [UI] Float slider   MIN 0.001  MAX 0.200  DEF 0.020  STEP 0.001   (Advanced, > gateDuvFull)
        float    gateDuvZero;

        // === HOST-FILLED (not a user control) ================================
        // The pixel format of THIS render call (AlgoPrIngest::ePrPixelFormat as
        // int32), filled by the host adapter. Needed by clipCtl_Auto to know
        // whether the output is _Linear. AlgoCtl::kPixelFormatUnknown when the
        // host has not set it; Auto then does not clamp (safe: integer and
        // encoded outputs still clamp at encode, _Linear keeps its HDR).
        // [UI] NOT a UI control - filled by the host per render call   DEF -1 (unknown)
        int32_t  pixelFormat;
    };


    // ------------------------------------------------------------------------
    // getAlgoControlsDefault - neutral starting state.
    //   Measure mode, source 6500 K (153.85 mired) / 0 tint, no fine trim,
    //   no Look, full strength, correctness settings on their safe defaults,
    //   full adaptation, and the confidence map DISABLED.
    // ------------------------------------------------------------------------
    inline AlgoControls getAlgoControlsDefault()
    {
        AlgoControls c{};
        // Tier 1
        c.pickX                = 0.5f;          // center of frame (not usable yet)
        c.pickY                = 0.5f;
        c.measureNow           = 0;
        c.temperatureMired     = AlgoCtl::kTempMiredDefault;      // 6500 K
        c.temperatureFineMired = AlgoCtl::kTempFineMiredDefault;
        c.tint                 = AlgoCtl::kTintDefault;           // on the locus
        c.tintFine             = AlgoCtl::kTintFineDefault;
        // Tier 2
        c.wbMode               = wbMode_Measure;
        // FIXED: was 1 with the comment "OFF -> render adjusted image", but a
        // non-zero value RENDERS THE MAP (see the field doc and main.cpp's
        // egress ternary). A defaulted bundle therefore never corrected.
        c.confidenceMap        = 0;             // OFF -> render adjusted image
        // Tier 3
        c.workingSpace         = ws_Auto;
        c.observer             = obs_CIE_1931_2deg;
        c.catModel             = cat_Bradford;
        c.adaptationDegree     = AlgoCtl::kAdaptDefault;
        // Tier 4 - Look: all neutral
        c.warmthMired          = AlgoCtl::kWarmthDefault;
        c.warmthFineMired      = AlgoCtl::kWarmthFineDefault;
        c.tintShift            = AlgoCtl::kTintDefault;
        c.tintShiftFine        = AlgoCtl::kTintFineDefault;
        c.targetPreset         = target_D65;
        // Tier 5
        c.strength             = AlgoCtl::kStrengthDefault;
        c.clipPolicy           = clipCtl_Auto;
        c.highlightSafe        = 0;
        // Tier 6
        c.confHighPct          = AlgoCtl::kConfHighPctDefault;
        c.confLowPct           = AlgoCtl::kConfLowPctDefault;
        c.confDuvCaution       = AlgoCtl::kConfDuvCautionDefault;
        c.acceptLowConf        = 0;
        // Tier 7
        c.gateDuvFull          = AlgoCtl::kGateFullDefault;
        c.gateDuvZero          = AlgoCtl::kGateZeroDefault;
        // host-filled
        c.pixelFormat          = AlgoCtl::kPixelFormatUnknown;
        return c;
    }


    // ------------------------------------------------------------------------
    // Mired <-> Kelvin. The ONLY conversion used anywhere: the UI readout and
    // the algorithm both call these, so they cannot disagree. Computed in
    // DOUBLE from the stored float mired value.
    // ------------------------------------------------------------------------
    constexpr double mired_to_kelvin (const double mired) noexcept
    { return AlgoCtl::kMiredScale / mired; }

    constexpr double kelvin_to_mired (const double kelvin) noexcept
    { return AlgoCtl::kMiredScale / kelvin; }

    // Effective SOURCE temperature in mireds: coarse + fine, clamped to the
    // coarse range (a fine trim cannot push past the documented limits).
    inline double control_source_mired (const AlgoControls& c) noexcept
    {
        double m = static_cast<double>(c.temperatureMired) +
                   static_cast<double>(c.temperatureFineMired);
        const double lo = static_cast<double>(AlgoCtl::kTempMiredMin);
        const double hi = static_cast<double>(AlgoCtl::kTempMiredMax);
        if (!(m >= lo)) m = lo;          // also catches NaN
        if (m > hi)     m = hi;
        return m;
    }

    // The Kelvin value to DISPLAY next to the temperature slider - derived
    // from exactly the value the algorithm uses.
    inline double control_source_kelvin (const AlgoControls& c) noexcept
    { return mired_to_kelvin(control_source_mired(c)); }

    // UI rounding: whole Kelvin. Rounding is display-only; the algorithm
    // always uses the unrounded double.
    inline int32_t control_source_kelvin_display (const AlgoControls& c) noexcept
    { return static_cast<int32_t>(std::floor(control_source_kelvin(c) + 0.5)); }


    // ------------------------------------------------------------------------
    // normalize_algo_controls - validate and repair a control bundle.
    //   * NaN / Inf -> the field's default
    //   * out of range -> clamped to the AlgoCtl limits
    //   * unknown enum values -> the enum's default
    //   * checkboxes -> 0 / 1
    //   * confLowPct is forced <= confHighPct
    //   * gateDuvZero is forced > gateDuvFull
    // Returns the number of fields it had to change (0 = input was valid), so
    // the host can log or flag a corrupted project. Call this at the boundary
    // (after the host fills the struct); Algorithm_Main also applies it to a
    // local copy defensively.
    // ------------------------------------------------------------------------
    namespace AlgoCtlDetail
    {
        inline int32_t fix_float (float& v, const float lo, const float hi,
                                  const float def) noexcept
        {
            if (!std::isfinite(v)) { v = def; return 1; }
            if (v < lo)            { v = lo;  return 1; }
            if (v > hi)            { v = hi;  return 1; }
            return 0;
        }
        inline int32_t fix_enum (int32_t& v, const int32_t lo, const int32_t hi,
                                 const int32_t def) noexcept
        {
            if (v < lo || v > hi) { v = def; return 1; }
            return 0;
        }
        inline int32_t fix_bool (int32_t& v) noexcept
        {
            if (0 != v && 1 != v) { v = 1; return 1; }   // any non-zero = on
            return 0;
        }
    } // namespace AlgoCtlDetail

    inline int32_t normalize_algo_controls (AlgoControls& c) noexcept
    {
        using namespace AlgoCtl;
        using namespace AlgoCtlDetail;
        int32_t n = 0;
        // pickX / pickY are not used, but keep them sane for the host.
        n += fix_float(c.pickX, 0.0f, 1.0f, 0.5f);
        n += fix_float(c.pickY, 0.0f, 1.0f, 0.5f);
        n += fix_bool (c.measureNow);
        n += fix_float(c.temperatureMired,     kTempMiredMin,     kTempMiredMax,     kTempMiredDefault);
        n += fix_float(c.temperatureFineMired, kTempFineMiredMin, kTempFineMiredMax, kTempFineMiredDefault);
        n += fix_float(c.tint,                 kTintMin,          kTintMax,          kTintDefault);
        n += fix_float(c.tintFine,             kTintFineMin,      kTintFineMax,      kTintFineDefault);
        n += fix_enum (c.wbMode,       wbMode_Measure,    wbMode_Manual,      wbMode_Measure);
        n += fix_bool (c.confidenceMap);
        n += fix_enum (c.workingSpace, ws_Auto,           ws_ACEScg,          ws_Auto);
        n += fix_enum (c.observer,     obs_CIE_1931_2deg, obs_CIE_1964_10deg, obs_CIE_1931_2deg);
        n += fix_enum (c.catModel,     cat_Bradford,      cat_VonKries,       cat_Bradford);
        n += fix_float(c.adaptationDegree,     kAdaptMin,         kAdaptMax,         kAdaptDefault);
        n += fix_float(c.warmthMired,          kWarmthMin,        kWarmthMax,        kWarmthDefault);
        n += fix_float(c.warmthFineMired,      kWarmthFineMin,    kWarmthFineMax,    kWarmthFineDefault);
        n += fix_float(c.tintShift,            kTintMin,          kTintMax,          kTintDefault);
        n += fix_float(c.tintShiftFine,        kTintFineMin,      kTintFineMax,      kTintFineDefault);
        n += fix_enum (c.targetPreset, target_D65,        target_DCIP3,       target_D65);
        n += fix_float(c.strength,             kStrengthMin,      kStrengthMax,      kStrengthDefault);
        n += fix_enum (c.clipPolicy,   clipCtl_Auto,      clipCtl_Never,      clipCtl_Auto);
        n += fix_bool (c.highlightSafe);
        n += fix_float(c.confHighPct,          kConfPctMin,       kConfPctMax,       kConfHighPctDefault);
        n += fix_float(c.confLowPct,           kConfPctMin,       kConfPctMax,       kConfLowPctDefault);
        if (c.confLowPct > c.confHighPct) { c.confLowPct = c.confHighPct; ++n; }
        n += fix_float(c.confDuvCaution,       kConfDuvCautionMin, kConfDuvCautionMax, kConfDuvCautionDefault);
        n += fix_bool (c.acceptLowConf);
        n += fix_float(c.gateDuvFull,          kGateFullMin,      kGateFullMax,      kGateFullDefault);
        n += fix_float(c.gateDuvZero,          kGateZeroMin,      kGateZeroMax,      kGateZeroDefault);
        if (!(c.gateDuvZero > c.gateDuvFull))
        {
            float z = c.gateDuvFull * kGateMinTaperRatio;
            if (z > kGateZeroMax) z = kGateZeroMax;
            // if even the max cannot exceed full, pull full down
            if (!(z > c.gateDuvFull)) c.gateDuvFull = z / kGateMinTaperRatio;
            c.gateDuvZero = z;
            ++n;
        }
        // pixelFormat is host-filled; anything negative means "unknown".
        if (c.pixelFormat < AlgoCtl::kPixelFormatUnknown)
        { c.pixelFormat = AlgoCtl::kPixelFormatUnknown; ++n; }
        return n;
    }


#endif // __IMAGELAB2_ALGO_CONTROL_HPP__
