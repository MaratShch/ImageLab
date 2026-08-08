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

#include <cstdint>

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
        float    pickX;                  // normalized 0..1 of layer width
        float    pickY;                  // normalized 0..1 of layer height

        // Explicit "run the measurement now" trigger (so the heavy super-pixel
        // pass runs on demand, not every parameter tick). Momentary.
        //   SDK: PF_ADD_BUTTON          range: n/a            default: 0
        int32_t  measureNow;             // set by the button handler, then cleared

        // Coarse target color temperature [Kelvin]. Primary adjustment. In
        // Measure mode it is populated with the measured CCT; in Manual mode
        // the user drives it.
        //   SDK: PF_ADD_FLOAT_SLIDERX   range: 1000 .. 26000  default: 6500
        //   step (UI increment): 100 K -> 250 intervals / 251 stops. Chosen
        //   over 200 K so the standard whites land EXACTLY on the grid
        //   (3200 / 5600 / 6500 K are all 1000 + n*100), i.e. the 6500 K
        //   default is directly reachable by stepping. Perceptually a 100 K
        //   step is ~2.3 mired at daylight (small, visible nudge) growing to
        //   ~24 mired at 2000 K (larger warm-end clicks) - no dead zone.
        //   NOTE: the 100 K increment applies to keyboard/arrow STEPPING;
        //   dragging is continuous (sub-100 K values reachable by drag). Snap
        //   in the param handler only if hard quantization is wanted.
        float    temperatureK;

        // Fine trim around the coarse temperature [Kelvin]. Zero-centered.
        //   SDK: PF_ADD_FLOAT_SLIDERX   range: -100 .. +100   default: 0
        //   step (UI increment): 2 K. Range is HALF a coarse step (+-100 vs
        //   200 K) so fine and coarse stay distinct jobs - fine bridges
        //   BETWEEN coarse detents, it is not a second coarse control. 2 K
        //   keeps every step sub-visible (<1 mired) across the whole range,
        //   including the warm end where 4 K would just reach the ~1-mired
        //   just-noticeable edge at ~2000 K.
        //   CAVEAT: because steps are in Kelvin, the fine slider's EFFECT
        //   shrinks toward the cool end (+-100 K ~ 12.8 mired at 5600 K but
        //   ~1 mired at 20000 K); acceptable for a Kelvin-based v1, and the
        //   reason not to narrow the range below +-100 K.
        float    temperatureFineK;

        // Tint on the green<->magenta axis. UI convention: POSITIVE = magenta,
        // negative = green (matches Lightroom/Resolve); the engine inverts the
        // sign to CIE Duv internally.
        //   SDK: PF_ADD_FLOAT_SLIDERX   range: -100 .. +100   default: 0
        float    tint;

        // === TIER 2: mode & output =========================================

        // Measure vs Manual (see eWbMode).
        //   SDK: PF_ADD_POPUP           range: enum           default: Measure
        int32_t  wbMode;

        // Output selector kept as the ORIGINAL field: when non-zero the target
        // buffer receives the CONFIDENCE MAP (kept pixels as-is, excluded
        // pixels black) instead of the corrected image.
        //   SDK: PF_ADD_CHECKBOX        range: 0/1            default: 0 (OFF)
        int32_t  confidenceMap;

        // === TIER 3: correctness settings (usually left on defaults) ========

        // Working / RGB space of the buffer (see eWorkingSpace). Auto-detect
        // preferred; this is the highest-impact correctness switch.
        //   SDK: PF_ADD_POPUP           range: enum           default: Auto
        int32_t  workingSpace;

        // CIE observer (see eObserver). Advanced.
        //   SDK: PF_ADD_POPUP           range: enum           default: 2 deg
        int32_t  observer;

        // Chromatic adaptation model (see eCatModel). Advanced.
        //   SDK: PF_ADD_POPUP           range: enum           default: Bradford
        int32_t  catModel;

        // Degree of adaptation D for CAT02/CAT16 (0 = none, 1 = full). Advanced;
        // fixed at 1.0 for v1. Kept in the struct so exposing it later needs no
        // signature change.
        //   SDK: PF_ADD_FLOAT_SLIDERX   range: 0.0 .. 1.0     default: 1.0
        float    adaptationDegree;
    };


    // ------------------------------------------------------------------------
    // getAlgoControlsDefault - neutral starting state.
    //   Measure mode, target 6500 K / 0 tint, no fine trim, correctness
    //   settings on their safe defaults, full adaptation, and the confidence
    //   map DISABLED (renders the corrected image, per the original field).
    // ------------------------------------------------------------------------
    inline AlgoControls getAlgoControlsDefault()
    {
        AlgoControls c{};
        // Tier 1
        c.pickX            = 0.5f;                 // center of frame
        c.pickY            = 0.5f;
        c.measureNow       = 0;
        c.temperatureK     = 6500.0f;              // D65-ish neutral start
        c.temperatureFineK = 0.0f;
        c.tint             = 0.0f;                 // on the locus
        // Tier 2
        c.wbMode           = wbMode_Measure;
        c.confidenceMap    = 1;                    // OFF -> render adjusted image
        // Tier 3
        c.workingSpace     = ws_Auto;
        c.observer         = obs_CIE_1931_2deg;
        c.catModel         = cat_Bradford;
        c.adaptationDegree = 1.0f;
        return c;
    }


#endif // __IMAGELAB2_ALGO_CONTROL_HPP__
