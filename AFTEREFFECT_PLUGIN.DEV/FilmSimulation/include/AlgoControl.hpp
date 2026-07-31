#pragma once

#include <cstdint>
#include "AE_Effect.h"

// ===========================================================================
//  AlgoControls -- user-facing parameters of the FilmSimulation core.
//
//  Mirrors the Python reference RenderSettings field for field (defaults are
//  IDENTICAL to film_sim.py so C++ output can be diffed against the reference
//  with no parameter translation), then adds the temporal/damage block that
//  exists only on the C++/plugin side (pipeline stages 3c, 9b, 15, 16).
//
//  CONVENTIONS
//    - Every "Scale" parameter multiplies a per-stock physical value from the
//      film profile. 1.0 = the calibrated profile value; 0 = off. This is why
//      the ranges look narrow: the profile carries the physics, the control
//      carries taste.
//    - Enum-like selectors are int32_t indices into the generated tables
//      (film::GetFilmDatabase() order = alphabetical, stable). Index 0 of the
//      print/format selectors means "use the stock's own default", so a fresh
//      instance renders every stock authentically with zero configuration.
//    - frameIdx and frameRate are NOT here: they are per-render facts supplied
//      by the host, passed as Algorithm_Main arguments.
//    - Ae SDK control notes use the CS6+ macro names (PF_ADD_POPUP,
//      PF_ADD_FLOAT_SLIDERX, PF_ADD_CHECKBOXX, PF_ADD_SLIDER, PF_ADD_TOPIC).
//      Group the fields into PF_ADD_TOPIC twirlies exactly as the section
//      banners below; Premiere honours the same macros (it ignores
//      START_COLLAPSED -- documented host difference).
//
//  PERFORMANCE NOTES quote the measured Python-reference costs at HD
//  (1258 ms/frame total) as RATIOS -- the C++ ratios will be similar even
//  when absolute times are ~50x lower.
// ===========================================================================

struct AlgoControls
{
    // =======================================================================
    //  GROUP 1 -- FILM & PROCESS CHAIN            (PF_ADD_TOPIC "Film Stock")
    // =======================================================================

    // -----------------------------------------------------------------------
    // WHAT : Which emulsion to simulate. Index into film::GetFilmDatabase(),
    //        which is ALPHABETICAL and stable across regenerations (62 stocks:
    //        AGFACOLOR_NEU_1936 ... TECHNICOLOR_THREE_STRIP).
    // WHY  : The master control -- selects curves, grain, MTF, halation,
    //        couplers, spectral response, native gauge, everything physical.
    // RANGE: 0 .. 61          DEFAULT: index of KODAK_VISION3_500T_5219
    // UNITS: table index (dimensionless)
    // IMAGE: total -- every stage reads the selected profile.
    // PERF : indirect. Stocks with zero halation gains, identity matrices, no
    //        reseau and no print stage (reversal) skip whole passes; Dufaycolor
    //        (reseau) and 3-strip (taking matrix) are the most expensive picks.
    // AE   : PF_ADD_POPUP, one item per stock name, alphabetical. Do NOT use a
    //        slider: the order is nominal, not ordinal. Set
    //        PF_ParamFlag_SUPERVISE if the UI greys out mono-only controls.
    // -----------------------------------------------------------------------
    int32_t filmStockIdx;

    // -----------------------------------------------------------------------
    // WHAT : Print stock / display transform for NEGATIVE stocks.
    //        0 = the stock's own default_print (recommended);
    //        1..N = explicit entry in film::GetPrintStocks() order
    //        (SCAN_DI, KODAK_2383_RELEASE, DUPE_FINE_GRAIN, TECHNICOLOR_IB,
    //        TASMA_POSITIVE_28).
    // WHY  : A negative is an intermediate; the print decides the final
    //        contrast, palette and Dmax. Reversal stocks IGNORE this (the film
    //        is the positive) -- grey the control out for them.
    // RANGE: 0 .. 5           DEFAULT: 0 (stock default)
    // UNITS: table index
    // IMAGE: system gamma (scan ~1.0 vs theatrical ~1.6), print dye palette,
    //        print grain, black level.
    // PERF : negligible -- same number of passes either way.
    // AE   : PF_ADD_POPUP with "(Stock default)" as item 1. SUPERVISE to
    //        disable when the selected stock is reversal.
    // -----------------------------------------------------------------------
    int32_t printStockIdx;

    // -----------------------------------------------------------------------
    // WHAT : Film gauge override. 0 = the stock's own default_format
    //        (8 mm stock renders as 8 mm, 35 mm still as 36 mm, etc.);
    //        1..14 = explicit entry of film::GetFilmFormats().
    // WHY  : px_per_mm = width_px / gauge_mm converts every physical number
    //        (um, cycles/mm) into pixels. THE resolution-independence
    //        mechanism, and the whole difference between 8 mm and 35 mm from
    //        one emulsion (measured: same stock, 3x grain size, 4-18x less
    //        resolvable detail across the frame).
    // RANGE: 0 .. 14          DEFAULT: 0 (stock native)
    // UNITS: table index; underlying value in mm of frame width
    // IMAGE: grain size in px, sharpness limit, halation reach, weave scale.
    // PERF : smaller gauge = physically larger blur radii in px = larger
    //        kernels / more FFT benefit. 8 mm at 4K is the worst case.
    // AE   : PF_ADD_POPUP, first item "(Stock native gauge)".
    // -----------------------------------------------------------------------
    int32_t filmFormatIdx;

    // -----------------------------------------------------------------------
    // WHAT : Number of duplication GENERATIONS (interpositive / dupe-negative
    //        pairs) inserted before the print, each pass = blur THEN grain on
    //        DUPE_FINE_GRAIN stock.
    // WHY  : Release prints of the photochemical era were 3rd-4th generation;
    //        generation loss (softening + grain accumulation) is the dominant
    //        part of the "old print" look, distinct from the negative's own
    //        character.
    // RANGE: 0 .. 6           DEFAULT: 0 (camera negative printed directly)
    // UNITS: generations (integer count)
    // IMAGE: each step visibly softens fine detail and adds a layer of dupe
    //        grain; 2 = classic theatrical release feel, 4+ = worn archival.
    // PERF : LINEAR and significant -- each generation adds one full blur +
    //        grain pass (~15-20% of frame time each in the reference).
    // AE   : PF_ADD_SLIDER (integer), 0..6.
    // -----------------------------------------------------------------------
    int32_t generations;

    // =======================================================================
    //  GROUP 2 -- EXPOSURE & SCENE                  (PF_ADD_TOPIC "Exposure")
    // =======================================================================

    // -----------------------------------------------------------------------
    // WHAT : Relative exposure applied to scene-linear input, out = in * 2^x.
    //        (Algo_02 -- already implemented and reference-verified.)
    // WHY  : Moves the scene along the H&D curve: the film's latitude, toe
    //        and shoulder behaviour ARE this control's response. The main
    //        creative control after stock choice, and the only honest way to
    //        show a stock's under/over-exposure character (e.g. Portra 800's
    //        underexposure latitude vs slide film's 5-stop cliff).
    // RANGE: -8.0 .. +8.0     DEFAULT: 0.0
    // UNITS: photographic stops (1 stop = 2x light)
    // IMAGE: negative: gentle toe/shoulder migration; reversal: rapid clipping
    //        beyond ~±2.5 stops -- correct, that is slide film.
    // PERF : free (one multiply per pixel, fused into the first pass).
    // AE   : PF_ADD_FLOAT_SLIDERX, range -8..8, slider -4..+4, 2 decimals,
    //        PF_Precision_HUNDREDTHS.
    // -----------------------------------------------------------------------
    float exposureStops;

    // -----------------------------------------------------------------------
    // WHAT : Colour temperature of the SCENE illuminant, compared against the
    //        stock's balance_kelvin to derive per-channel gains (von Kries).
    // WHY  : Tungsten film in daylight goes blue, daylight film under bulbs
    //        goes orange -- shooting-practice reality. ORWO sits at 4500 K,
    //        DS-4 at 5600 K, tungsten cine at 3200 K: the mismatch is look.
    // RANGE: 2000 .. 12000    DEFAULT: 5500
    // UNITS: kelvin
    // IMAGE: global cast BEFORE the curve, so casts crush asymmetrically into
    //        toe/shoulder like real cross-shot film, not like a WB slider.
    // PERF : free (three gains, fused).
    // AE   : PF_ADD_FLOAT_SLIDERX 2000..12000, default 5500, 0 decimals.
    //        (A popup of presets Daylight 5500 / Tungsten 3200 / Shade 7000
    //        plus this slider is friendlier; popup drives slider via
    //        PF_ParamFlag_SUPERVISE.)
    // -----------------------------------------------------------------------
    float sceneKelvin;

    // -----------------------------------------------------------------------
    // WHAT : How much of the scene/stock kelvin mismatch is corrected before
    //        the curve. 0 = none (shoot uncorrected, full cast), 1 = fully
    //        corrected (as if the right conversion filter was on the lens).
    // WHY  : Real practice was BOTH: sometimes an 80A/85 filter, sometimes
    //        shot raw and timed later. This picks the point between.
    // RANGE: 0.0 .. 1.0       DEFAULT: 0.0 (matches Python reference)
    // UNITS: fraction
    // IMAGE: 0 keeps the full cross-shooting cast; 1 neutralises it.
    // PERF : free.
    // AE   : PF_ADD_FLOAT_SLIDERX 0..1, 2 decimals.
    // -----------------------------------------------------------------------
    float wbStrength;

    // -----------------------------------------------------------------------
    // WHAT : Display-linear value that an 18% scene grey lands on after the
    //        whole chain. The anchor the per-frame printer-lights solve hits.
    // WHY  : The photochemical "printer point" in one number. Raising it
    //        prints brighter, lowering prints darker -- while every nonlinear
    //        stage keeps behaving correctly around the new anchor.
    // RANGE: 0.05 .. 0.50     DEFAULT: 0.18
    // UNITS: display-linear fraction
    // IMAGE: overall print density; unlike output gain it interacts with the
    //        curve, so highlights/shadows roll instead of clip.
    // PERF : free (changes the setup solve only, 0.4% of frame).
    // AE   : PF_ADD_FLOAT_SLIDERX 0.05..0.5, default 0.18, 3 decimals.
    // -----------------------------------------------------------------------
    float greyTarget;

    // =======================================================================
    //  GROUP 3 -- PHYSICAL LOOK SCALES        (PF_ADD_TOPIC "Film Character")
    //  All multiply per-stock profile physics. 1.0 = calibrated truth.
    // =======================================================================

    // -----------------------------------------------------------------------
    // WHAT : Multiplies the stock's calibrated grain amplitude (its
    //        rms_granularity, applied in the density domain with the
    //        sqrt(D - dmin + fog) law).
    // WHY  : Datasheet-true grain is sometimes more (or less) than a shot
    //        wants; also the only honest "make it filmic" strength knob.
    // RANGE: 0.0 .. 4.0       DEFAULT: 1.0
    // UNITS: multiplier of profile RMS
    // IMAGE: 0 = clinically clean (loudest digital tell -- avoid); 1 = the
    //        film's measured granularity; >2 = pushed/expired feel. Grain
    //        SIZE does not change (that is gauge + clump_um); only amplitude.
    // PERF : 0 skips the grain synthesis pass entirely (~10% of frame);
    //        any nonzero value costs the same.
    // AE   : PF_ADD_FLOAT_SLIDERX 0..4, slider 0..2, 2 decimals.
    // -----------------------------------------------------------------------
    float grainScale;

    // -----------------------------------------------------------------------
    // WHAT : Multiplies the stock's halation gains (red-dominant base-bounce
    //        glow, energy-conserving, thresholded to highlights).
    // WHY  : Halation strength varied with remjet/AH quality -- CineStill 1.05
    //        vs VISION3 0.3 vs MACO CUBE 0. Taste control around truth.
    // RANGE: 0.0 .. 4.0       DEFAULT: 1.0
    // UNITS: multiplier of profile gains
    // IMAGE: red-orange bloom around speculars and windows; 0 = digital-crisp
    //        highlights, 2+ = CineStill-style signature glow on any stock.
    // PERF : 0 skips three threshold+blur passes (~12% of frame) on stocks
    //        that have halation; no cost change on zero-halation stocks.
    // AE   : PF_ADD_FLOAT_SLIDERX 0..4, slider 0..2, 2 decimals.
    // -----------------------------------------------------------------------
    float halationScale;

    // -----------------------------------------------------------------------
    // WHAT : Multiplies DIR-coupler inter-image strength (both the blurred
    //        cross-channel inhibition and the sharp edge term).
    // WHY  : Couplers raise saturation WITHOUT raising gamma -- the modern
    //        colour-negative signature. Scaling down de-modernises a stock;
    //        up gives the hyper-clean 90s Ektachrome ad look.
    // RANGE: 0.0 .. 2.0       DEFAULT: 1.0
    // UNITS: multiplier of CouplerSpec strengths
    // IMAGE: colour separation/saturation and edge "snap", constant contrast.
    //        No effect on mono stocks or pre-1950 stocks (strength 0).
    // PERF : 0 skips one blurred cross-channel pass (~8%); linear otherwise no.
    // AE   : PF_ADD_FLOAT_SLIDERX 0..2, 2 decimals.
    // -----------------------------------------------------------------------
    float couplerScale;

    // -----------------------------------------------------------------------
    // WHAT : Scanner/printer optics MTF, 50%-response frequency.
    //        0 = take from the resolved print stock (mtf_f50) -- recommended.
    // WHY  : Everything seen from film today passed a scanner; its aperture is
    //        a real low-pass AND the pre-sampling band-limit that keeps fine
    //        grain from aliasing. Also the DM-16 knob: one negative, many
    //        scan qualities (Steenbeck ~40, 2K DI ~80, 4K archival ~150).
    // RANGE: 0 (=from print stock), else 10.0 .. 300.0   DEFAULT: 0
    // UNITS: cycles/mm on the negative
    // IMAGE: global sharpness ceiling and grain rendering fidelity; too high
    //        with big grain = aliased "digital sand".
    // PERF : part of the existing frequency-domain pass -- free to change.
    // AE   : PF_ADD_FLOAT_SLIDERX 0..300, default 0, 0 decimals, with 0
    //        labelled "Auto (print stock)" in the param name.
    // -----------------------------------------------------------------------
    float scannerF50;

    // -----------------------------------------------------------------------
    // WHAT : Multiplies the stock's RMS channel misregistration (um).
    // WHY  : Three-strip's 26 um fringing is its signature; modern integral
    //        stock sits at 4-6 um. Exaggerating sells "old colour process".
    // RANGE: 0.0 .. 4.0       DEFAULT: 1.0
    // UNITS: multiplier of profile misregistration_um
    // IMAGE: coloured edge fringing, mostly R vs G/B. Mono stocks: none.
    // PERF : free (phase term in an existing FFT pass).
    // AE   : PF_ADD_FLOAT_SLIDERX 0..4, slider 0..2, 2 decimals.
    // -----------------------------------------------------------------------
    float misregScale;

    // -----------------------------------------------------------------------
    // WHAT : Veiling flare of the taking lens as a fraction of light scattered
    //        into a broad haze. NEGATIVE = use the stock's own default_flare
    //        (era-matched: 1930s uncoated glass 2-3%, modern ~0).
    // RANGE: -1.0 (=auto) or 0.0 .. 0.25      DEFAULT: -1.0 (auto)
    // UNITS: fraction of total light
    // IMAGE: lifts blacks and compresses contrast globally BEFORE the curve --
    //        the reason 1930s stocks must not render modern blacks. Applied
    //        after the curve it would just be a lift; here it breathes.
    // PERF : cheap (downsample-blur-upsample pyramid, ~3% of frame).
    // AE   : PF_ADD_FLOAT_SLIDERX -1..0.25 is ugly; better a PF_ADD_CHECKBOXX
    //        "Flare: Auto (era)" + PF_ADD_FLOAT_SLIDERX 0..0.25 enabled when
    //        unchecked (SUPERVISE). Struct keeps the single float: <0 = auto.
    // -----------------------------------------------------------------------
    float flare;

    // -----------------------------------------------------------------------
    // WHAT : Adds the PRINT stock's own fine grain on top of the negative's.
    // WHY  : A real print contributes its own (much finer) granularity;
    //        disabling emulates a direct high-grade scan of the negative.
    // RANGE: false/true       DEFAULT: true
    // IMAGE: subtle fine texture floor, most visible in mids of clean stocks.
    // PERF : off saves one small grain pass (~4%).
    // AE   : PF_ADD_CHECKBOXX.
    // -----------------------------------------------------------------------
    bool printGrain;

    // -----------------------------------------------------------------------
    // WHAT : Enables the additive-mosaic (reseau) path for stocks that have
    //        one (Dufaycolor). Ignored by all others.
    // WHY  : The mosaic needs >=3 px per grid pitch; below that the renderer
    //        auto-disables with a warning. This is the manual override for
    //        speed or for a "registered dye screen removed" restoration look.
    // RANGE: false/true       DEFAULT: true
    // IMAGE: Dufaycolor only: the visible RGB grid texture and its pastel
    //        colour reconstruction; off = plain B&W record through curves.
    // PERF : on Dufaycolor, the reseau adds mask build + reconstruction
    //        (~10%); irrelevant for the other 61 stocks.
    // AE   : PF_ADD_CHECKBOXX.
    // -----------------------------------------------------------------------
    bool reseau;

    // =======================================================================
    //  GROUP 4 -- FILM DAMAGE, PHYSICAL MODEL      (PF_ADD_TOPIC "Film Damage")
    //
    //  DESIGN CONTRACT (what "physics, not overlay" means here):
    //
    //  1. EVERY defect is an event on a physical film element -- an exposure
    //     obstruction, a removal of emulsion, a deformation of the base, or a
    //     lamp/optics event -- inserted at the pipeline stage where that
    //     element lives. It then inherits everything downstream FOR FREE:
    //     print gamma, print softness, dupe-generation blur, scanner MTF.
    //     Nothing is composited onto the finished image.
    //
    //  2. POLARITY IS DERIVED, NEVER AUTHORED. On a printed NEGATIVE:
    //       exposure-time obstruction (dust)  -> less density -> WHITE mark
    //       emulsion removal (scratch, blob)  -> clear base   -> BLACK mark
    //       base-side scratch                 -> refraction   -> soft grey line
    //     On REVERSAL all three invert automatically (profile.kind decides).
    //     Gate/projection-side defects block the lamp: ALWAYS dark, any stock.
    //
    //  3. DAMAGE DEPTH DECIDES COLOUR. Each emulsion-damage event samples a
    //     penetration depth into the layer stack. Monochrome: depth only sets
    //     strength -> white/black marks, exactly like real B&W. Colour tripack
    //     (top->bottom blue/green/red-sensitive, forming yellow/magenta/cyan
    //     dye): partial depth removes only the top dye(s), so the mark takes
    //     the colour of what REMAINS --
    //       through yellow only        -> blue-ish mark on the print
    //       through yellow + magenta   -> the classic green-cyan print scratch
    //       full depth to base         -> black (print) / white (reversal)
    //     Coloured blobs on colour stock and neutral blobs on B&W therefore
    //     need ZERO extra controls: both fall out of the stack model.
    //
    //  4. LIFETIME IS PHYSICAL. Dust lives 1 frame (falls off), hair 5-40
    //     frames (lodged in the gate, jitters, leaves), scratches PERSIST to
    //     the end of the roll (a stone in the gate does not heal). All
    //     stateless: derived from (damageSeed, birth frame) by the bounded
    //     birth-window scan -- scrub-safe, order-independent, re-render-exact.
    //
    //  5. COPY-CHAIN INTERACTION. `generations` (Group 1) is the master/2nd/
    //     3rd-copy control. Negative-side damage events sample WHICH element
    //     of the chain they live on: dirt born on the camera negative prints
    //     through every later stage (softest, most organic); dirt on the last
    //     dupe stays one generation sharp. Later copies thus accumulate MORE
    //     total dirt in distinct softness layers -- exactly how a real
    //     3rd-generation print looks, and why a flat overlay never does.
    //
    //  6. FULLY CONTROLLABLE, FULLY OFF-ABLE. damageEnable is a hard bypass;
    //     every class rate defaults to 0; damagePreset=Off zeroes the preset
    //     contribution. Factory default renders ZERO damage of any kind --
    //     the pristine profile physics only.
    //
    //  Rates: negative-side classes are per SECOND OF FILM (damage is baked
    //  into the film, so it follows layer time-stretch); gate-side classes
    //  are per second of COMPOSITION time (a projector does not slow down
    //  because the editor slowed the clip).
    // =======================================================================

    // -----------------------------------------------------------------------
    // WHAT : MASTER DAMAGE SWITCH. false = hard bypass of stages 3c, 9b, 15,
    //        16 -- no flicker, no defects, no weave, no gate dirt, regardless
    //        of every other control in this group.
    // WHY  : Explicit, greppable OFF. Also the A/B switch for judging the
    //        clean emulsion look against the damaged one.
    // RANGE: false/true       DEFAULT: false
    // IMAGE: false = pristine profile physics only.
    // PERF : false skips all four damage stages -- weave's full-frame
    //        resample included -- at zero per-pixel cost.
    // AE   : PF_ADD_CHECKBOXX at the top of the topic; SUPERVISE greys the
    //        whole group when off.
    // -----------------------------------------------------------------------
    bool damageEnable;

    // -----------------------------------------------------------------------
    // WHAT : Era preset scaling ALL class rates below at once.
    //        0=Off(x0)  1=Pristine lab print(x0.05)  2=Archival(x0.3)
    //        3=Worn theatrical(x1.0 -- sliders mean what they say)
    //        4=Junk / grindhouse(x3)
    // RANGE: 0 .. 4           DEFAULT: 3 (neutral multiplier; harmless while
    //        damageEnable=false and all rates=0)
    // IMAGE: global density of every class; character unchanged.
    // AE   : PF_ADD_POPUP.
    // -----------------------------------------------------------------------
    int32_t damagePreset;

    // ------------------------- NEGATIVE-SIDE CLASSES -----------------------
    // Stage 9b: DENSITY domain, before print. Soft, printed-through; polarity
    // and colour per contracts 2-3.
    // -----------------------------------------------------------------------

    // -----------------------------------------------------------------------
    // WHAT : DUST & DIRT present during exposure/processing. Poisson births,
    //        1-frame life (rarely 2-3 when a particle sticks), soft irregular
    //        silhouettes (noisy-radius discs -- never circles). Physical
    //        effect: light obstruction -> underexposure -> WHITE on printed
    //        negative, BLACK on reversal. Log-normal sizes, median dustSizeUm.
    // RANGE: rate 0..20 /s of film  DEFAULT 0;  size 5..200 um  DEFAULT 25
    // IMAGE: per-frame archival "snow". 25 um is ~2 px on 4K super35 -- real
    //        dust is SMALL; oversized dust is the loudest fake-damage tell.
    // PERF : sparse, <1%.
    // AE   : PF_ADD_FLOAT_SLIDERX x2 (rate 0..20 slider 0..5 1dec;
    //        size 5..200 0dec).
    // -----------------------------------------------------------------------
    float dustRateHz;
    float dustSizeUm;

    // -----------------------------------------------------------------------
    // WHAT : PROCESSING BLOBS -- chemical splashes, developer spots, air
    //        bells, drying marks. EMULSION events: depth-sampled (contract 3),
    //        so colour stock shows coloured spots (blue-ish, green-cyan,
    //        orange...) and B&W shows white/black, automatically. Life 1-3
    //        frames; the drying-mark variant drifts over tens of frames.
    // RANGE: 0 .. 10 events/s of film      DEFAULT: 0
    // IMAGE: soft-edged blotches 0.1-2 mm -- the old-newsreel blotch.
    // PERF : sparse, <1%.
    // AE   : PF_ADD_FLOAT_SLIDERX 0..10, slider 0..3, 1 decimal.
    // -----------------------------------------------------------------------
    float blobRateHz;

    // -----------------------------------------------------------------------
    // WHAT : LONGITUDINAL SCRATCHES from transport; sub-kind per event by
    //        scratchEmulsionBias:
    //          EMULSION-side -- material removed: depth-coloured/black line,
    //          near-vertical, sub-pixel width (AA or it crawls), slight
    //          per-frame wander, PERMANENT once born.
    //          BASE-side -- refractive groove, nothing removed: soft LOW-
    //          CONTRAST grey line; genuinely vanishes under wet-gate style
    //          diffuse scanning.
    // RANGE: rate 0..5 births/s of film DEFAULT 0;
    //        bias 0..1 (0=all base, 1=all emulsion) DEFAULT 0.5
    // IMAGE: the tramline. One birth every few seconds is already heavy --
    //        they never leave.
    // PERF : sparse, <1%.
    // AE   : PF_ADD_FLOAT_SLIDERX x2.
    // -----------------------------------------------------------------------
    float scratchRateHz;
    float scratchEmulsionBias;

    // -----------------------------------------------------------------------
    // WHAT : SPLICES -- a cement/tape joint passes the gate: 1-2 frames of
    //        horizontal band at the overlap, a vertical picture jump, a
    //        density step, often a dirt burst trapped at the joint. Stage 15
    //        reads the same event stream to kick the weave for one frame.
    // RANGE: 0 = off, else mean interval 2..600 s of film   DEFAULT: 0
    // UNITS: seconds of film between splices (Poisson about the mean)
    // IMAGE: the reel-change hiccup of assembled prints.
    // PERF : negligible.
    // AE   : PF_ADD_FLOAT_SLIDERX 0..600, 0 decimals, 0 labelled "Off".
    // -----------------------------------------------------------------------
    float spliceIntervalS;

    // -----------------------------------------------------------------------
    // WHAT : LUMA BLINK -- exposure flicker as a 2^x multiplier BEFORE the
    //        curve (stage 3c). 1/f spectrum from octave-spaced sinusoids,
    //        phases keyed on damageSeed ONLY (keying on frame whitens it).
    //        flickerColourSpread de-phases the three channels slightly for
    //        the colour breathing of badly processed colour stock.
    // WHY THIS IS PHYSICS: hand-crank speed variation, printer-lamp drift and
    //        development unevenness are all multiplicative on EXPOSURE.
    //        Before the curve, the shoulder compresses the bright swings and
    //        the image BREATHES; after the curve (the cheap route) it pumps
    //        like an opacity keyframe. The placement is the entire
    //        difference.
    // RANGE: amount 0..1 stops DEFAULT 0;  base 0.05..8 Hz film-time
    //        DEFAULT 0.5;  colourSpread 0..1 DEFAULT 0.15
    // IMAGE: 0.05-0.1 archival shimmer; 0.3+ silent-era pulse.
    // PERF : free (per-frame scalars, not per-pixel).
    // AE   : PF_ADD_FLOAT_SLIDERX x3.
    // -----------------------------------------------------------------------
    float flickerStops;
    float flickerBaseHz;
    float flickerColourSpread;

    // --------------------------- GATE-SIDE CLASSES -------------------------
    // Stages 15/16: AFTER everything film-side. Sharp (no downstream MTF),
    // always DARK, composition-time rates.
    // -----------------------------------------------------------------------

    // -----------------------------------------------------------------------
    // WHAT : GATE WEAVE -- the film moves in the gate; the image translates
    //        on a 1/f path, sub-pixel, Catmull-Rom resampled (bilinear would
    //        grind the grain down and partly undo stage 11). Runs BEFORE the
    //        hair/dirt below: the PICTURE weaves while the HAIR stays put --
    //        the single strongest projection tell.
    // RANGE: x,y 0..200 um on film DEFAULT 0,0 (y ~2x x on real projectors:
    //        perforation pitch dominates the vertical);
    //        corner 0.1..12 Hz comp-time DEFAULT 2.0
    // PERF : the ONLY expensive defect -- one full-frame separable resample
    //        (~10-15% of frame). Zero amplitude skips it entirely.
    // AE   : PF_ADD_FLOAT_SLIDERX x3 (x,y 0..200 slider 0..60 0dec).
    // -----------------------------------------------------------------------
    float weaveAmpXUm;
    float weaveAmpYUm;
    float weaveHzCorner;

    // -----------------------------------------------------------------------
    // WHAT : HAIR IN THE GATE. A fibre lodges at the aperture: enters from a
    //        frame edge, lives 5-40 frames, jitters a few px/frame, leaves.
    //        3-4 point Catmull-Rom spline with tapering width, AA-rasterised
    //        -- a straight line never reads as hair. Own RNG stream, so it
    //        never correlates with film-side damage.
    // RANGE: 0 .. 4 births/s (comp time)   DEFAULT: 0
    // IMAGE: the classic dancing hair -- razor sharp, pure dark.
    // PERF : sparse, <1%.
    // AE   : PF_ADD_FLOAT_SLIDERX 0..4, 2 decimals.
    // -----------------------------------------------------------------------
    float hairRateHz;

    // -----------------------------------------------------------------------
    // WHAT : GATE DUST & PROJECTION SCRATCHES -- platen dirt and scratches
    //        inflicted at projection. Same geometry generators as the
    //        film-side cousins, inserted at stage 16 instead: SHARP and DARK
    //        on any stock -- exactly what visually separates them from 9b.
    // RANGE: 0 .. 20 events/s (comp time)  DEFAULT: 0
    // PERF : sparse, <1%.
    // AE   : PF_ADD_FLOAT_SLIDERX 0..20, slider 0..5, 1 decimal.
    // -----------------------------------------------------------------------
    float gateDefectRateHz;

    // -----------------------------------------------------------------------
    // WHAT : Global strength/opacity multiplier for ALL damage classes
    //        (scales each event's density delta, not the count).
    // RANGE: 0.0 .. 2.0       DEFAULT: 1.0
    // AE   : PF_ADD_FLOAT_SLIDERX 0..2, 2 decimals.
    // -----------------------------------------------------------------------
    float damageStrength;

    // -----------------------------------------------------------------------
    // WHAT : Master seed for ALL stochastic content: grain fields, flicker
    //        phases, damage births, depth samples, weave path. Every random
    //        value is a pure function of (damageSeed, frameIdx, stageId,
    //        ordinal) via a counter-based RNG -- no state, so Premiere's
    //        out-of-order, speculative, multi-instance rendering reproduces
    //        exactly, and re-renders never re-roll the damage.
    // RANGE: 0 .. 2^31-1      DEFAULT: 12345 (matches Python reference)
    // IMAGE: layout changes; statistics do not. New seed = new roll of film.
    // AE   : PF_ADD_SLIDER (integer) + optional PF_ADD_BUTTON "New Seed".
    // -----------------------------------------------------------------------
    int32_t damageSeed;

    // =======================================================================
    //  GROUP 6 -- OUTPUT                              (PF_ADD_TOPIC "Output")
    // =======================================================================

    // -----------------------------------------------------------------------
    // WHAT : TPDF dither before the caller's bit-depth reduction (stage 17).
    // WHY  : After all this physical modelling, 8/10-bit banding in a sky
    //        gradient is what a viewer actually notices first.
    // RANGE: false/true       DEFAULT: true
    // IMAGE: invisible except as the ABSENCE of banding; adds ~0.5 LSB noise.
    // PERF : free (fused into the final clamp pass).
    // AE   : PF_ADD_CHECKBOXX. Hide entirely at 32-bpc project depth
    //        (PF_OutFlag2_SUPPORTS_SMART_RENDER path knows the depth).
    // -----------------------------------------------------------------------
    bool ditherOutput;
};

constexpr size_t AlgoControlsSize = sizeof(AlgoControls);

// Factory: defaults exactly as documented above (and identical to the Python
// reference RenderSettings where a counterpart exists), so a fresh effect
// instance renders any stock authentically with zero user configuration.
AlgoControls getAlgoControlsDefault (void);


PF_Err
SetupControlElements
(
    const PF_InData*  in_data,
          PF_OutData* out_data
);

const AlgoControls
GetControlElements
(
    PF_ParamDef* params[]
);


