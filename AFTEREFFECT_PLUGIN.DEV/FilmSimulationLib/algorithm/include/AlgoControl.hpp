#pragma once

#include <cstdint>
#include <cmath>     // std::sqrt (AlgoGrainTemporalScale)

// eFILM_PROFILE: the generated profile index enumeration. Every algorithm
// parameter reaches the engine through this structure, and the film selection is
// an algorithm parameter, so it lives here rather than in the call signature.
#include "film_enum.hpp"
#include "AlgoControlEnums.hpp"

/**
 * @file AlgoControl.hpp
 * @brief User-facing controls for the film-simulation algorithm.
 *
 * ===========================================================================
 *  DOCUMENTATION STANDARD - READ THIS BEFORE READING ANY FIELD
 * ===========================================================================
 *
 * Every control below carries a fourteen-item block. The items are fixed, in
 * this order, and none is omitted:
 *
 *    1  NAME           the field identifier
 *    2  TYPE           the declared C++ type
 *    3  AE CONTROL     Adobe Effect Control Panel representation
 *    4  UNIT           physical / engineering unit, or "dimensionless" with the
 *                      quantity it scales named
 *    5  MIN            minimum, and whether the engine enforces it
 *    6  MAX            maximum, and whether the engine enforces it
 *    7  DEFAULT        the value getAlgoControlsDefault() / getFilmDamageDefault()
 *                      actually assigns, cited to AlgoControl.cpp
 *    8  STEP           increment, including the effective UI increment
 *    9  PURPOSE        what the control is for
 *   10  OUTPUT EFFECT  what it does to the rendered image
 *   11  STAGES         which pipeline stage numbers it reaches
 *   12  INTERACTIONS   dependencies on, and interactions with, other controls
 *   13  SCALAR/AVX2    any difference between the two instruction-set paths
 *   14  FULL/LITE      participation in the Full path, the Lite path, or both
 *
 * ---------------------------------------------------------------------------
 *  THREE CONVENTIONS THAT APPLY TO EVERY BLOCK
 * ---------------------------------------------------------------------------
 *
 * DOCUMENTED RANGE vs ENFORCED RANGE. Items 5 and 6 distinguish the two, and
 * the distinction is not cosmetic. Until 2026-10-02 NO numeric control in
 * this structure had an enforced upper bound: stages floor their inputs with
 * MAX_VALUE(x, 0) and nothing else, and the only ceilings cap a *product*
 * rather than a control (ALGO_DEFOCUS_MAX_LOSS at Algo_06_Sim.cpp,
 * ALGO_DUST_DENSITY_MAX at Algo_09_Sim.cpp). ⚠ SINCE 2026-10-02 the entry
 * point enforces the Min/Max of AlgoControlEnums.hpp: AlgorithmMain passes
 * the host's structure through AlgoControlsClamped (declared at the end of
 * this file) before any stage reads it. So a "MAX ... advisory" line below
 * is now the clamp the engine applies; the host panel should still enforce
 * it, because a clamped value is silently a different value from the one the
 * user typed.
 *
 * STEP values marked [proposed]. There is no UI layer in this source tree, so
 * no increment is implementation-backed. Rather than leave item 8 blank or
 * invent a number silently, each [proposed] step is derived from the
 * parameter's own numerical sensitivity and the rationale is stated. These are
 * recommendations for the panel author and must be confirmed when the Adobe
 * panel is built - they are not currently read by anything.
 *
 * ITEM 14 IS PENDING THROUGHOUT. Full/Lite participation cannot be documented
 * against this tree because the modes do not exist in it: there is no
 * AlgoControls::simMode field and no QualityPolicy struct. Every item 14 below
 * records the *expected* participation with its basis, marked PENDING, and must
 * be revisited when the driver restructure and the Lite preset land. Nothing in
 * item 14 is asserted as implemented.
 *
 * ---------------------------------------------------------------------------
 *  TWO GROUPS, AND THE DISTINCTION IS LOAD-BEARING
 * ---------------------------------------------------------------------------
 *
 *   AlgoControls        25 fields. It mirrors film_sim.RenderSettings field for
 *                       field with one deliberate exception (the damage group,
 *                       which the reference model does not have), so a C++
 *                       render with getAlgoControlsDefault() is directly
 *                       comparable against the Python reference. That
 *                       comparability is what let Algo 02 be verified to 1e-15;
 *                       keep the two in step or the reference stops being a
 *                       reference.
 *
 *   FilmDamage          17 fields, a named sub-struct of AlgoControls gated by
 *                       AlgoControls::filmDamageEnabled. ALL SEVENTEEN are
 *                       consumed since 2026-10-10 - by stages 03c, 09b, 15
 *                       and 16 and by the storage resolver. (Earlier
 *                       revisions said nine, then eleven; both were true when
 *                       written.)
 *
 * ---------------------------------------------------------------------------
 *  WHAT IS *NOT* HERE
 * ---------------------------------------------------------------------------
 *   Film properties. Those live in FilmProfile (see film_profiles.hpp) and are
 *   data, not controls. A control never replaces a profile number; it scales or
 *   overrides it, and every such field says so explicitly below.
 *
 *   Two controls named in the project requirements do not exist in this
 *   structure and are NOT documented below, because documenting an absent field
 *   would be an invention:
 *
 *     simMode         the Preview/Full selector. Specified, not implemented.
 *                     It must be appended LAST when it lands, together with
 *                     QualityPolicy, so the structure layout changes once
 *                     rather than twice.
 *
 *     CCT / Duv       a two-axis colour control with coarse/fine increments is
 *                     referenced in the requirements as established. No such
 *                     control exists here. The only related field is
 *                     sceneKelvin - a single correlated-colour-temperature
 *                     axis with no Duv component and no coarse/fine split. The
 *                     specification is not recoverable from this tree and must
 *                     come from the project owner before anything is written.
 *
 * ---------------------------------------------------------------------------
 *  SENTINEL CONVENTION
 * ---------------------------------------------------------------------------
 *   Two fields - flare and vignette - use "< 0 means use the stock's own
 *   default". The test is strictly less-than-zero in both cases, so 0.0 means
 *   "genuinely none" and -1.0 means "whatever this stock would have done".
 *   Passing 0.0 where -1.0 was meant silently disables an effect rather than
 *   restoring it.
 *
 * ---------------------------------------------------------------------------
 *  LAYOUT DISCIPLINE
 * ---------------------------------------------------------------------------
 *   Changing the field order, or removing a field, changes this structure's
 *   layout and breaks any caller that serialises it. New fields are APPENDED
 *   LAST, following the same discipline as the database schema. This revision
 *   changes comments only: the field set, the field order and the layout are
 *   byte-for-byte what they were.
 */


// ===========================================================================
// Damage controls
// ===========================================================================
/**
 * Physical film damage, gated by AlgoControls::filmDamageEnabled.
 *
 * CONSUMPTION STATUS, MEASURED AGAINST THIS TREE
 *   Consumed (17 of 17, since 2026-10-10):
 *                 damageStrength, damageSeed -> every damage site below
 *                 flickerStops               -> stage 03c
 *                 dustLevel, debrisLevel, fibreLevel, dirtClumping,
 *                 scratchTransport, scratchHandling,
 *                 processingQuality, dryingMarks, colourVeil -> stage 09b
 *                 weaveAmount                -> stage 15
 *                 gateDirt, damageEvents, scannerArtifacts -> stage 16
 *                 storageSeverity            -> frame setup, storage resolver
 *   Unconsumed: none. The six that had no reader until 2026-10-10 kept their
 *   place in the layout throughout, so presets serialised while they were
 *   inert still deserialise - and now do something, exactly as happened to
 *   the two scratch fields below. Their amplitudes are DEFINITIONS, not
 *   measurements; each item 10 says so.
 *
 *   ⚠ THE TWO SCRATCH FIELDS MOVED FROM UNCONSUMED TO CONSUMED, and the layout
 *   did not change with them - they were always here, in this order, and only
 *   their behaviour is new. Anything that serialised this structure while they
 *   were inert still deserialises, and the presets it wrote now do something.
 *
 * TWO FAMILIES, DELIBERATELY SPLIT FROM FILM PROPERTIES
 *   These are POST-HOC events on a developed strip and are emulsion
 *   independent: a dust particle on VISION3 looks like a dust particle on
 *   Svema. That is why they are controls.
 *
 *   What does NOT belong here, and is profile data instead: dye fade (per dye
 *   set), base yellowing and shrinkage (per base material), scratch COLOUR
 *   (depth determines which dye layers survive, so it needs the tripack), and
 *   blob polarity (white on a print, dark on a negative, inverted again on
 *   reversal). Those live in AgingSpec and CoatingSpec.
 *
 * PHYSICAL UNITS -- NOT RATES PER SECOND
 *   Every geometric and areal quantity in this group is PHYSICAL: areal density
 *   per square millimetre of film, size in micrometres on the film. Never per
 *   second, never per frame, never in pixels, never as a fraction of the frame.
 *
 *   This is rule R6 of the defect model requirements, and it is the reason one
 *   set of numbers serves 35 mm still, Super 35, 16 mm and Regular 8 without
 *   re-tuning. A 25 um dust particle is 25 um on every gauge; what changes is
 *   only how much of the picture it covers, and that is DERIVED from the format
 *   at render time, never authored.
 *
 *   Frame rate still enters, but only for the TEMPORAL classes -- how long a
 *   defect persists -- never for how much of it there is.
 *
 * WHAT IS A CONTROL AND WHAT IS A CONSTANT
 *   These seventeen fields are the whole user-facing surface. The defect model
 *   specifies roughly two hundred parameters; the other ~185 are MEASURED FACTS
 *   about film, not choices, and live as named constants in the stage headers
 *   with their measured value and evidence grade in the comment.
 *
 *   Deliberately NOT here: the dust size exponent (gamma = 2.6), the clumping
 *   field spectral slope (beta = 1.0), scratch width (26 um), scratch
 *   straightness (0.98), the 3.5:1 longitudinal orientation bias, the median
 *   3.5 per cent contrast amplitude, the weave X:Y ratio and its 0.8 Hz corner,
 *   the T1/T2/T3 population shares. Exposing them would be 200 sliders nobody
 *   touches.
 *
 *   The four scratch figures in that list now live where the paragraph says
 *   they should - ALGO_SCRATCH_WIDTH_UM, ALGO_SCRATCH_STRAIGHTNESS,
 *   ALGO_SCRATCH_ORIENT_RATIO and ALGO_SCRATCH_CONTRAST_MEDIAN in
 *   AlgoNegativeDefects.hpp, each with its value and where it came from.
 *
 *   ⚠ 2026-10-20 (owner-approved stochastic defect model): the single 26 um
 *   width, which no document in the tree traces to a measurement, is replaced
 *   by per-population log-normal widths (transport 45 um, handling 38 um,
 *   sigma_ln 0.22) read from the owner's reference collage, provisional; the
 *   contrast keeps its 3.5 per cent median for handling marks and gains a
 *   dispersion and a transport median. A defectOptics switch (bit 51) was
 *   added at the end of FilmDamage, which now has EIGHTEEN fields; "seventeen"
 *   above is the count of the original surface.
 *
 * ZERO DISABLES, AND IT COSTS NOTHING
 *   Every scale below follows the engine's existing convention: 0 switches the
 *   class off completely, and the corresponding generator is not run and its
 *   buffer not written. Same rule as grainScale, halationScale, coatingScale.
 *
 * THE PROFILE SUPPLIES THE ERA, THE CONTROL SUPPLIES THE INTENT - INTENDED,
 * NOT YET IN FORCE
 *   AgingSpec in each film profile carries dust_area_ppm, mottle_amplitude,
 *   scratch_rate_base_per_m, dye_fade_c/m/y and dmin_lift, and the design is
 *   that the controls below MULTIPLY those. That cannot work today: every stock
 *   in the generated database ships AgingSpec entirely zero, documented as
 *   "fresh", so multiplying would silence the defect layer on every stock. The
 *   levels are therefore ABSOLUTE for now - dustLevel 1.0 means the measured
 *   density whatever the stock. See AlgoControl.cpp for the full rationale and
 *   for why the era term should re-enter as ADDITIVE rather than multiplicative.
 *
 * STATELESSNESS REQUIREMENT
 *   Every generator is a pure function of (damageSeed, frameIndex, stage salt,
 *   ordinal) via a counter-based RNG, with a bounded birth-frame scan for
 *   objects that persist. Any frame must be renderable alone, out of order, on
 *   any thread -- the same rule the coating field already follows.
 *
 *   The five-level seed hierarchy the requirements demand needs no new fields:
 *   L0 stock is the profile index, L1 roll is damageSeed, L2 segment derives
 *   from frameIndex over the segment length, L3 frame is frameIndex, and L4 is
 *   the per-defect ordinal.
 *
 *   NOTE on the salt: the value each stage XORs with damageSeed is a
 *   compile-time per-stage constant (ALGO_SALT_NEG_DEFECTS, ALGO_SALT_WEAVE,
 *   ALGO_SALT_GATE_DEFECTS in AlgorithmMain.cpp), NOT AlgoControls::seed.
 *   AlgoControls::seed never enters a damage stream.
 */
struct FilmDamage
{
    // =======================================================================
    //  MASTER
    // =======================================================================

    /**
     *  1  NAME           damageStrength
     *  2  TYPE           double
     *  3  AE CONTROL     slider
     *  4  UNIT           dimensionless multiplier applied to every consumed
     *                    class level below; carries no unit of its own
     *  5  MIN            0.0, ENFORCED - MAX_VALUE floor at the three reading
     *                    stages (09b, 15, 16). Stages 09b and 16 additionally
     *                    return immediately when it is <= 0
     *  6  MAX            4.0 advisory - NOT ENFORCED anywhere
     *  7  DEFAULT        1.0   (AlgoControl.cpp, getFilmDamageDefault)
     *  8  STEP           0.01 [proposed] - the levels it multiplies are
     *                    themselves proposed at 0.01, and the master must not
     *                    be coarser than the thing it scales
     *  9  PURPOSE        One control that dials the whole damage look up or
     *                    down without disturbing the balance between classes.
     * 10  OUTPUT EFFECT  Multiplies dust, debris, fibre, gate-dirt and weave
     *                    amplitude simultaneously. At 0 the entire damage
     *                    subsystem is skipped for one branch per frame.
     *                    NOTE it does NOT multiply dirtClumping - clumping is a
     *                    distribution shape, not an amount.
     * 11  STAGES         09b, 15, 16
     * 12  INTERACTIONS   Subordinate to filmDamageEnabled, which is tested
     *                    first. Multiplies dustLevel, debrisLevel, fibreLevel,
     *                    gateDirt, damageEvents and weaveAmount. Does not touch
     *                    dirtClumping or damageSeed.
     * 13  SCALAR/AVX2    Same semantics. The level derivation is HighPrecType
     *                    (double) in BOTH builds; only the per-pixel
     *                    rasterisers differ.
     * 14  FULL/LITE      PENDING. Expected Full only - stage 09b is on the Lite
     *                    drop list and stages 15/16 measure ~15 per cent of a
     *                    monochrome frame between them. Confirm at Tasks 7/8.
     */
    double damageStrength;

    /**
     *  1  NAME           damageSeed
     *  2  TYPE           int32_t
     *  3  AE CONTROL     integer field + randomise button
     *  4  UNIT           dimensionless seed integer; XOR operand only, never
     *                    used in physical arithmetic
     *  5  MIN            none - the whole int32_t range is valid. Reinterpreted
     *                    as uint32_t at every use, so negatives wrap, which is
     *                    well defined and harmless to the mixer
     *  6  MAX            none, same reason
     *  7  DEFAULT        20250803   (AlgoControl.cpp, getFilmDamageDefault)
     *  8  STEP           1 [proposed] - it is an identifier; the panel should
     *                    offer a randomise button rather than a drag
     *  9  PURPOSE        Seeds every damage generator. This is also the ROLL
     *                    seed (level L1): holding it fixed while changing
     *                    frameIndex gives different frames OF THE SAME ROLL -
     *                    same stains, same scratches, different dust.
     * 10  OUTPUT EFFECT  Changes which realisation of the damage is drawn.
     *                    Changes nothing about how much there is.
     * 11  STAGES         09b, 15, 16
     * 12  INTERACTIONS   Deliberately independent of AlgoControls::seed, so
     *                    re-rolling the grain does not re-roll the dirt and
     *                    vice versa. Verified: the damage stages never read
     *                    AlgoControls::seed. Combined with frameIndex and a
     *                    per-stage compile-time salt.
     * 13  SCALAR/AVX2    Same semantics, and the RNG header is byte-identical
     *                    between the trees, so the drawn integer sequences are
     *                    bit-identical in both builds.
     * 14  FULL/LITE      PENDING. Both, wherever a damage stage runs at all.
     */
    int32_t damageSeed;

    // =======================================================================
    //  PARTICULATE -- three separate classes
    //
    //  Split rather than merged because they differ by three orders of magnitude
    //  in rate, fifty times in size, and need three different render primitives:
    //
    //    dust    0.5-3.4 /mm2    18-107 um            soft-edged blob
    //    debris  0.0023-0.0069   0.3-1.5 mm           opaque polygon, 5-9 sides
    //    fibres  0.0001-0.0035   20-80 um x 1-20 mm   stroked spline with a curl
    //
    //  All three are placed by the same clumped (Cox) spatial process, because
    //  they share a deposition mechanism.
    // =======================================================================

    /**
     *  1  NAME           dustLevel
     *  2  TYPE           double
     *  3  AE CONTROL     slider
     *  4  UNIT           dimensionless multiplier on an AREAL DENSITY in
     *                    particles per square millimetre of film. It multiplies
     *                    ALGO_DUST_DENSITY_PER_MM2 = 2.0 /mm2, and the product
     *                    is the Poisson/Cox intensity over a film-millimetre
     *                    cell grid
     *  5  MIN            0.0, ENFORCED (MAX_VALUE floor at stage 09b)
     *  6  MAX            4.0 advisory - NOT ENFORCED on the control. There IS a
     *                    physical saturation downstream: the product
     *                    dustLevel * damageStrength saturates at 7.0 through
     *                    ALGO_DUST_DENSITY_MAX = 14.0, after which the embedded
     *                    fraction 0.6 gives a hard ceiling of 8.4 particles/mm2.
     *                    Values above that change nothing visible.
     *  7  DEFAULT        1.0   (AlgoControl.cpp, getFilmDamageDefault)
     *  8  STEP           0.01 [proposed] - one hundredth of the measured
     *                    central figure; at 512 px on super35 a step of 0.01
     *                    moves the expected particle count by well under one
     *  9  PURPOSE        Fine embedded particulate. 1.0 is not a slider
     *                    position that looked right - it is the measured areal
     *                    density of embedded particles on amateur
     *                    hand-processed film.
     * 10  OUTPUT EFFECT  Places soft-edged dark blobs, 18-107 um on the film,
     *                    at the scaled areal density. Film-locked: a particle
     *                    appears for exactly one frame and vanishes.
     * 11  STAGES         09b
     * 12  INTERACTIONS   Multiplied by damageStrength. Distributed by
     *                    dirtClumping. Gated by filmDamageEnabled. Shares the
     *                    class early-out with debrisLevel and fibreLevel: if
     *                    all three are <= 0 the stage returns.
     * 13  SCALAR/AVX2    Same semantics; the rate derivation block is textually
     *                    identical between the trees and stays in double in
     *                    both. Only the blob rasteriser is vectorised.
     * 14  FULL/LITE      PENDING. Expected Full only - stage 09b is on the Lite
     *                    drop list.
     */
    double dustLevel;

    /**
     *  1  NAME           debrisLevel
     *  2  TYPE           double
     *  3  AE CONTROL     slider
     *  4  UNIT           dimensionless multiplier on an areal density in
     *                    particles per square millimetre; multiplies
     *                    ALGO_DEBRIS_DENSITY_PER_MM2 = 0.0046 /mm2
     *  5  MIN            0.0, ENFORCED (MAX_VALUE floor at stage 09b)
     *  6  MAX            4.0 advisory - NOT ENFORCED, and unlike dust there is
     *                    no downstream saturation either: the Poisson intensity
     *                    scales without limit
     *  7  DEFAULT        1.0   (AlgoControl.cpp, getFilmDamageDefault)
     *  8  STEP           0.01 [proposed] - matches dustLevel so the three
     *                    particulate sliders behave alike under the same drag
     *  9  PURPOSE        Coarse opaque lint and chemistry fragments,
     *                    0.3-1.5 mm. Rare - a few per 35 mm still frame - but
     *                    individually conspicuous and fully opaque.
     * 10  OUTPUT EFFECT  Places opaque 5-9 sided polygons. Because they are
     *                    fully opaque rather than attenuating, a single one is
     *                    far more visible than its area suggests.
     * 11  STAGES         09b
     * 12  INTERACTIONS   Multiplied by damageStrength. Distributed by
     *                    dirtClumping. Shares the class early-out with
     *                    dustLevel and fibreLevel.
     * 13  SCALAR/AVX2    Same semantics; derivation identical and in double in
     *                    both builds.
     * 14  FULL/LITE      PENDING. Expected Full only (stage 09b).
     */
    double debrisLevel;

    /**
     *  1  NAME           fibreLevel
     *  2  TYPE           double
     *  3  AE CONTROL     slider
     *  4  UNIT           dimensionless multiplier on an areal density in
     *                    fibres per square millimetre; multiplies
     *                    ALGO_FIBRE_DENSITY_PER_MM2 = 0.0012 /mm2
     *  5  MIN            0.0, ENFORCED (MAX_VALUE floor at stage 09b)
     *  6  MAX            4.0 advisory - NOT ENFORCED, no downstream saturation
     *  7  DEFAULT        1.0   (AlgoControl.cpp, getFilmDamageDefault)
     *  8  STEP           0.01 [proposed] - matches the other two particulate
     *                    levels
     *  9  PURPOSE        Hair and textile fibres. Distinguished from a scratch
     *                    by near-constant width, free ends and a curl; a fibre
     *                    lies ON the film, a scratch is IN it.
     * 10  OUTPUT EFFECT  Places stroked splines, 20-80 um wide by 1-20 mm long,
     *                    with a curl.
     * 11  STAGES         09b
     * 12  INTERACTIONS   Multiplied by damageStrength. Distributed by
     *                    dirtClumping. Shares the class early-out with
     *                    dustLevel and debrisLevel.
     * 13  SCALAR/AVX2    Same semantics; derivation identical and in double in
     *                    both builds.
     * 14  FULL/LITE      PENDING. Expected Full only (stage 09b).
     */
    double fibreLevel;

    /**
     *  1  NAME           dirtClumping
     *  2  TYPE           double
     *  3  AE CONTROL     slider
     *  4  UNIT           dimensionless multiplier on a COEFFICIENT OF
     *                    VARIATION. It scales ALGO_DEFECT_FIELD_CV = 0.88, the
     *                    measured CV of the local particle rate; the scaled CV
     *                    becomes the sigma of a log-normal Cox intensity field
     *  5  MIN            0.0, ENFORCED. At exactly 0 the Cox field is bypassed
     *                    and the process degenerates to uniform Poisson
     *  6  MAX            2.0 advisory - NOT ENFORCED. The derived sigma grows
     *                    without limit, so large values produce a few extreme
     *                    patches and empty space elsewhere
     *  7  DEFAULT        1.0   (AlgoControl.cpp, getFilmDamageDefault)
     *  8  STEP           0.01 [proposed] - the CV it scales is a measured 0.88,
     *                    so 0.01 is finer than the measurement's own precision
     *  9  PURPOSE        How much the LOCAL particulate rate varies across the
     *                    frame. 0 gives uniform Poisson scatter, which is the
     *                    single most recognisable failure of existing
     *                    film-emulation products - real dirt arrives in
     *                    patches, some regions carrying five times the average.
     *                    1 is the measured film behaviour.
     * 10  OUTPUT EFFECT  Redistributes the SAME expected number of particles
     *                    into patches. It adds no dirt and removes none.
     * 11  STAGES         09b
     * 12  INTERACTIONS   Modifies the placement of all three particulate
     *                    classes. NOT multiplied by damageStrength - it is a
     *                    shape, not an amount - though damageStrength = 0 still
     *                    short-circuits the stage before it is read. Does
     *                    nothing at all while all three levels are 0.
     * 13  SCALAR/AVX2    Same semantics. The Cox intensity helper lives in a
     *                    translation unit the AVX2 build links from the scalar
     *                    tree, so it is literally the same code.
     * 14  FULL/LITE      PENDING. Expected Full only (stage 09b).
     */
    double dirtClumping;

    // =======================================================================
    //  SCRATCHES -- two classes with opposite geometry and opposite temporal
    //  behaviour, which is why they are not one control
    // =======================================================================

    /**
     *  1  NAME           scratchTransport
     *  2  TYPE           double
     *  3  AE CONTROL     slider ("Transport Scratch Rate")
     *  4  UNIT           NEW TRAMLINE EVENTS PER MINUTE OF SCREEN TIME
     *                    (2026-10-20, owner-approved; was "expected number
     *                    running at once"). Time is frameIndex / frameRate,
     *                    so the event rate is the same at 24, 25, 30 or 60 fps.
     *                    Events are a RENEWAL process: the gap to the next
     *                    event is drawn independently, uniform on
     *                    [0.4 m, 1.6 m] with m = 60 / rate seconds. At the
     *                    minimum non-zero setting (ScratchTransportFloor =
     *                    4.8 / min, m = 12.5 s) every gap lies in 5 - 20 s
     *                    (owner's specification 2026-10-20). A running
     *                    scratch is ONE event for its whole lifetime
     *                    (scratchLifetime); the number visible at once is
     *                    derived, about rate x lifetime / 60.
     *  5  MIN            0.0 = no transport scratches. Any value in
     *                    (0, 4.8) is raised to 4.8 at stage 09b.
     *  6  MAX            120.0 (gap 0.2 - 0.8 s; with the default lifetime one
     *                    or more tramlines are on every frame)
     *  7  DEFAULT        4.8   (ScratchTransportDef)
     *  8  STEP           0.1
     *  9  PURPOSE        Longitudinal transport scratches - long, straight,
     *                    parallel to film travel, continuing across frame
     *                    boundaries. The defining motion-picture defect
     *                    ("rain", "tramlines"), and machine-fixed: it holds a
     *                    fixed position on screen while the image moves past
     *                    it.
     * 10  OUTPUT EFFECT  (2026-10-20: log-normal width, median 45 um, with
     *                    wander, along-run width / contrast variation and a
     *                    faint companion track on half of them; see
     *                    AlgoNegativeDefects.hpp.)
     *                    Draws straight 26 um strokes locked to the transport
     *                    axis at a fixed position across the web, each present
     *                    for a run of frames whose mean is the stock's own
     *                    TemporalSpec::scratch_persistence_frames. Orientation
     *                    is horizontal on a still frame and vertical on every
     *                    common cine format, because film travels along the
     *                    long axis of a still and the short axis of a cine
     *                    frame - derived from the format by AlgoFilmCoord,
     *                    never authored. Each mark is DARK OR LIGHT on the
     *                    negative depending on whether it cuts the emulsion or
     *                    burnishes the base.
     *                    ⚠ INERT ON SHEET AND PACK FORMATS. They report a frame
     *                    pitch of zero and no transport axis, so there is no
     *                    web for an abrader to run against; the handling class
     *                    still applies to them.
     * 11  STAGES         09b
     * 12  INTERACTIONS   Multiplied by damageStrength (then floored at 4.8 if
     *                    non-zero). Lifetime: scratchLifetime (0 = the stock's
     *                    TemporalSpec::scratch_persistence_frames at 24 fps).
     *                    Contrast: defectVisibility. Along-run temporal change:
     *                    defectVariability. Needs the host frameRate. NOT
     *                    distributed by dirtClumping - the clumping field is a
     *                    property of the FILM and a transport scratch belongs to
     *                    the MACHINE.
     * 13  SCALAR/AVX2    Same semantics, and stronger than that: the whole
     *                    class is HighPrecType in both trees and its source is
     *                    character-for-character identical between them, so the
     *                    two builds differ only by the float32 store.
     * 14  FULL/LITE      PENDING. Expected Full only (stage 09b).
     */
    double scratchTransport;

    /**
     *  1  NAME           scratchHandling
     *  2  TYPE           double
     *  3  AE CONTROL     slider
     *  4  UNIT           dimensionless multiplier on an AREAL DENSITY in marks
     *                    per square millimetre of film; it multiplies
     *                    ALGO_SCRATCH_HANDLING_PER_MM2 = 0.01 /mm2, which is
     *                    about 4.6 marks on a super35 frame and 8.6 on a 35 mm
     *                    still frame.
     *                    ⚠ THE 0.01 IS A DEFINITIONAL ANCHOR, NOT A
     *                    MEASUREMENT, for the same reason scratchTransport's
     *                    is: AgingSpec.scratch_rate_base_per_m is zero on every
     *                    stock, so there is no measured rate to scale.
     *  5  MIN            0.0, ENFORCED (MAX_VALUE floor at stage 09b)
     *  6  MAX            4.0 advisory - NOT ENFORCED, no downstream saturation
     *  7  DEFAULT        0.30   (AlgoControl.cpp, getFilmDamageDefault)
     *  8  STEP           0.01 [proposed]
     *  9  PURPOSE        Random handling scratches - short, curved, 0.3-4 mm,
     *                    individually very faint but numerous. Locked to the
     *                    film, not to the machine, so they travel with the web
     *                    exactly as the particulate classes do and have no
     *                    persistence: a handling scratch is a single event and
     *                    then a permanent part of the emulsion.
     * 10  OUTPUT EFFECT  (2026-10-20: log-normal width median 38 um and length
     *                    median 0.5 mm on 0.1-4 mm, and BURSTS of 2-4 parallel
     *                    marks - the gap noted below is closed with the
     *                    reference collage's figures, provisional.)
     *                    Places stroked 26 um marks of 0.3-4 mm, curved to the
     *                    measured 0.98 chord-over-arc straightness that
     *                    separates a scratch from a fibre, oriented with the
     *                    measured 3.5:1 preference for the transport axis, and
     *                    dark or light according to cut against burnish.
     *                    ⚠ NOT GENERATED IN BURSTS, which the description above
     *                    calls for and which is a known gap: a burst needs a
     *                    size and an inter-mark spacing, and nothing in the
     *                    tree measures either. The clumping field puts marks in
     *                    patches, which is part of the effect but not the
     *                    parallelism.
     * 11  STAGES         09b
     * 12  INTERACTIONS   Multiplied by damageStrength. Distributed by
     *                    dirtClumping, unlike the transport class - a wipe
     *                    damages a patch of FILM, so it belongs in the same
     *                    Cox process as the particulate. Shares the class
     *                    early-out with the other four levels.
     * 13  SCALAR/AVX2    Same semantics; derivation and rasterisation are
     *                    HighPrecType in both builds.
     * 14  FULL/LITE      PENDING. Expected Full only (stage 09b).
     */
    double scratchHandling;

    // =======================================================================
    //  PROCESSING AND DRYING
    // =======================================================================

    /**
     *  1  NAME           processingQuality
     *  2  TYPE           double
     *  3  AE CONTROL     slider
     *  4  UNIT           dimensionless 0..1 quality axis (0 a professional
     *                    continuous machine, 1 a hand tank in a bathroom)
     *  5  MIN            0.0   (ProcessingQualityMin, clamped on entry)
     *  6  MAX            1.0   (ProcessingQualityMax, clamped on entry)
     *  7  DEFAULT        0.30   (AlgoControl.cpp, getFilmDamageDefault)
     *  8  STEP           0.01
     *  9  PURPOSE        CONSUMED SINCE 2026-10-10. Inverse quality of the
     *                    development bath, as two development non-uniformities
     *                    that express agitation quality: low-frequency MOTTLE
     *                    (two lattice octaves, 3 mm and 0.75 mm of film) and AIR
     *                    BELLS (discs where a bubble kept developer off).
     *                    Surge marks, chemical stains and reticulation are NOT
     *                    modelled.
     * 10  OUTPUT EFFECT  Net density (D - D_min, per record) multiplied by
     *                    1 + 0.04 x level x n(film position), n in [-1, 1],
     *                    the same on all records; air bells at 0.5 per cm^2 x
     *                    level, radius 0.08-0.40 mm, reduce the net density at
     *                    their centre by 20-60 %. level = processingQuality x
     *                    damageStrength. ⚠ ALL FIGURES ARE DEFINITIONS, NOT
     *                    MEASUREMENTS (AlgoProcessingDefects.hpp).
     * 11  STAGES         09b (AlgoProcessingDefects.hpp), film-locked.
     * 12  INTERACTIONS   Gated by filmDamageEnabled and damageStrength. Applied
     *                    BEFORE the 9b particulates, so embedded dust is not
     *                    scaled by it. Reads profile curves D_min.
     * 13  SCALAR/AVX2    One shared header, HighPrecType in both builds; only
     *                    the store into the AlgoType plane rounds.
     * 14  FULL/LITE      PENDING. Expected Full only (stage 09b).
     */
    double processingQuality;

    /**
     *  1  NAME           dryingMarks
     *  2  TYPE           double
     *  3  AE CONTROL     slider
     *  4  UNIT           dimensionless rate multiplier
     *  5  MIN            0.0   (DryingMarksMin, clamped on entry)
     *  6  MAX            2.0   (DryingMarksMax, clamped on entry)
     *  7  DEFAULT        0.25   (AlgoControl.cpp, getFilmDamageDefault)
     *  8  STEP           0.01
     *  9  PURPOSE        CONSUMED SINCE 2026-10-10. Drying-stage water spots:
     *                    a drop dries from its edge and leaves a ring of
     *                    residue (tide line) with a faint interior deposit.
     *                    Squeegee bands are NOT modelled.
     * 10  OUTPUT EFFECT  0.15 marks per cm^2 x level, radius 0.5-2.5 mm of
     *                    film, neutral additive density 0.02-0.08 at the ring
     *                    crest (half-width 0.06 mm) and 0.2 of that inside.
     *                    Sized in FILM millimetres, so the gauge dependence is
     *                    derived, not authored: a 5 mm mark covers 2.3 per cent
     *                    of a 35 mm still frame and 25.6 per cent of a 16 mm
     *                    frame. ⚠ ALL FIGURES ARE DEFINITIONS, NOT
     *                    MEASUREMENTS (AlgoProcessingDefects.hpp).
     * 11  STAGES         09b (AlgoProcessingDefects.hpp), film-locked.
     * 12  INTERACTIONS   Gated by filmDamageEnabled and damageStrength.
     * 13  SCALAR/AVX2    One shared header, HighPrecType in both builds.
     * 14  FULL/LITE      PENDING. Expected Full only (stage 09b).
     */
    double dryingMarks;

    // =======================================================================
    //  AGE AND STORAGE
    // =======================================================================

    /**
     *  1  NAME           storageSeverity
     *  2  TYPE           double
     *  3  AE CONTROL     slider
     *  4  UNIT           dimensionless 0..1 severity axis (0 a cold vault, 1 a
     *                    warm attic for forty years)
     *  5  MIN            0.0   (StorageSeverityMin, clamped on entry)
     *  6  MAX            1.0   (StorageSeverityMax, clamped on entry)
     *  7  DEFAULT        0.20   (AlgoControl.cpp, getFilmDamageDefault)
     *  8  STEP           0.01
     *  9  PURPOSE        CONSUMED SINCE 2026-10-10. Turns the axis definition
     *                    into dark-storage time: 40 x storageSeverity x
     *                    damageStrength years at 30 degC (the warmest node of
     *                    Wilhelm Table 5.3), restated to storageCelsius and
     *                    added to storageYears before AlgoResolveStorageAge.
     * 10  OUTPUT EFFECT  Dye fade through film::DyeStabilitySpec, on the stocks
     *                    that publish a dark-fade rate; on every other stock it
     *                    is consumed and has no visible effect, exactly as
     *                    storageYears. Crossover, age fog, fungus and base
     *                    deterioration have no rate in the corpus and are NOT
     *                    modelled. 40 years / 30 degC are the control's own
     *                    definition, not a measurement.
     * 11  STAGES         frame setup (AlgoSeverityStorageYears,
     *                    AlgoStorageAge.hpp, called from AlgorithmMain.cpp).
     * 12  INTERACTIONS   Gated by filmDamageEnabled and damageStrength; adds to
     *                    storageYears, restated through storageCelsius.
     *                    Distinct from colourVeil - see that field's item 9.
     * 13  SCALAR/AVX2    Shared resolver, identical in both trees.
     * 14  FULL/LITE      PENDING. Both (a profile resolver, no pixel work).
     */
    double storageSeverity;

    /**
     *  1  NAME           colourVeil
     *  2  TYPE           double
     *  3  AE CONTROL     slider
     *  4  UNIT           additive density offset in one dye record
     *  5  MIN            0.0   (ColourVeilMin, clamped on entry)
     *  6  MAX            2.0   (ColourVeilMax, clamped on entry)
     *  7  DEFAULT        0.15   (AlgoControl.cpp, getFilmDamageDefault)
     *  8  STEP           0.01
     *  9  PURPOSE        CONSUMED SINCE 2026-10-10. A flat additive level
     *                    shift in ONE dye record, with no neutral point
     *                    anywhere in the tonal range.
     *
     *                    SEPARATE from the crossover storageSeverity would
     *                    drive, and deliberately so. Crossover is tone
     *                    dependent - shadows lean one way, highlights the
     *                    other, mid tones pass through neutral. A veil has no
     *                    neutral point at all. Measurement on aged ORWOCOLOR
     *                    found both on different parts of one roll, and the
     *                    second cannot be produced by turning up the first.
     *                    Merging them into one "fading" slider is the
     *                    commonest error in colour-fade emulation.
     * 10  OUTPUT EFFECT  colourVeil x damageStrength density added to one
     *                    record, chosen once per roll from damageSeed; level
     *                    only - contrast in the record is unchanged (an
     *                    earlier analysis claimed otherwise and was withdrawn
     *                    on re-measurement). On a monochrome stock, which has
     *                    one record, the offset is neutral.
     * 11  STAGES         09b (AlgoColourVeil, AlgoProcessingDefects.hpp).
     * 12  INTERACTIONS   Gated by filmDamageEnabled and damageStrength.
     * 13  SCALAR/AVX2    One shared header, identical in both builds.
     * 14  FULL/LITE      PENDING. Expected Full only (stage 09b).
     */
    double colourVeil;

    // =======================================================================
    //  MACHINE-SIDE -- motion picture
    // =======================================================================

    /**
     *  1  NAME           gateDirt
     *  2  TYPE           double
     *  3  AE CONTROL     slider
     *  4  UNIT           dimensionless multiplier on the persistent gate
     *                    population: initial count ALGO_GATE_INITIAL_COUNT
     *                    = 4.0 and STEADY-STATE COUNT ALGO_GATE_STEADY_COUNT
     *                    = 10 particles at level 1 (era-scaled), and on the
     *                    one-frame sparkle rate. 2026-10-20b: the accretion
     *                    rate is count / dustLifetime per SECOND (was
     *                    ALGO_GATE_ACCRETION_PER_FRAME per frame), so this is
     *                    a pure COUNT control
     *  5  MIN            0.0, ENFORCED (MAX_VALUE floor at stage 16)
     *  6  MAX            2.0. Indirect caps downstream: the arrival window
     *                    ALGO_GATE_WINDOW_MAX = 1024 ordinals and the sparkle
     *                    cap ALGO_SPARKLE_MAX_PER_FRAME = 64
     *  7  DEFAULT        0.60   (AlgoControl.cpp, getFilmDamageDefault)
     *  8  STEP           0.01 [proposed]. Note the quantity is small: at
     *                    default the initial population is 4.0 * 0.6 = 2.4
     *                    particles, so a 0.01 step is well under one particle
     *                    and the control reads as continuous only because the
     *                    accretion term is continuous
     *  9  PURPOSE        The particulate population lodged in the projector or
     *                    telecine gate rather than riding on the film. It holds
     *                    a FIXED SCREEN POSITION for hundreds to hundreds of
     *                    thousands of frames, while film-borne dust appears for
     *                    exactly one frame and vanishes.
     *
     *                    Separate from dustLevel because the same physical dust
     *                    splits into two behaviourally opposite populations,
     *                    and a simulation implementing only one of them is
     *                    immediately identifiable. Still photography has no
     *                    equivalent of this distinction, which is why it is
     *                    easy to miss.
     *
     *                    UNCALIBRATED: the reference scanner was measurably
     *                    clean, so no gate-dirt statistics could be derived.
     *                    The default is a starting point, not a measurement -
     *                    the only level in this struct of which that is true.
     * 10  OUTPUT EFFECT  Persistent screen-locked marks that accrete over a
     *                    reel, plus a one-frame sparkle population.
     * 11  STAGES         16
     * 12  INTERACTIONS   Multiplied by damageStrength. Gated by
     *                    filmDamageEnabled. Shares stage 16's class early-out
     *                    with damageEvents: if both are <= 0 the stage returns.
     *                    The sparkle branch additionally requires the profile's
     *                    dirt_events_per_frame to be non-zero. Its visual
     *                    conviction depends on weaveAmount - see that field.
     * 13  SCALAR/AVX2    Same semantics; derivation identical and in double in
     *                    both builds.
     * 14  FULL/LITE      PENDING. Expected Full only - stage 16 is on the Lite
     *                    drop list.
     */
    double gateDirt;

    /**
     *  1  NAME           weaveAmount
     *  2  TYPE           double
     *  3  AE CONTROL     slider
     *  4  UNIT           dimensionless multiplier on a MICROMETRE RMS
     *                    amplitude. The profile supplies weave_amp_x_um and
     *                    weave_amp_y_um in um; the stage converts um -> mm ->
     *                    pixels via pxPerMm, and this control scales the
     *                    amplitude before that conversion
     *  5  MIN            0.0, ENFORCED (MAX_VALUE floor at stage 15)
     *  6  MAX            2.0 advisory - NOT ENFORCED
     *  7  DEFAULT        0.50   (AlgoControl.cpp, getFilmDamageDefault)
     *  8  STEP           0.01 [proposed] - at typical amplitudes and 1024 px on
     *                    super35 this is a sub-hundredth-pixel change, which is
     *                    below the stage's own ALGO_WEAVE_MIN_SHIFT_PX =
     *                    1/512 px discard threshold, so the control reads
     *                    smooth rather than stepped
     *  9  PURPOSE        The small per-frame displacement of the image within
     *                    the gate. Scales the measured amplitude envelope.
     * 10  OUTPUT EFFECT  Shifts the whole frame sub-pixel per frame, resampled
     *                    with a 4x4 Catmull-Rom kernel.
     *
     *                    Weave is the CARRIER that makes every machine-fixed
     *                    defect convincing. Film-borne defects move with the
     *                    image, so they do not move relative to it; gate-borne
     *                    defects do not move with the image, so they appear to
     *                    shift by exactly the weave amount. That inverse
     *                    relationship is why a real gate scratch shimmers
     *                    instead of sitting still, and why a mathematically
     *                    static line reads as a digital overlay.
     * 11  STAGES         15
     * 12  INTERACTIONS   Multiplied by damageStrength. Gated by
     *                    filmDamageEnabled. Requires non-zero profile weave
     *                    amplitudes. Two further discards: amplitude in pixels
     *                    <= 0 (which is the only place pxPerMm acts as a gate
     *                    in the damage group), and a realised shift below
     *                    ALGO_WEAVE_MIN_SHIFT_PX. Its perceptual partner is
     *                    gateDirt.
     * 13  SCALAR/AVX2    Same displacement arithmetic in both. The resample
     *                    differs in implementation only: the scalar path calls
     *                    a per-pixel 4x4 Catmull-Rom sampler, the AVX2 path
     *                    hoists the frame-constant tap weights and vectorises
     *                    the identical separable convolution. Difference is
     *                    float-vs-double rounding, not algorithm.
     * 14  FULL/LITE      PENDING. Expected Full only - stage 15 is on the Lite
     *                    drop list, and a still preview does not want weave.
     */
    double weaveAmount;

    /**
     *  1  NAME           damageEvents
     *  2  TYPE           double
     *  3  AE CONTROL     slider
     *  4  UNIT           dimensionless RATE multiplier. It DIVIDES an interval
     *                    of ALGO_SPLICE_INTERVAL_SECONDS = 90 s, so 1.0 means
     *                    one event per 90 seconds of running film
     *  5  MIN            0.0, ENFORCED (MAX_VALUE floor at stage 16)
     *  6  MAX            2.0 advisory - NOT ENFORCED. There is an implicit
     *                    switch-off far above the advisory range: once the
     *                    derived interval falls below one frame the generator
     *                    stops entirely (at 24 fps that is a value above 2160)
     *  7  DEFAULT        0.20   (AlgoControl.cpp, getFilmDamageDefault)
     *  8  STEP           0.01 [proposed] - one hundredth of "one event per
     *                    90 s"; finer would be below the resolution of any
     *                    clip short enough to preview
     *  9  PURPOSE        Event-localised damage: splices, light leaks, static
     *                    discharge marks, cinch marks. These appear, evolve
     *                    over a few frames to a few hundred, and disappear - as
     *                    opposed to every other class here, which is either
     *                    per-frame or permanent.
     * 10  OUTPUT EFFECT  Transient localised events. Note a light leak adds
     *                    EXPOSURE, not brightness, so it passes through the
     *                    stock's characteristic curve and lifts shadows
     *                    dramatically while barely touching highlights. That is
     *                    why the defect layer needs access to the active stock
     *                    model and cannot be a post-process.
     * 11  STAGES         16
     * 12  INTERACTIONS   Multiplied by damageStrength. Gated by
     *                    filmDamageEnabled. Shares stage 16's class early-out
     *                    with gateDirt. Divides by the frame rate, so
     *                    frameRate changes the per-frame probability while
     *                    leaving the per-second rate fixed - which is the whole
     *                    point of the rule that rates are physical.
     * 13  SCALAR/AVX2    Same semantics; derivation identical and in double in
     *                    both builds.
     * 14  FULL/LITE      PENDING. Expected Full only (stage 16).
     */
    double damageEvents;

    /**
     *  1  NAME           flickerStops
     *  2  TYPE           double
     *  3  AE CONTROL     slider
     *  4  UNIT           RMS exposure gain in photographic STOPS
     *  5  MIN            0.0   (FlickerStopsMin, clamped on entry) -- except
     *                    the SENTINEL FlickerStopsStock = -1 (#209, 2026-10-16):
     *                    any negative value selects the STOCK'S OWN era flicker,
     *                    TemporalSpec::flicker_pct read as log2(1 + pct/100)
     *                    RMS stops (tier-3 era estimates on all 226 stocks)
     *  6  MAX            0.5   (FlickerStopsMax, clamped on entry)
     *  7  DEFAULT        0.15   (AlgoControl.cpp, getFilmDamageDefault) -- the
     *                    default is unchanged, so the sentinel moves nothing
     *                    unless the host selects it
     *  8  STEP           0.01 - one hundredth of a stop is below the visible
     *                    threshold for frame-to-frame flicker
     *  9  PURPOSE        CONSUMED SINCE 2026-10-10. Temporal exposure flicker:
     *                    hand-cranked cameras and early intermittent mechanisms
     *                    did not deliver equal exposure to successive frames.
     * 10  OUTPUT EFFECT  Linear exposure x 2^(flickerStops x damageStrength x
     *                    v(t)), v unit-variance value noise in time with knots
     *                    every 1 / TemporalSpec::flicker_hz s (1 Hz if unset),
     *                    t = frameIndex / frameRate. Acts on EXPOSURE, before
     *                    the characteristic curve: a brightness change applied
     *                    after development is a grade and does not move
     *                    highlights through the shoulder. COMMON-MODE: one
     *                    gain on all three channels, because no control and no
     *                    database field gives a per-channel split. The knot
     *                    blend is a definition (no source gives the flicker
     *                    spectrum). TemporalSpec::flicker_pct is not read.
     * 11  STAGES         03c (AlgoFlickerGain, AlgoTemporalFlicker.hpp).
     * 12  INTERACTIONS   Gated by filmDamageEnabled and damageStrength (stage
     *                    skipped by AlgoPass03c when inert). Reads frameIndex,
     *                    frameRate (fallback fps_native, then 24) and damageSeed.
     * 13  SCALAR/AVX2    Gain computed once per frame by the shared helper in
     *                    HighPrecType; the multiply is the same loop in both.
     * 14  FULL/LITE      PENDING. Both (one multiply per pixel).
     *
     * !! The earlier note about flickerBaseHz and flickerColourSpread is
     *   resolved: neither field exists, and none was added. The corner
     *   frequency comes from the stock's TemporalSpec and the channel split is
     *   common-mode - see AlgoTemporalFlicker.hpp.
     */
    double flickerStops;

    // =======================================================================
    //  SCANNER -- a separate layer, not a film defect
    // =======================================================================

    /**
     *  1  NAME           scannerArtifacts
     *  2  TYPE           double
     *  3  AE CONTROL     slider
     *  4  UNIT           dimensionless amplitude multiplier
     *  5  MIN            0.0   (ScannerArtifactsMin, clamped on entry)
     *  6  MAX            2.0   (ScannerArtifactsMax, clamped on entry)
     *  7  DEFAULT        0.20   (AlgoControl.cpp, getFilmDamageDefault)
     *  8  STEP           0.01
     *  9  PURPOSE        CONSUMED SINCE 2026-10-10. Digitisation artefacts -
     *                    illumination shading, fixed-pattern banding, Newton's
     *                    rings. Switchable independently of everything above,
     *                    because "clean scan of damaged film" and "good film,
     *                    cheap scanner" are both common requests.
     * 10  OUTPUT EFFECT  Transmittance multiplied by: 1 - 0.06 x level x
     *                    (r / r_corner)^2 shading; a per-column, per-channel
     *                    gain uniform in 1 +- 0.004 x level (trilinear line
     *                    sensor); and Newton's rings 1 + 0.03 x level x
     *                    exp(-r^2 / (6 mm)^2) x cos(pi r^2 / r1^2), r1 = 1 mm at
     *                    the green channel scaled by sqrt(lambda), centre drawn
     *                    once from damageSeed. No frame-index dependence: the
     *                    same scanner draws the same pattern on every frame.
     *                    ⚠ ALL AMPLITUDES ARE DEFINITIONS, NOT MEASUREMENTS
     *                    (AlgoScannerArtifacts.hpp). Worth keeping in mind: in
     *                    the reference dataset the strongest high-frequency
     *                    texture across 81 crops of damaged film was the
     *                    SCANNER's weave pattern, not any film defect.
     * 11  STAGES         16, applied last (AlgoScannerArtifacts.hpp).
     * 12  INTERACTIONS   Gated by filmDamageEnabled and damageStrength.
     *                    Independent of scannerFixedPattern, which is the
     *                    always-available non-damage control.
     * 13  SCALAR/AVX2    One shared header, HighPrecType in both builds.
     * 14  FULL/LITE      PENDING. Expected Full only (stage 16).
     */
    double scannerArtifacts;

    // =======================================================================
    //  2026-10-20 (owner-approved): APPENDED AT THE END OF THE STRUCT, so no
    //  existing field moves. Panel bit 51 (eCTRL_BIT_DEFECT_OPTICS).
    // =======================================================================

    /**
     *  1  NAME           defectOptics
     *  2  TYPE           bool
     *  3  AE CONTROL     checkbox ("Defect Reader Optics"), greyed / ignored
     *                    while filmDamageEnabled is false
     *  4  UNIT           none (switch)
     *  5  MIN            false
     *  6  MAX            true
     *  7  DEFAULT        false  (DefectOpticsDef; AlgoControl.cpp,
     *                    getFilmDamageDefault) -- OFF is bit-identical to the
     *                    defect model without this switch
     *  8  STEP           -
     *  9  PURPOSE        Makes the particulate and scratch defects of stage 9b
     *                    respond to the READER'S light geometry
     *                    (scannerSpecular) the way real dirt does: the
     *                    light-SCATTERING part of a defect (translucent dust,
     *                    fibres, a burnished base, the walls of a cut) is strong
     *                    under a directed / condenser reader and fades under a
     *                    diffuse one, while the ABSORBING part (opaque material,
     *                    emulsion removed) is the same under any reader. Also
     *                    removes the defects from the film's own Callier
     *                    steepening (stage 12b), which belongs to developed
     *                    silver and not to dirt (Trumpy & Gschwind, ACM JOCCH
     *                    8(2) 2015; doc/DESIGN_2026-10-19_defect_reader_geometry.md).
     * 10  OUTPUT EFFECT  Off: unchanged. On: each defect's density increment is
     *                    split by its class's provisional scattering share
     *                    (AlgoNegativeDefects.hpp, ALGO_*_SCATTER_SHARE) and the
     *                    scattering part is read through
     *                    T = (1 - s) + s * 10^-Ds, anchored so that at
     *                    s = ALGO_DEFECT_OPTICS_S_REF (0.25, provisional: the
     *                    reference scans' reader is undocumented) every defect
     *                    keeps its calibrated density. On a stock with Callier
     *                    Q > 1 the defect is composed in the reader's density
     *                    domain, so stage 12b steepens only the silver.
     * 11  STAGES         09b (AlgoDefectOptics.hpp)
     * 12  INTERACTIONS   Requires filmDamageEnabled and a non-zero dust / debris
     *                    / fibre / scratch level; reads scannerSpecular. Does
     *                    not touch processingQuality, dryingMarks, colourVeil
     *                    (density in the image itself, which Callier rightly
     *                    steepens).
     * 13  SCALAR/AVX2    One shared header, HighPrecType in both builds.
     * 14  FULL/LITE      Full only (stage 9b).
     */
    bool defectOptics;

    // =======================================================================
    //  TEMPORAL DEFECT CONTROLS -- 2026-10-20 (owner-approved spec «Improve
    //  the film-damage simulation ... temporally coherent»). APPENDED at the
    //  end of FilmDamage, panel bits 52 - 55, so no existing field or bit
    //  moves. Ignored while filmDamageEnabled is false.
    //  Semantics, units and the distribution behind each: AlgoNegativeDefects.hpp
    //  (scratches), AlgoGateDefects.hpp (persistent dust),
    //  doc/RESULT_2026-10-20b_temporal_defects.md.
    // =======================================================================

    /**
     *  1  NAME           scratchLifetime
     *  2  TYPE           double
     *  3  AE CONTROL     slider ("Transport Scratch Lifetime")
     *  4  UNIT           seconds of screen time, MEAN lifetime of one
     *                    transport scratch from its appearance to its end
     *  5  MIN            0.0 = the stock's own run length,
     *                    TemporalSpec::scratch_persistence_frames / 24 s
     *  6  MAX            60.0
     *  7  DEFAULT        0.0 (ScratchLifetimeDef, the stock value)
     *  8  STEP           0.1
     *  9  PURPOSE        How long a tramline persists once it has appeared -
     *                    distinct from how often new ones appear
     *                    (scratchTransport).
     * 10  OUTPUT EFFECT  Each event's lifetime is log-normal (sigma_ln 0.5)
     *                    with this mean, truncated to [0.25, 3] x mean, so a
     *                    scratch neither flashes for one frame nor lasts for
     *                    ever. The stretch of web it marks is its lifetime x
     *                    frameRate x frame pitch.
     * 11  STAGES         09b
     * 12  INTERACTIONS   Visible count = rate x lifetime / 60 (derived).
     * 13  SCALAR/AVX2    Shared source; identical.
     * 14  FULL/LITE      Full only.
     */
    double scratchLifetime;

    /**
     *  1  NAME           dustLifetime
     *  2  TYPE           double
     *  3  AE CONTROL     slider ("Persistent Dust Lifetime")
     *  4  UNIT           seconds of screen time, MEAN residence of one
     *                    persistent (gate / scanner) dust particle
     *  5  MIN            0.5
     *  6  MAX            600.0
     *  7  DEFAULT        42.0 (DustLifetimeDef; the previous 1000 frames at
     *                    24 fps)
     *  8  STEP           0.5
     *  9  PURPOSE        How long a persistent dust particle stays before it
     *                    is shed - distinct from how many there are (gateDirt)
     *                    and from film-borne dust (dustLevel, which moves with
     *                    the film and is new on every frame because every frame
     *                    is a new piece of film).
     * 10  OUTPUT EFFECT  Shed is a Poisson process with this mean residence;
     *                    the arrival rate is gateDirt's population divided by
     *                    it, so changing the lifetime changes how often the
     *                    pattern renews, NOT how many particles are present.
     * 11  STAGES         16
     * 12  INTERACTIONS   gateDirt (count), defectVariability (drift),
     *                    defectVisibility (opacity), host frameRate.
     * 13  SCALAR/AVX2    Shared header code; identical.
     * 14  FULL/LITE      Full only.
     */
    double dustLifetime;

    /**
     *  1  NAME           defectVariability
     *  2  TYPE           double
     *  3  AE CONTROL     slider ("Defect Temporal Variability")
     *  4  UNIT           dimensionless, 0 - 1; 0.5 = the reference amplitudes
     *  5  MIN            0.0 (persistent defects frozen in place and opacity)
     *  6  MAX            1.0 (twice the reference)
     *  7  DEFAULT        0.5 (DefectVariabilityDef)
     *  8  STEP           0.01
     *  9  PURPOSE        How much persistent defects change over time, without
     *                    changing how many there are or how strong they are.
     * 10  OUTPUT EFFECT  Persistent dust: smooth, time-correlated drift of
     *                    position (60 um), rocking (0.35 rad) and opacity
     *                    (down to 0.55 x) at the reference, correlated over
     *                    ALGO_GATE_DRIFT_CORR_S - replaces the former
     *                    independent per-frame jitter and the 10 % random blink.
     *                    Transport scratches: lateral wander and along-run
     *                    contrast change scaled by 2 x this.
     * 11  STAGES         09b, 16
     * 12  INTERACTIONS   Independent of the counts, rates and visibility.
     * 13  SCALAR/AVX2    Shared code; identical.
     * 14  FULL/LITE      Full only.
     */
    double defectVariability;

    /**
     *  1  NAME           defectVisibility
     *  2  TYPE           double
     *  3  AE CONTROL     slider ("Defect Visibility")
     *  4  UNIT           multiplier on each defect's density change (or, for
     *                    machine-side dirt, on its optical density -log10(1-a))
     *  5  MIN            0.0 (defects invisible)
     *  6  MAX            3.0
     *  7  DEFAULT        1.0 (DefectVisibilityDef; the calibrated contrast)
     *  8  STEP           0.01
     *  9  PURPOSE        How prominent defects are, separately from how many
     *                    there are (the level controls) and how often they
     *                    change (the rate / lifetime controls).
     * 10  OUTPUT EFFECT  Opacity a becomes 1 - (1 - a)^v: density scales by v,
     *                    opaque stays opaque, faint scales linearly. At 1.0
     *                    exactly the previous renders are reproduced.
     * 11  STAGES         09b (dust, debris, fibres, scratches), 16 (gate dirt,
     *                    sparkle)
     * 12  INTERACTIONS   damageStrength scales the AMOUNT (counts, rates);
     *                    this scales the CONTRAST.
     * 13  SCALAR/AVX2    Shared code; identical.
     * 14  FULL/LITE      Full only.
     */
    double defectVisibility;
};


// ===========================================================================
// Live controls -- 28 fields, 26 consumed by the renderer
//
// ⚠ 2026-09-28b (schema v57): storageCelsius added after storageYears, and it
// IS consumed (AlgoResolveStorageAge). The field count in the heading above
// was not maintained after 2026-09-09c -- batchPosition, scannerFixedPattern,
// exposureTimeS and the catalogue enumerators arrived since -- and is kept as
// the record of that date rather than restated from memory.
//
// ⚠ 25 -> 27 ON 2026-09-08: developmentMinutes and developmentCelsius were
// added and are NOT YET CONSUMED. They are declared now, sentinel-defaulted
// and required to hide themselves where they cannot act, for the same reason
// the eight unconsumed FilmDamage controls stay in the layout -- the panel and
// every serialised preset can then be built ONCE against a stable structure.
// Each says so in its own item 10.
//
// ⚠⚠ AND ONE PRE-EXISTING DEFAULT IS WRONG, FOUND WHILE ADDING THEM.
// `processVariant`'s item 7 says DEFAULT -1, and AlgorithmMain.cpp:475 states
// "processVariant is -1 unless the caller selects" -- but
// getAlgoControlsDefault() never assigns it and `AlgoControls controls{}`
// zero-initialises it to 0, which is variant INDEX ZERO, not the sentinel.
// It is currently HARMLESS BY ACCIDENT, not by design: variant[0] on all 8
// stocks that carry variants is the as-shipped row (REFINAL, "EI 800 (box
// speed)", "C-41 cross-process, as shipped", ...), every one of which has no
// curves and gamma_scale 1.0, so no pixel moves. The day a stock is added
// whose variant[0] is a push, the shipped default silently renders it.
// The same trap now applies to the two new controls, whose sentinel is
// strictly < 0 while zero-init gives 0.0 -- so BOTH must be assigned
// explicitly in getAlgoControlsDefault().
//
// ✅ FIXED 2026-09-08b. getAlgoControlsDefault() now assigns all three
// explicitly (AlgoControl.cpp): processVariant = -1, developmentMinutes = -1.0,
// developmentCelsius = -1.0. The paragraph above is kept because it is the only
// record of WHY -- a default that is correct by accident reads identical to one
// that is correct by design, right up to the day a stock is added whose
// variant[0] is a push.
//
// ⚠ +1 ON 2026-10-01 (schema v60): developerIndex added after
// developmentCelsius, and it IS consumed (AlgoResolveDeveloper). Sentinel -1,
// assigned explicitly in getAlgoControlsDefault().
// ⚠ 2026-10-16f: developerIndex is now a DeveloperCtrl list-box value (same
// size and position, int32 underlying type); eAS_SHIPPED = 0 is the default.
//
// ⚠ +1 ON 2026-10-16 (#209): filmAgeMonths added after storageCelsius, and it
// IS consumed (AlgoResolveFilmAge, AlgoFilmAgeLogShift). Sentinel 0 (fresh),
// assigned explicitly in getAlgoControlsDefault().
//
// ⚠ 27 -> 28 ON 2026-09-09c: blackPointStretch added, and it IS consumed
// (stage 14 plus the anchor solve), so the consumed count moves with it:
// 25/27 on 2026-09-08, 26/28 now. The two unconsumed fields are still
// developmentMinutes and developmentCelsius.
// ===========================================================================
//: The three catalogue controls -- filmFormat, printStock and dupeStock -- are
//: strongly typed enumerators defined in AlgoControlEnums.hpp. They were
//: fixed-size character arrays until the enumeration migration; the capacity
//: constants that sized those arrays are gone with them, along with the
//: truncation hazard where an over-long key silently resolved to the profile
//: default and then looked like a working setting.

struct AlgoControls
{
    // -- film selection -----------------------------------------------------

    /**
     *  1  NAME           filmProfile
     *  2  TYPE           film::eFILM_PROFILE (generated enumeration)
     *  3  AE CONTROL     dropdown / popup, populated from film_names.txt
     *  4  UNIT           dimensionless array index into the vector returned by
     *                    film::GetFilmDatabase(); equivalently line
     *                    (value + 1) of film_names.txt
     *  5  MIN            0. NOT ENFORCED - see item 10
     *  6  MAX            profile count - 1. NOT ENFORCED
     *  7  DEFAULT        static_cast<film::eFILM_PROFILE>(0)
     *                    (AlgoControl.cpp, getAlgoControlsDefault)
     *  8  STEP           1, by enumeration. The panel presents a list, so there
     *                    is no drag increment
     *  9  PURPOSE        Selects which film stock to simulate. This is the one
     *                    control that changes every other control's meaning,
     *                    because all the others scale or override numbers the
     *                    selected profile supplies.
     * 10  OUTPUT EFFECT  Selects the whole parameter set. ⚠ SINCE 2026-10-02
     *                    the index is clamped to [0, profileCount - 1] by
     *                    AlgoControlsClamped on entry, once per frame. Before
     *                    that it was dereferenced with no range check and an
     *                    out-of-range value was an out-of-bounds read. The
     *                    host should still send a valid index: a clamped one
     *                    silently renders a different stock.
     * 11  STAGES         every stage except 02 and 17, which do not receive the
     *                    profile
     * 12  INTERACTIONS   Supplies the fallback for filmFormat
     *                    (profile.default_format), printStock
     *                    (profile.default_print) and, through printStock,
     *                    dupeStock. Supplies default_flare and
     *                    default_vignette for the two sentinel controls.
     *                    profile.is_monochrome disables wbStrength, part of
     *                    couplerScale and misregScale. profile.has_reseau gates
     *                    reseau. profile.isReversal() disables printGrain and
     *                    the whole duplication chain.
     * 13  SCALAR/AVX2    Identical - the profile is dereferenced in the shared
     *                    driver translation unit, which both builds compile.
     * 14  FULL/LITE      PENDING. Both, necessarily and identically. The
     *                    architectural rule is that Preview and Full consume
     *                    the SAME authoritative FilmProfile; neither mode may
     *                    substitute or duplicate film data.
     *
     * !! PERSISTENCE WARNING, CURRENT AS OF THIS TREE: profiles are ordered by
     *   name, so inserting a stock renumbers every enumerator after it and a
     *   saved project would silently render a DIFFERENT film after a database
     *   update. A saved project must therefore store the NAME, not this number.
     *   NOTE: a frozen-identifier scheme (film_ids.lock plus a separate
     *   display-order list) exists in the project design and removes this
     *   hazard, but it is NOT present in this tree. When it lands, this warning
     *   must be replaced - not deleted - by the frozen-id contract.
     */
    film::eFILM_PROFILE filmProfile;

    /**
     *  1  NAME           frameRate
     *  2  TYPE           double
     *  3  AE CONTROL     numeric field, host-supplied
     *  4  UNIT           FRAMES PER SECOND of film, after any layer time
     *                    stretch. Confirmed by the arithmetic: fps divided by a
     *                    hertz corner frequency yields a period in frames, and
     *                    seconds multiplied by fps yields frames
     *  5  MIN            1.0, ENFORCED - but only at the two sites that use it
     *                    (stages 15 and 16), not at the driver. Zero and
     *                    negative values are silently raised to 1.0 there
     *  6  MAX            none documented and NONE ENFORCED
     *  7  DEFAULT        24.0   (AlgoControl.cpp, getAlgoControlsDefault)
     *  8  STEP           0.001 [proposed] - it must express 23.976 and 29.97
     *                    exactly enough to be recognisable; a coarser step
     *                    cannot
     *  9  PURPOSE        Converts the temporal stages' per-SECOND rates into
     *                    per-frame rates. It belongs in the controls rather
     *                    than the call signature because it changes what the
     *                    algorithm computes: a defect rate tuned at 24 fps must
     *                    not run twice as fast on a 50 fps timeline.
     * 10  OUTPUT EFFECT  Sets the weave period and the gate-event probability
     *                    per frame. It does not change how much damage there
     *                    is per second - that is the invariant it exists to
     *                    protect.
     * 11  STAGES         03c (flicker time base, since 2026-10-10), 15 and 16.
     *                    It is also passed to 09b, which discards it because
     *                    none of its classes has a per-second rate.
     * 12  INTERACTIONS   Paired with frameIndex at every temporal call site.
     *                    Reaches nothing unless filmDamageEnabled is true and
     *                    the relevant damage level is non-zero.
     * 13  SCALAR/AVX2    Same semantics. The fps arithmetic is HighPrecType
     *                    (double) in BOTH builds. One narrowing does differ:
     *                    the driver casts frameRate to AlgoType, which is
     *                    double in the scalar build and float in the AVX2
     *                    build.
     * 14  FULL/LITE      PENDING. Expected Full only, since both its consumers
     *                    are on the Lite drop list.
     */
    double frameRate;

    // -- film format --------------------------------------------------------
    //
    //  NOTHING IN THIS STRUCTURE DESCRIBES THE HOST'S PIXEL BUFFERS.
    //
    //  This structure and Algorithm_Main are a plain C++14 API with no knowledge
    //  of any render infrastructure. They must compile and behave identically
    //  when called from Premiere, After Effects, Photoshop, a command-line test
    //  harness or a third-party application, and none of those may need to be
    //  named or accommodated here.
    //
    //  So sample storage is NOT a control. The engine's boundary is planar
    //  ImgType normalised to 0..1, chosen once in AlgoTypes.hpp; the host's own
    //  unpack and repack decide what 8u, 16u, 10u, 32f or anything else means,
    //  and the round trip is the host's to keep symmetric. An output-depth field
    //  here could only ever disagree with the buffer the host actually supplied,
    //  which is why the two that used to live below have been removed:
    //
    //      bitDepth   claimed to select 8 or 16 bit output, defaulted to 16, and
    //                 was read by no stage. It offered no float option at all,
    //                 so any host handing over 32f had no way to describe it. It
    //                 also let the caller ask for an output width unrelated to
    //                 the input width, which is exactly the coupling this API
    //                 must not have.
    //
    //      maxDim     a debug downscale of the OUTPUT, likewise read by nothing.
    //                 Render extent already arrives as the sizeX and sizeY
    //                 arguments of Algorithm_Main, which is the authoritative
    //                 geometry; a second, disagreeing extent in the controls
    //                 would silently fight it. A caller wanting a cheap render
    //                 passes smaller sizeX and sizeY.
    //
    //  Removing them changes the layout of this structure, so any caller that
    //  assigned either field now fails to compile. That is intended: a silent
    //  drop would leave host code believing it still selected an output depth.
    //
    //  What DOES stay here is the film gauge, and the distinction is worth being
    //  precise about, because it looks superficially similar. The gauge is not a
    //  buffer property - it is the physical width of the film in millimetres,
    //  which together with the render width in pixels yields px_per_mm and makes
    //  every spatial quantity in the simulation resolution independent. It
    //  changes what the algorithm computes. A pixel format does not.

    // ----------------------------------------------------------------------
    //  ⚠⚠ THE THREE NAME-KEY CONTROLS ARE FIXED-SIZE CHARACTER ARRAYS, NOT
    //  `const char*`. CORRECTED 2026-09-09 AT THE OWNER'S INSTRUCTION, and the
    //  reason is OWNERSHIP rather than style:
    //
    //    * A POINTER CANNOT BE SERIALISED. AlgoControls is the struct an After
    //      Effects or Premiere project must persist. Storing a `const char*`
    //      writes an address that means nothing on reload, and reading one
    //      back is undefined behaviour rather than a wrong value -- the
    //      failure mode is a crash or garbage, not a bad render.
    //    * A POINTER HAD NO OWNER. The struct never said who allocated the
    //      string or how long it lived, so the only safe caller was one
    //      passing a literal with static lifetime. Nothing enforced that, and
    //      a host passing a temporary would have worked in testing and failed
    //      in the field.
    //    * ARRAYS MAKE THE STRUCT FLAT AND COPYABLE. `AlgoControls c = other;`
    //      now copies the whole configuration, which is what a preset, an undo
    //      step and a render-thread hand-off each need.
    //
    //  ⚠ WHY ARRAYS RATHER THAN ENUMS, since either was permitted. Both are
    //  sound. Arrays were chosen because these three keys are looked up BY
    //  NAME against tables the DATABASE owns -- `FORMAT_GEOM` and the print
    //  stock list -- and the profile's own `default_format` / `default_print`
    //  fallbacks are themselves strings. An enum would need a name<->value
    //  translation table kept in step with the generated database BY HAND,
    //  which is the class of duplication this project has been removing (cf.
    //  the N_DATA_SLOTS note: the slot count is read from the emitter, never
    //  repeated). Arrays fix the ownership defect and change no lookup or
    //  comparison anywhere.
    //  ⚠ THE FILM STOCK ITSELF IS NOT ONE OF THESE and is deliberately left
    //  alone: `filmProfile` is already a generated enum (`film::eFILM_PROFILE`,
    //  184 values in film_enum.hpp) and its legal values are NOT enumerated in
    //  these comments -- they are the database, and duplicating 184 names here
    //  would be the exact duplication the paragraph above rejects.
    //
    //  ⚠ EVERY WRITE MUST BE BOUNDED AND NUL-TERMINATED. `strncpy` does not
    //  terminate on truncation; pair it with an explicit terminator or use a
    //  helper that guarantees both.
    //
    //  ⚠ THE .cpp SIDE STILL NEEDS EDITING AND WAS NOT TOUCHED HERE.
    //  `getAlgoControlsDefault()` assigns all three by pointer today
    //  (`controls.filmFormat = "super35";`), which no longer compiles against
    //  an array; each becomes a bounded copy. Every READER is unaffected --
    //  they compare with strcmp, and an array decays to `const char*` at the
    //  call site.
    // ----------------------------------------------------------------------

    /**
     *  1  NAME           filmFormat
     *  2  TYPE           FilmFormatCtrl (AlgoControlEnums.hpp)
     *                    ⚠ WAS `const char*` until 2026-09-09 -- see the
     *                    ownership note above this block
     *  3  AE CONTROL     dropdown, populated from FORMAT_GEOM. ⚠ THE COMPLETE
     *                    LEGAL SET IS 14 VALUES, exhaustive as of schema v28,
     *                    with the frame WIDTH each one selects and the number
     *                    of the 191 stocks that name it as their own default:
     *                      "8mm"             4.800 mm    2 stocks
     *                      "super8"          5.790 mm    -
     *                      "16mm"           10.260 mm    6 stocks
     *                      "super16"        12.520 mm    -
     *                      "academy35"      21.950 mm    4 stocks
     *                      "anamorphic35"   21.950 mm    -
     *                      "super35"        24.890 mm   64 stocks
     *                      "techni35"       24.890 mm    1 stock
     *                      "ff35"           36.000 mm   94 stocks
     *                      "medium645"      56.000 mm    2 stocks
     *                      "imax15"         70.410 mm    -
     *                      "polaroid_sx70"  79.000 mm    1 stock
     *                      "polaroid_pack"  95.000 mm    2 stocks
     *                      "large4x5"      127.000 mm    8 stocks
     *                    ⚠ academy35/anamorphic35 and super35/techni35 are
     *                    PAIRS AT ONE WIDTH and are NOT interchangeable: they
     *                    differ in frame height and pitch, so the panel must
     *                    offer all four rather than collapsing either pair.
     *  4  UNIT           a name key, dimensionless. The quantity it selects is
     *                    LENGTH IN MILLIMETRES - frame width, height and pitch
     *  5  MIN            n/a (string). An EMPTY FIRST BYTE means "use the
     *                    stock's default_format". ⚠ nullptr is no longer
     *                    possible and no longer has to be tested: the field is
     *                    storage, not a pointer
     *  6  MAX            n/a. An unrecognised key falls back to the profile
     *                    default; if that also fails to resolve, the geometry
     *                    is zeroed rather than guessed. ⚠ A value longer than
     *                    15 characters cannot be stored and must be rejected by
     *                    the host, not silently truncated -- a truncated key
     *                    would resolve to the profile default and look like a
     *                    working setting
     *  7  DEFAULT        "super35"   (AlgoControl.cpp, getAlgoControlsDefault)
     *  8  STEP           n/a - enumeration by list, 14 entries
     *  9  PURPOSE        Selects the film gauge. With the render width in
     *                    pixels this gives px_per_mm, the single mechanism that
     *                    makes every spatial quantity in the simulation -
     *                    grain, halation, MTF, registration, defect size -
     *                    resolution independent.
     * 10  OUTPUT EFFECT  Changes the physical scale of every spatial effect. A
     *                    zeroed geometry disables the spatial stages outright,
     *                    because they all test pxPerMm > 0.
     * 11  STAGES         through pxPerMm: 03b, 05, 06, 07, 09, 09b, 10, 11, 13,
     *                    14, 14b, 15, 16. Through the mm extents and frame
     *                    pitch: 04, 09b, 10b, 15, 16.
     * 12  INTERACTIONS   Falls back to filmProfile's default_format. The
     *                    derived pxPerMm gates reseau (below ~3 px per reseau
     *                    cell the mosaic is switched off) and scales the
     *                    effects that grainScale, halationScale, couplerScale
     *                    and misregScale multiply.
     * 13  SCALAR/AVX2    Resolved in the shared driver, so identical. Note the
     *                    derived pxPerMm is AlgoType - double in scalar, float
     *                    in AVX2 - and the AVX2 stages re-widen it to double
     *                    before use.
     * 14  FULL/LITE      PENDING. Both, identically. Format is geometry, not
     *                    quality, and must not differ between modes or the two
     *                    would not be comparable.
     */
    FilmFormatCtrl filmFormat;

    /**
     *  1  NAME           printStock
     *  2  TYPE           PrintStockCtrl (AlgoControlEnums.hpp)
     *                    ⚠ WAS `const char*` until 2026-09-09 -- see the
     *                    ownership note above the filmFormat block
     *  3  AE CONTROL     dropdown
     *                    ⚠ THE COMPLETE LEGAL SET IS 11 VALUES plus the empty
     *                    sentinel, exhaustive as of schema v28. Print stocks
     *                    are a SEPARATE TABLE from the 184 film profiles and
     *                    are NOT in film_names.txt:
     *                      ""                         use the stock's own
     *                                                 default_print
     *                      "SCAN_DI"                  digital-intermediate
     *                                                 inversion -- the default
     *                                                 of 190 of 191 profiles
     *                      "KODAK_2383_RELEASE"       modern release print
     *                      "KODAK_VISION3_DI_2254"    DI intermediate
     *                      "KODAK_5302"               fine-grain release
     *                                                 positive, 16 mm
     *                      "EASTMANCOLOR_5382_1953"   1950s release print
     *                      "TECHNICOLOR_IB"           imbibition (dye
     *                                                 transfer) -- the
     *                                                 default_print of exactly
     *                                                 1 profile
     *                      "DUPE_FINE_GRAIN"          fine-grain duplicating
     *                      "TSP_1_POSITIVE"           Soviet TSP-1
     *                      "TSP_3_POSITIVE"           Soviet TSP-3
     *                      "TSP_6_POSITIVE"           Soviet TSP-6
     *                      "TASMA_POSITIVE_28"        Tasma positive 28
     *  4  UNIT           a name key, dimensionless. The quantities it selects
     *                    are the print stock's mtf_f50 in CYCLES PER MILLIMETRE
     *                    and its characteristic curves in OPTICAL DENSITY
     *  5  MIN            n/a. An EMPTY FIRST BYTE (the default) means "use the
     *                    stock's own default_print"
     *  6  MAX            n/a. An unmatched name degrades silently to the
     *                    profile default, and failing that to no print stock at
     *                    all. ⚠ A value longer than 23 characters must be
     *                    rejected by the host rather than truncated: the
     *                    longest legal key is 22 characters, so a truncation
     *                    can only ever produce a name that matches nothing
     *  7  DEFAULT        ""   (AlgoControl.cpp, getAlgoControlsDefault)
     *  8  STEP           n/a - enumeration by list, 11 entries + the sentinel
     *  9  PURPOSE        Selects the print emulsion. Nobody looks at a
     *                    negative; this second emulsion is what produces
     *                    correct highlight rolloff and shadow crush.
     * 10  OUTPUT EFFECT  Sets the print curves used in the anchor solve and the
     *                    print MTF used as the scan sigma. Resolving to nothing
     *                    disables the duplication stage entirely and zeroes the
     *                    scan f50.
     * 11  STAGES         08 and 08b via the anchor solve; 13; 14. Through the
     *                    derived scan sigma also 10, 11 and 13.
     * 12  INTERACTIONS   Falls back to filmProfile's default_print. It gates
     *                    dupeStock's own fallback. Enters the anchor solve
     *                    jointly with greyTarget and couplerScale. Its
     *                    grain_rms is one of the five conditions printGrain
     *                    needs.
     * 13  SCALAR/AVX2    Resolved in the shared driver, so identical.
     * 14  FULL/LITE      PENDING. Both. The print stock is part of the film's
     *                    identity - tone scale and dmin/dmax - which the Lite
     *                    design explicitly keeps at full quality.
     */
    PrintStockCtrl printStock;

    /**
     *  1  NAME           dupeStock
     *  2  TYPE           PrintStockCtrl (AlgoControlEnums.hpp)
     *                    ⚠ WAS `const char*` until 2026-09-09 -- see the
     *                    ownership note above the filmFormat block. Shares
     *                    printStock's capacity because it draws on the SAME
     *                    table
     *  3  AE CONTROL     dropdown. ⚠ THE LEGAL SET IS THE SAME 11 PRINT STOCK
     *                    KEYS listed under printStock, plus the empty
     *                    sentinel. It is one table, not two -- any print stock
     *                    may serve as an intermediate. In practice only
     *                    "DUPE_FINE_GRAIN" and "KODAK_VISION3_DI_2254" are
     *                    intermediates by design. ⚠ SINCE 2026-10-02 ANY
     *                    OTHER SELECTION (a release print, SCAN_DI,
     *                    eSTOCKS_OWN) IS REPLACED BY DupeStockCtrlDef in
     *                    AlgoControlsClamped: each generation is two passes
     *                    through this stock's curve, and a high-gamma stock
     *                    compounds contrast until the image is binary (the
     *                    owner's black-and-white report of 2026-10-02).
     *  4  UNIT           a name key, dimensionless. Selects mtf_f50 in cycles
     *                    per millimetre, dmin in optical density, and an rms
     *                    grain figure
     *  5  MIN            n/a. An EMPTY FIRST BYTE degrades to printStock
     *  6  MAX            n/a. Unmatched degrades to the print stock, then to
     *                    nothing - and nothing forces the duplication chain to
     *                    zero passes whatever generations says. ⚠ Over 23
     *                    characters must be rejected, not truncated
     *  7  DEFAULT        "DUPE_FINE_GRAIN"
     *                    (AlgoControl.cpp, getAlgoControlsDefault)
     *  8  STEP           n/a - enumeration by list, same 11 entries
     *  9  PURPOSE        Selects the intermediate stock for the duplication
     *                    chain.
     * 10  OUTPUT EFFECT  NONE AT DEFAULT SETTINGS, because generations defaults
     *                    to 0 and the chain is skipped. With generations > 0 it
     *                    sets the grain and softening each intermediate adds.
     * 11  STAGES         13 only
     * 12  INTERACTIONS   Fallback chain depends on printStock and filmProfile.
     *                    Multiplied out by generations. Its grain contribution
     *                    is gated by grainScale.
     * 13  SCALAR/AVX2    Resolved in the shared driver, so identical.
     * 14  FULL/LITE      PENDING. Expected Full only in effect - the Lite
     *                    design drops duplication generations beyond one, so
     *                    this control's influence is reduced rather than the
     *                    control being ignored.
     */
    DupeStockCtrl dupeStock;

    /**
     *  1  NAME           generations
     *  2  TYPE           int32_t
     *  3  AE CONTROL     integer slider
     *  4  UNIT           dimensionless COUNT of interpositive / dupe-negative
     *                    PAIRS. The stage runs 2 x generations passes
     *  5  MIN            0, ENFORCED (CLAMP_VALUE at stage 13)
     *  6  MAX            4, ENFORCED (CLAMP_VALUE against
     *                    ALGO_DUPE_MAX_GENERATIONS). Until 2026-10-02 this was
     *                    the only control with an enforced upper bound; every
     *                    control is now clamped by AlgoControlsClamped.
     *                    The cap exists so a mistyped control cannot turn one
     *                    frame into a minute
     *  7  DEFAULT        0   (AlgoControl.cpp, getAlgoControlsDefault) -
     *                    camera negative straight to print
     *  8  STEP           1 - integral by type, and the effective UI increment
     *                    is also 1
     *  9  PURPOSE        How many intermediate generations sit between camera
     *                    negative and print.
     * 10  OUTPUT EFFECT  Each generation adds grain and softens detail, and the
     *                    grain-to-detail ratio worsens monotonically - which is
     *                    the measurable signature of a dupe rather than a
     *                    stylistic guess.
     * 11  STAGES         13 only
     * 12  INTERACTIONS   Forced to no effect when dupeStock or printStock fails
     *                    to resolve, and when the stock is reversal. Per-
     *                    generation grain is gated by grainScale. couplerScale
     *                    sets the neutral mid density each generation anchors
     *                    against; greyTarget sets the final print target after
     *                    the chain.
     * 13  SCALAR/AVX2    Same semantics; identical clamp and identical pass
     *                    count. Integer arithmetic in both.
     * 14  FULL/LITE      PENDING. Expected: Lite caps it at one generation
     *                    rather than dropping the stage.
     */
    int32_t generations;

    // -- exposure and tone --------------------------------------------------

    /**
     *  1  NAME           exposureStops
     *  2  TYPE           double
     *  3  AE CONTROL     slider
     *  4  UNIT           PHOTOGRAPHIC STOPS - a base-2 logarithmic exposure
     *                    ratio. Confirmed by the code: the field is used
     *                    directly as the exponent of 2, so +1 doubles and -1
     *                    halves the light
     *  5  MIN            -4 advisory - NOT ENFORCED
     *  6  MAX            +4 advisory - NOT ENFORCED. There is no guard of any
     *                    kind; a large value produces an overflowing gain
     *  7  DEFAULT        0.0   (AlgoControl.cpp, getAlgoControlsDefault)
     *  8  STEP           0.1 [proposed], with 1/3-stop detents at 0.333 if the
     *                    panel supports them - one third of a stop is the
     *                    photographic convention and the smallest increment a
     *                    cinematographer thinks in
     *  9  PURPOSE        The exposure the cinematographer chose. Not a
     *                    brightness trim.
     * 10  OUTPUT EFFECT  Moves the scene along the characteristic curve, so the
     *                    toe and shoulder do the work. This is why it is not
     *                    interchangeable with a gain applied later: the same
     *                    ratio applied after development is a grade and does
     *                    not compress highlights through the shoulder.
     * 11  STAGES         02 only
     * 12  INTERACTIONS   NONE with any other control. Worth stating explicitly
     *                    because it looks otherwise: stage 02 divides by a
     *                    compile-time constant ALGO_MID_GREY = 0.18 which
     *                    shares its value with greyTarget's default but is a
     *                    different quantity entirely and is not connected to
     *                    that control. Stage 02 does not even receive the film
     *                    profile.
     * 13  SCALAR/AVX2    Same semantics. Both builds evaluate the pow() in
     *                    double and narrow once; the AVX2 build then broadcasts
     *                    a float gain, the scalar build keeps double.
     * 14  FULL/LITE      PENDING. Expected both, at full quality - stage 02 is
     *                    on the Lite keep-at-full-quality list and costs
     *                    almost nothing.
     */
    double exposureStops;

    /**
     *  1  NAME           exposureTimeS
     *  2  TYPE           double
     *  3  AE CONTROL     slider, or a text field -- the useful range spans eight
     *                    decades, which no linear slider covers usefully, so a
     *                    logarithmic slider or a typed value is the practical
     *                    form
     *  4  UNIT           SECONDS of exposure. It is an absolute time, not a
     *                    ratio: reciprocity failure is a function of the actual
     *                    duration, which is why a stops offset cannot express it
     *  5  MIN            0 is the OFF sentinel, not a minimum. The active range
     *                    begins at 1e-5 s
     *  6  MAX            3600 s
     *  7  DEFAULT        0 -- reciprocity off
     *  8  STEP           logarithmic [proposed]: 1/3 decade per detent, or a
     *                    typed value. A linear step cannot serve 1e-5 and 3600
     *                    from one control
     *  9  PURPOSE        Drives the reciprocity-failure model. Film does not
     *                    obey the reciprocity law at long or very short
     *                    exposures: the same total light delivered slowly
     *                    produces less density than the same light delivered
     *                    quickly, and the shortfall is per-layer, so it shifts
     *                    colour balance as well as speed.
     * 10  OUTPUT EFFECT  Applies a per-layer exposure correction before the
     *                    characteristic curve. 0 reproduces pre-field renders
     *                    BIT FOR BIT -- that is the contract the field was added
     *                    under, and it is what makes it safe to ship.
     * 11  STAGES         applied before stage 08; the correction is folded into
     *                    the log exposure the curve sees
     * 12  INTERACTIONS   Compounds with exposureStops -- one is the aperture and
     *                    shutter the photographer chose, the other is how long
     *                    the shutter was open, and the film responds to both but
     *                    not identically. Its per-layer asymmetry interacts with
     *                    sceneKelvin and wbStrength, since both move colour
     *                    balance.
     * 13  SCALAR/AVX2    External semantics identical. The correction is a
     *                    frame-setup scalar, so it resolves before the pixel
     *                    loop in both paths.
     * 14  FULL/LITE      PENDING. Expected both, at full quality -- it resolves
     *                    into a per-layer exposure scalar with zero per-pixel
     *                    cost, which is exactly the class of physics the Lite
     *                    design keeps.
     *
     * !! ITEMS 5, 6, 7, 11 AND 12 ARE TAKEN FROM THE PROJECT HANDOFF'S CONTROL
     *    TABLE, NOT READ FROM THE CONSUMING STAGE. This field does not exist in
     *    the 2026-08-27 source archive this documentation pass was performed
     *    against, so its clamps, its exact gating and its stage placement could
     *    not be verified the way every other field's were. Treat these five
     *    items as UNVERIFIED until the reciprocity stage is read. Everything
     *    else in this block follows from the field's stated purpose and unit.
     */
    double exposureTimeS;

    /**
     *  1  NAME           processVariant
     *  2  TYPE           ProcessVariantCtrl (AlgoControlEnums.hpp)
     *  3  AE CONTROL     list box of SIX steps (2026-10-11): As shipped,
     *                    Pull -2, Pull -1, Push +1, Push +2, Push +3
     *                    (ProcessVariantCtrlStr). The host enables only the
     *                    steps the selected stock offers, found by matching
     *                    ProcessVariantCtrlKey against that stock's own
     *                    `process_variants`, and greys the rest. Every colour
     *                    negative offers Push +1 / +2 (own data or the family
     *                    model); Pull -2 is offered by no stock today.
     *                    \warning TOTAL_PROCESSES IS NOT AN ITEM.
     *  4  UNIT           an enumerator naming a STEP, the same on every stock.
     *                    A project saved before 2026-10-11 stores a value of
     *                    the old 49-entry list; ProcessVariantCtrlFromLegacy()
     *                    maps it (ProcessVariantCtrlMigrate).
     *  5  MIN            eAS_SHIPPED (0)
     *  6  MAX            TOTAL_PROCESSES - 1 (ePUSH_3, 5). A value outside
     *                    [0, TOTAL_PROCESSES), or a step this stock does not
     *                    offer, is treated as eAS_SHIPPED rather than clamped:
     *                    a stale preset should render the stock as shipped,
     *                    not render some other development in its place
     *  7  DEFAULT        eAS_SHIPPED -- the development the stored curves
     *                    represent
     *  8  STEP           1
     *  9  PURPOSE        Selects a pushed or pulled DEVELOPMENT of the same
     *                    emulsion. Where the manufacturer plotted it the record
     *                    carries its own TRACED curve set; for a colour negative
     *                    without one at Push +1 / +2 it carries the family
     *                    model (mean gamma scale and D-min shift of the traced
     *                    colour-negative push ladders) -- an ESTIMATE.
     * 10  OUTPUT EFFECT  Replaces the profile's characteristic curves (own
     *                    data) or scales gamma and shifts D-min (family model),
     *                    and sets the exposure index where the record states
     *                    one, before anything reads either. eAS_SHIPPED
     *                    reproduces pre-field renders BIT FOR BIT: the base
     *                    profile is returned by reference and nothing is copied.
     * 11  STAGES         resolved in frame setup, before the anchor solve. Read
     *                    downstream by stage 8, stage 11's grain amplitude and
     *                    stage 13's dupe chain, all of which take the profile
     *                    and therefore see one consistent film.
     * 12  INTERACTIONS   Compounds with nothing: it selects the curve every
     *                    other control is then applied to. A pushed variant
     *                    already includes the speed change the photographer
     *                    made, so it should NOT be combined with an
     *                    exposureStops offset meant to represent the same push.
     * 13  SCALAR/AVX2    External semantics identical. Resolution is a frame
     *                    setup step with no pixel loop in either path.
     * 14  FULL/LITE      Both, at full quality: one profile copy per frame,
     *                    and only when a variant is selected.
     */
    ProcessVariantCtrl processVariant;

    /**
     *  1  NAME           batchPosition
     *  2  TYPE           double
     *  3  AE CONTROL     slider, and it MUST HIDE OR DISABLE ITSELF on a stock
     *                    that carries no `tolerance` record -- the same rule
     *                    processVariant follows for an empty variant list.
     *                    \warning THAT IS 220 OF 232 STOCKS -- the twelve it
     *                    moves are the Soviet TU / GOST films whose documents
     *                    print two-sided gamma or D-min bands (the GOST 24876
     *                    Foto films and FN-64 print one-sided limits it does
     *                    not move). The host reads it per film from bit
     *                    eCTRL_BIT_BATCH_POSITION of kFilmControlAvailability
     *                    (film_params_mask.hpp; the bit was missing until
     *                    2026-09-30), and batch_position_headers.py asserts
     *                    that bit against the resolver on every build.
     *                    film_enum.hpp repeats this control's range as
     *                    film::eBATCH_POSITION_{MIN,MAX,DEF,STEP}, copied from
     *                    AlgoControlEnums.hpp; AlgoControl.cpp static_asserts
     *                    that the two agree. Suggested label
     *                    "Batch position", with -1 read out as "lower limit",
     *                    0 as "as specified" and +1 as "upper limit".
     *  4  UNIT           a FRACTION of the distance from the stored value to
     *                    the edge of that film's own published acceptance
     *                    band. Not stops, not density, not a percentage of
     *                    contrast: the distance differs per layer and per
     *                    stock because the bands do.
     *  5  MIN            -1.0, the lower acceptance edge
     *  6  MAX            +1.0, the upper acceptance edge
     *  7  DEFAULT        0.0 -- the curve as the database stores it
     *  8  STEP           0.05
     *  9  PURPOSE        Renders the roll the SPECIFICATION allows rather than
     *                    only the roll the database stores. Ten stocks in this
     *                    database are specified by a manufacturing acceptance
     *                    band instead of by typical data -- the nine Soviet ТУ
     *                    films and the one ГОСТ film -- and a «Технические
     *                    условия» prints limits, not measurements: «не менее
     *                    100» means a roll testing at 99 is rejected and says
     *                    nothing about what a good roll did. The profile can
     *                    hold one number per layer, so it holds the mid-point
     *                    of a two-sided band or the WORST LEGAL EXAMPLE where
     *                    the norm is one-sided. This control reaches the rest
     *                    of the band.
     *                    \warning IT IS NOT A CONTRAST CONTROL. It moves each
     *                    layer by a different amount, only as far as that
     *                    film's own document allows, and not at all on a stock
     *                    whose manufacturer published typical values.
     * 10  OUTPUT EFFECT  Moves the per-layer characteristic-curve gamma, and
     *                    the per-layer D_min where the source prints both
     *                    edges of a band for it. 0.0 reproduces pre-field
     *                    renders BIT FOR BIT: the base profile is returned by
     *                    reference and nothing is copied.
     *                    \warning WHAT IT DOES NOT MOVE IS A LIST OF REFUSALS,
     *                    not a backlog. Speed: the bands state an acceptance
     *                    window but no stage reads `exposure_index` as an
     *                    exposure, so moving it would change a reported number
     *                    and not a pixel -- `exposureStops` is the honest
     *                    control for that. Granularity, MTF and reversal
     *                    D_max: their bands are one-sided and the stored value
     *                    already sits at the limit, so there is nothing to
     *                    interpolate toward. Ageing: that is a cupboard, not a
     *                    coating, and it belongs to `storageYears`.
     * 11  STAGES         resolved in frame setup, after the variant, the
     *                    development time and the storage age, before the
     *                    anchor solve. Read downstream by every stage that
     *                    takes the profile, so they all see one film.
     * 12  INTERACTIONS   Compounds with `developmentMinutes`, and the two are
     *                    different axes: development time moves the whole
     *                    emulsion along its own time-gamma family, batch
     *                    position moves it within the spread the factory was
     *                    allowed to ship. A stock can have both.
     * 13  SCALAR/AVX2    External semantics identical, and there is no vector
     *                    code at all: resolution is a frame-setup step with no
     *                    pixel loop in either path, so the two engines run the
     *                    same header.
     * 14  FULL/LITE      Both, at full quality: one profile copy per frame,
     *                    and only when the control is off its default on a
     *                    stock that carries a band.
     */
    double batchPosition;

    /**
     *  1  NAME           developmentMinutes
     *  2  TYPE           double
     *  3  AE CONTROL     slider, and it MUST HIDE OR DISABLE ITSELF on a stock
     *                    that carries no time-gamma family -- same rule
     *                    processVariant already follows for an empty variant
     *                    list. ⚠ THAT IS 187 OF 231 STOCKS TODAY. Showing a
     *                    live slider that cannot move a pixel is worse than
     *                    showing none, because the user then reads the absence
     *                    of change as a broken effect rather than as absent
     *                    data. Suggested label "Development time", with the
     *                    stock's own stated time shown as the default readout.
     *  4  UNIT           MINUTES, absolute, not a multiplier. The stored
     *                    families are printed in minutes against gamma
     *                    (`ProcessingFamily.points[].minutes`), so an absolute
     *                    axis interpolates them directly; a multiplier would
     *                    need a reference time that 102 of the 189 stocks with
     *                    a developer do not state
     *  5  MIN            -1.0 as the sentinel. Effective minimum is the
     *                    SELECTED STOCK'S OWN shortest traced point, which
     *                    ranges 1.9 min (AGFA APX 25) to 10.0 min (KODAK
     *                    VERICHROME 1952). NOT a fixed number -- it is a
     *                    per-stock bound, like processVariant's MAX
     *  6  MAX            the selected stock's LONGEST traced point: 7.0 min
     *                    (PANATOMIC-X) to 36.0 min (VERICHROME). ⚠ DO NOT
     *                    EXTRAPOLATE past either end. Outside the traced range
     *                    the value is treated as the sentinel rather than
     *                    clamped, for processVariant's reason: a stale preset
     *                    pointing outside the new stock's range must render the
     *                    stock AS SHIPPED, not render its nearest endpoint
     *  7  DEFAULT        -1.0 -- the development the stored curves represent,
     *                    i.e. `profile.processing.minutes` where stated.
     *                    Reproduces every pre-field render BIT FOR BIT
     *  8  STEP           0.1 min [proposed]. The traced families resolve to
     *                    about 0.1 min; APX 25 carries 62 points over
     *                    1.9-16.9 min, which is finer than any slider needs
     *  9  PURPOSE        Selects a DIFFERENT DEVELOPMENT TIME of the same
     *                    emulsion in the same developer. This is the axis the
     *                    stored curve sits on: every characteristic curve in
     *                    this database is one development, and until now that
     *                    development was unnamed and unreachable.
     * 10  OUTPUT EFFECT  Interpolates gamma along the stock's own traced
     *                    time-gamma family and rebuilds the characteristic
     *                    curve from it, before anything reads a curve --
     *                    exactly where processVariant already substitutes one.
     *                    ⚠⚠ IT IS A DELIBERATE NO-OP WHERE THERE IS NO FAMILY,
     *                    AND THAT IS NOT A LIMITATION TO BE ENGINEERED AROUND.
     *                    ⚠ 30 of 191 stocks carry a family as of 2026-09-17b
     *                    and 22 of those carry gamma values -- up from 12 and
     *                    9, because queue P61 traced the 1956 Data Book's
     *                    time-gamma insets onto four existing stocks and
     *                    queue P63 added five more stocks that each arrive
     *                    with the family Kodak prints on their own sheet.
     *                    16 of the 30 pass `development_family`'s own
     *                    requirement of two or more gamma-bearing points
     *                    sharing one (developer, dilution, vessel, edition):
     *                      AGFA_APX_25 / _100 / _400   62 / 64 / 73 points,
     *                                                  gamma 0.55-0.75
     *                      EASTMAN_DOUBLE_X_5222        5 points, 0.50-1.05
     *                      KODAK_ORTHO_X_SHEET_1952     5 points, 0.60-1.00
     *                      KODAK_PANATOMIC_X_SHEET_1952 5 points, 0.60-1.00
     *                      KODAK_TRI_X_SHEET_1952       5 points, 0.60-1.00
     *                      KODAK_VERICHROME_1952        5 points, 0.60-1.00
     *                      SUPER_ANSCOCHROME_1957       4 points, 3.55-4.57
     *                    The remaining three families (FUJI_NEOPAN_1600,
     *                    ILFORD_PAN_F, SVEMA_DS_5M) carry times and contrast
     *                    index but NO gamma, so they must behave as no-family
     *                    stocks until a gamma is traced. Ten of the twelve are
     *                    monochrome, so this control is very nearly a
     *                    black-and-white feature as the data stands.
     *                    ⚠ A GAMMA SCALE APPLIED TO A STOCK WITH NO FAMILY
     *                    WOULD BE AN INVENTED EFFECT. The relation is not
     *                    transferable between emulsions: over these nine
     *                    stocks the same time change buys anything from 0.20
     *                    of gamma (APX, 0.55-0.75) to 1.02 (SUPER
     *                    ANSCOCHROME, 3.55-4.57). There is no house curve.
     *                    ⚠ AND SUPER_ANSCOCHROME_1957 IS A KNOWN-BAD ROW.
     *                    Its family reads 3.55-4.57 and its stored curve gamma
     *                    is 5.27, which verify.py G-GAMMA pins as one of two
     *                    physically impossible curves awaiting a re-trace
     *                    (NotFound.md 2026-09-08b). The family CORROBORATES
     *                    the defect rather than excusing it -- both numbers
     *                    are far outside any real reversal film -- so this
     *                    control should stay disabled on that stock until the
     *                    re-trace lands, even though it has a family.
     * 11  STAGES         resolved in FRAME SETUP, before the anchor solve, for
     *                    processVariant's reason: stage 8, stage 11's grain
     *                    amplitude and stage 13's dupe chain all take the
     *                    profile and must see one consistent film
     * 12  INTERACTIONS   ⚠ MUTUALLY EXCLUSIVE WITH processVariant IN PRACTICE.
     *                    A selected variant already IS a different development
     *                    and carries its own traced curves; layering a time
     *                    change on top would apply the same physical change
     *                    twice. The host should disable this control whenever
     *                    processVariant != -1. Compounds with nothing else: it
     *                    picks the curve every other control is applied to.
     *                    ⚠ NOT to be combined with an exposureStops offset
     *                    meant to represent the same push -- a longer
     *                    development already carries its own speed change.
     * 13  SCALAR/AVX2    External semantics identical, and the resolution must
     *                    live in the SHARED header rather than in either twin,
     *                    like AlgoGrainAmpBuild: one interpolation, evaluated
     *                    once per frame in HighPrecType, so the two paths
     *                    cannot compute different curves. A per-twin copy of a
     *                    curve rebuild is exactly the divergence
     *                    cpp_parity.py's TWIN_LAW_TOKENS exists to catch.
     * 14  FULL/LITE      Both, at full quality: one profile copy and one
     *                    interpolation per frame, and only when the control is
     *                    off its sentinel.
     *
     *   SENTINEL: the test is strictly developmentMinutes < 0, so -1 means
     *   "use profile.processing.minutes" and any value inside the stock's
     *   traced range selects that development. ⚠ THE SENTINEL PATH MUST RETURN
     *   THE BASE PROFILE BY REFERENCE AND COPY NOTHING, so -1 reproduces
     *   pre-field renders bit for bit -- the contract processVariant already
     *   meets.
     *
     *   ⚠ WHAT THIS CONTROL DOES NOT DO, stated so it is not filed as a bug:
     *   it does not change base fog, it does not change granularity, and it
     *   does not change speed. All three move with development time in
     *   reality; none of the three has a traced relation in this corpus. The
     *   DevelopmentPoint record already has base_fog and exposure_index fields
     *   for the day they do.
     */
    double developmentMinutes;

    /**
     *  1  NAME           developmentCelsius
     *  2  TYPE           double
     *  3  AE CONTROL     slider, and it MUST HIDE OR DISABLE ITSELF unless the
     *                    selected stock's family actually varies temperature.
     *                    ⚠ THAT IS THREE STOCKS -- AGFA_APX_25, _100 and _400,
     *                    which trace 18-24 °C. Every other family in the
     *                    database was measured at ONE temperature (20 °C, or
     *                    21 °C for EASTMAN_DOUBLE_X_5222), so on those stocks
     *                    this axis has exactly one point and no slope.
     *                    Suggested label "Development temperature".
     *  4  UNIT           DEGREES CELSIUS, absolute. The sheets print °C for the
     *                    still films and °F for the Kodak cine process manuals;
     *                    the database stores °C (`ProcessingSpec.celsius`,
     *                    `DevelopmentPoint.celsius`) and the conversion belongs
     *                    in the panel, not here. ⚠ H-24's ECN-2 is 41.1 °C
     *                    (106 °F) and TB C41's C-41 is 37.8 °C ± 0.15 -- so a
     *                    colour process tolerance is a TENTH of a degree,
     *                    which is why this must never be a free slider on a
     *                    colour stock.
     *  5  MIN            -1.0 as the sentinel. Effective minimum is the
     *                    selected stock's own lowest traced temperature:
     *                    18 °C on the APX family. NOT a fixed number
     *  6  MAX            the stock's highest traced temperature: 24 °C on the
     *                    APX family. ⚠ DO NOT EXTRAPOLATE. Outside the traced
     *                    range the value is treated as the sentinel rather
     *                    than clamped, for the same reason as
     *                    developmentMinutes
     *  7  DEFAULT        -1.0 -- the temperature the stored curves represent,
     *                    i.e. `profile.processing.celsius` where stated (82 of
     *                    191 stocks). Reproduces every pre-field render BIT
     *                    FOR BIT
     *  8  STEP           0.5 °C [proposed] for the monochrome families, whose
     *                    traced span is 6 °C. ⚠ A colour process would need
     *                    0.05 °C to respect C-41's stated ± 0.15, which is
     *                    another reason this control has no business being
     *                    enabled on a colour stock today
     *  9  PURPOSE        Selects a different development TEMPERATURE of the
     *                    same emulsion in the same developer, the second axis
     *                    of the same physical change developmentMinutes moves.
     *                    Kept SEPARATE from that control rather than folded
     *                    into one "development" slider, because the traced
     *                    families are two-dimensional -- the APX sheets print
     *                    a time axis at each of several temperatures -- and
     *                    collapsing them would discard half the measurement.
     * 10  OUTPUT EFFECT  Selects which time-gamma curve of the family the
     *                    time axis is then read along, and rebuilds the
     *                    characteristic curve. ⚠ CORRECTED 2026-10-01: IT ACTS
     *                    ON 5 STOCKS (PLUS-X 125, T-MAX 400, T-MAX P3200,
     *                    TRI-X 400TX, VERICHROME PAN) -- those carrying both a fitted
     *                    ProcessingFamily::temperature_coeff_per_c (schema
     *                    v39, the stock's OWN rows) and a gamma-bearing time
     *                    family -- and only together with developmentMinutes.
     *                    film_params_mask.hpp bit 7 now says so; it read 0
     *                    everywhere, which hid a live control. On every other
     *                    stock it is a DELIBERATE NO-OP, because its family
     *                    holds one temperature or none. That is the honest behaviour:
     *                    time-temperature equivalence charts DO exist for real
     *                    developers, but none has been adopted here, and
     *                    substituting a published chart from another developer
     *                    would be an invented effect of exactly the kind
     *                    method rule 23 forbids.
     * 11  STAGES         resolved in FRAME SETUP with developmentMinutes, in
     *                    the same step and from the same family, so the pair
     *                    cannot disagree about which development was selected
     * 12  INTERACTIONS   ⚠ NOT INDEPENDENT OF developmentMinutes -- the two
     *                    are coordinates on ONE traced surface and must be
     *                    resolved together. Reading temperature first and time
     *                    second against a different family row is the obvious
     *                    way to get this wrong. Also mutually exclusive with
     *                    processVariant, for that control's reason.
     *                    ⚠ It does NOT interact with processingQuality, which
     *                    is a FilmDamage control describing bath agitation and
     *                    mottle. Different family, different physics: quality
     *                    is surge marks and air bells, this is gamma.
     * 13  SCALAR/AVX2    External semantics identical; the resolution is the
     *                    same shared frame-setup step as developmentMinutes,
     *                    for the same anti-divergence reason
     * 14  FULL/LITE      Both, at full quality: frame setup only, no pixel
     *                    loop in either path.
     *
     *   SENTINEL: strictly developmentCelsius < 0 means "use
     *   profile.processing.celsius". The base profile is returned by reference
     *   and nothing is copied, so -1 is bit-exact.
     *
     *   ⚠ WHY BOTH CONTROLS EXIST AT ALL WHEN SO FEW STOCKS CAN USE THEM.
     *   82 of 191 stocks now state a developer, a time and a temperature --
     *   45 of them from a PROCESS MANUAL rather than a film datasheet (KODAK
     *   H-24 for ECN-2, FUJIFILM TB C41 for C-41). That block of data has no
     *   consumer at all today and is the largest populated-but-unread field
     *   group in the database after the temporal chain. Declaring the two
     *   controls now, sentinel-defaulted and hidden where they cannot act,
     *   means the panel and every serialised preset can be built ONCE against
     *   a stable structure -- the same reason the then-unconsumed FilmDamage
     *   controls were left in the layout (all consumed since 2026-10-10). What is missing is the traced
     *   time-gamma relation on the other 172 stocks, and that is a research
     *   item, not a code one.
     */
    double developmentCelsius;

    /**
     *  1  NAME           developerIndex
     *  2  TYPE           DeveloperCtrl (enum class, int32_t underlying) --
     *                    a LIST BOX value since 2026-10-16f
     *  3  AE CONTROL     popup, items DeveloperCtrlStr (AlgoControlEnums.hpp);
     *                    item n (1-based) <-> DeveloperCtrl(n - 1). Grey the
     *                    entries the selected stock was not measured in
     *                    (AlgoDeveloperRowFromCtrl returns -1), and hide or
     *                    disable the whole control on a stock with no rows --
     *                    226 of 230.
     *  4  UNIT           one developer, by name
     *  5  MIN            eAS_SHIPPED (0): the developer the stored profile
     *                    represents (grain_reference_developer)
     *  6  MAX            DeveloperCtrlCount - 1; any value outside the range
     *                    is clamped to eAS_SHIPPED
     *  7  DEFAULT        eAS_SHIPPED. Reproduces every pre-v60 render BIT FOR
     *                    BIT
     *  8  STEP           1 (list item)
     *  9  PURPOSE        Develops the film in a different developer, from one
     *                    laboratory's measurements of THAT film in each
     *                    (Bernhard W. Schmidt: T-MAX 100, T-MAX 400, Tri-X,
     *                    Pan F).
     * 10  OUTPUT EFFECT  rms granularity x rms_d[dev]/rms_d[ref] (grain_um
     *                    kept, so the spectrum keeps its shape);
     *                    MTFSpec::adjacency_um x halo[dev]/halo[ref] (the
     *                    development-halo width at 8 lp/mm). RATIOS ONLY: the
     *                    source's rms scale is not Kodak's 48 um scale. Speed
     *                    and contrast are NOT moved.
     * 11  STAGES         resolved in FRAME SETUP straight after
     *                    developmentMinutes (AlgoDeveloperRowFromCtrl ->
     *                    AlgoResolveDeveloper); read by stage 06 (adjacency)
     *                    and stage 11 (grain)
     * 12  INTERACTIONS   Composes with developmentMinutes: both scale rms by a
     *                    factor, and the time's gamma law is the REFERENCE
     *                    developer's -- an approximation the panel should not
     *                    hide. Mutually exclusive with processVariant in
     *                    practice, for that control's reason.
     * 13  SCALAR/AVX2    External semantics identical; one shared frame-setup
     *                    resolver in AlgoDevelopmentTime.hpp.
     * 14  FULL/LITE      Both: one profile copy per frame when a developer
     *                    other than the reference is chosen.
     *
     *   ⚠ CHANGED 2026-10-16f (owner request): until then this field was an
     *   int32_t ROW INDEX into the stock's own grain_points with -1 as the
     *   sentinel, which a list box cannot fill (row n named a different
     *   developer on different stocks). It is now the global enumeration,
     *   so the host must assign a DeveloperCtrl; an int assignment no longer
     *   compiles, deliberately.
     */
    DeveloperCtrl developerIndex;

    /**
     *  1  NAME           storageYears
     *  2  TYPE           double
     *  3  AE CONTROL     slider, and it MUST HIDE OR DISABLE ITSELF on a stock
     *                    carrying no published dark-fade rate -- the same rule
     *                    processVariant and developmentMinutes follow. ⚠ THAT
     *                    IS 188 OF 191 STOCKS TODAY.
     *  4  UNIT           years of dark storage since processing
     *  5  MIN            0, which is also the OFF sentinel and means fresh
     *  6  MAX            100 -- see the note in AlgoControlEnums.hpp: a bound
     *                    on where a first-order fade stops restating the
     *                    published figure, not a physical limit
     *  7  DEFAULT        0
     *  8  STEP           0.5
     *  9  PURPOSE        Ages the image dyes the way dark storage does. The
     *                    database publishes a time to a 10 % loss from density
     *                    1.0 for the LEAST STABLE dye; this control turns that
     *                    rate into a state.
     * 10  OUTPUT EFFECT  Scales ToneCurve::gamma on the record belonging to
     *                    each faded dye -- cyan in red, magenta in green,
     *                    yellow in blue -- before anything reads a curve.
     *                    ⚠⚠ ONLY THE DYE THE SOURCE NAMES FADES, which on both
     *                    stocks is yellow, so the visible result is a NEGATIVE
     *                    THAT HAS LOST BLUE DENSITY and prints warm. That
     *                    asymmetry is the effect: Wilhelm's whole subject is
     *                    that the three dyes fade at different rates and the
     *                    differential is what makes a faded negative
     *                    uncorrectable. Fading all three together would be a
     *                    density change wearing the costume of one.
     *                    ⚠ dmin IS NOT TOUCHED. Part of a masked negative's
     *                    D-min is orange mask, which fades, and part is
     *                    support, which does not; ToneCurve stores the sum and
     *                    nothing separates them. A faded negative rendered
     *                    here keeps all of its mask.
     * 11  STAGES         resolved in FRAME SETUP, after processVariant and
     *                    developmentMinutes -- the order is chronological,
     *                    since storage happens after processing
     * 12  INTERACTIONS   Compounds with both of the other profile resolvers
     *                    and with nothing else.
     * 13  SCALAR/AVX2    External semantics identical; the resolution lives in
     *                    the shared header AlgoStorageAge.hpp and is evaluated
     *                    once per frame in HighPrecType, so the two paths
     *                    cannot compute different curves.
     * 14  FULL/LITE      Both: one profile copy per frame, and only when a
     *                    stock with a published rate is aged.
     *
     *   SENTINEL: storageYears <= 0 is fresh film; the base profile is
     *   returned by reference and nothing is copied, so 0 reproduces
     *   pre-field renders bit for bit.
     *
     *   ⚠ WHAT THE NUMBER DOES NOT CARRY, stated because the source states it:
     *   the published years are a 40 % RH figure that HALVES at 60 % RH, and
     *   they count dye fading only -- the yellowish stain that usually becomes
     *   visible before the 10 % dye loss does is not modelled at all. There is
     *   also no storage-temperature control: the source gives factors at two
     *   refrigerator temperatures and three points do not define a continuous
     *   law.
     *   ⚠ SUPERSEDED 2026-09-28b (schema v57): Wilhelm's Table 5.3 prints ten
     *   points from -26 to +30 degC that are one Arrhenius law, and the
     *   temperature is now the separate control storageCelsius below. The
     *   humidity caveat and the stain caveat above still stand.
     */
    double storageYears;

    /**
     *  1  NAME           storageCelsius
     *  2  TYPE           double
     *  3  AE CONTROL     slider, shown and enabled EXACTLY WHEN storageYears
     *                    is -- it only restates the elapsed years, so on a
     *                    stock with no published dark-fade rate, or at 0
     *                    years, it moves nothing
     *  4  UNIT           degrees Celsius, the temperature the film was kept at
     *  5  MIN            -26 -- the coldest point Wilhelm's Table 5.3 prints
     *  6  MAX            +30 -- the warmest point it prints. The resolver HOLDS
     *                    the end factors outside the table rather than
     *                    extrapolating, so a value beyond either end renders
     *                    as that end
     *  7  DEFAULT        24 -- 75 degF, the reference temperature of every
     *                    stored record, so the default is the identity and
     *                    reproduces pre-field renders bit for bit
     *  8  STEP           0.5
     *  9  PURPOSE        Says WHERE the years in storageYears were spent. A
     *                    film kept in a refrigerator ages far more slowly
     *                    than one on a shelf; Table 5.3 puts 4 degC at 16x
     *                    and -18 degC at 340x the room-temperature life.
     * 10  OUTPUT EFFECT  None by itself. Converts storageYears into the
     *                    equivalent years at the record's reference
     *                    temperature -- years * F(ref) / F(celsius), F read
     *                    from Table 5.3 with ln F linear in 1/T -- before the
     *                    first-order fade is evaluated. ⚠ IT IS KODAK'S
     *                    RELATION for Kodak's chromogenic dyes (Bard et al.
     *                    1980); applying it to the two Fuji negatives with a
     *                    rate is a transfer the source does not make, and
     *                    their provenance says so. ⚠ RELATIVE HUMIDITY IS NOT
     *                    A CONTROL: the only published humidity factor is
     *                    three points for Kodak yellow dyes.
     * 11  STAGES         resolved in FRAME SETUP inside AlgoResolveStorageAge,
     *                    with storageYears
     * 12  INTERACTIONS   Only with storageYears, and it cannot act without it.
     * 13  SCALAR/AVX2    External semantics identical; the table and the
     *                    interpolation live in the shared header
     *                    AlgoStorageAge.hpp and run once per frame in
     *                    HighPrecType.
     * 14  FULL/LITE      Both: frame setup only, no pixel loop.
     *
     *   ⚠ 0 IS NOT A SENTINEL. 0 degC is a real storage temperature, 28x
     *   slower than room temperature, so getAlgoControlsDefault() assigns 24
     *   explicitly; a zero-initialised AlgoControls would render a
     *   refrigerated film.
     */
    double storageCelsius;

    /**
     *  1  NAME           filmAgeMonths
     *  2  TYPE           double
     *  3  AE CONTROL     slider, and it MUST HIDE OR DISABLE ITSELF on a stock
     *                    whose standard prints no ageing bound -- the rule
     *                    storageYears follows. ⚠ THAT IS 214 OF 231 STOCKS:
     *                    only the 17 Soviet ТУ / ГОСТ stocks carry one
     *                    (film_params_mask.hpp bit eCTRL_BIT_FILM_AGE_MONTHS).
     *  4  UNIT           months between manufacture and exposure (the age of
     *                    the UNEXPOSED stock -- not storage after processing,
     *                    which is storageYears)
     *  5  MIN            0, which is also the OFF sentinel and means fresh
     *  6  MAX            600 (FilmAgeMonthsMax) -- a slider range; the model is
     *                    held at the standard's bound beyond the guarantee
     *  7  DEFAULT        0
     *  8  STEP           1
     *  9  PURPOSE        'Expired film'. Renders a roll as old as the stated
     *                    months, bounded by the only raw-stock ageing data in
     *                    the corpus: the acceptance bound a standard prints for
     *                    the END of the guarantee period.
     * 10  OUTPUT EFFECT  Fraction a = 0 through drift_free_months, linear to 1
     *                    at guarantee_months, held at 1 beyond. Speed: a
     *                    log10(1 - L a) shift of the emulsion's exposure, added
     *                    to the reciprocity shift before stage 8. Negative:
     *                    D-min + drift a in every record. Reversal: D-max -
     *                    |drift| a with D-min held (gamma scaled). ⚠ THE RAMP IS
     *                    A DEFINITION -- the standard fixes the end points only
     *                    -- and nothing past the guarantee is extrapolated.
     * 11  STAGES         FRAME SETUP (AlgoResolveFilmAge, after the developer
     *                    resolver) and the per-frame log-exposure constants
     *                    before stage 8 (AlgoFilmAgeLogShift)
     * 12  INTERACTIONS   Compounds with the development controls (gamma and
     *                    D-min changes commute) and with exposureTimeS (the two
     *                    log shifts add). Independent of storageYears, which
     *                    ages the PROCESSED image.
     * 13  SCALAR/AVX2    External semantics identical; the shared header
     *                    AlgoFilmAge.hpp runs once per frame in HighPrecType.
     * 14  FULL/LITE      Both: frame setup only, no pixel loop.
     *
     *   SENTINEL: filmAgeMonths <= 0 is fresh stock; the base profile is
     *   returned by reference and the shift is exactly 0, so 0 reproduces
     *   pre-field renders bit for bit. Added 2026-10-16 (#209); zero-init is
     *   already the sentinel, and getAlgoControlsDefault() assigns it anyway.
     */
    double filmAgeMonths;

    /**
     *  1  NAME           scannerSpecular
     *  2  TYPE           double
     *  3  AE CONTROL     slider
     *  4  UNIT           dimensionless FRACTION on 0..1 -- how specular the
     *                    scanner's illumination is. 0 is fully diffuse, 1 fully
     *                    specular
     *  5  MIN            0
     *  6  MAX            1
     *  7  DEFAULT        0.796   (AlgoControl.cpp, getAlgoControlsDefault) --
     *                    1 - E with E = 0.204 fitted to Streiffert's PRINTED
     *                    table (Trumpy & Flueckiger 2019 Fig. 3; 0.853 = 1 -
     *                    0.1471 from the 2015 drawing until 2026-10-26c). ⚠
     *                    RAISED FROM 0 ON 2026-09-06 BY OWNER DECISION, and it
     *                    moves pixels: a monochrome stock reads Q 1.429 at net
     *                    density 1.0 (1.485 at 0.853). ⚠ It is a DENSITOMETER geometry adopted
     *                    as a provisional stand-in, NOT a measurement of any
     *                    scanner. 0 remains the setting that reproduces the
     *                    stored (diffuse) characteristic curves exactly
     *  8  STEP           0.01 [proposed] -- a 0..1 axis with 100 positions
     *  9  PURPOSE        Drives the Callier coefficient. A silver image
     *                    scatters light, so its measured density depends on
     *                    whether the densitometer -- or the scanner -- collects
     *                    the scattered light or only the directly transmitted
     *                    beam. Specular readings come out HIGHER than diffuse
     *                    ones, by the Callier Q factor, and the effect is
     *                    strong on silver black-and-white and weak to absent on
     *                    dye images, which scatter far less.
     * 10  OUTPUT EFFECT  Raises effective contrast on silver images as the
     *                    setting moves toward specular. 0 reproduces the
     *                    pre-Callier behaviour exactly, which is the contract
     *                    the field was added under -- but 0 is NO LONGER THE
     *                    DEFAULT, so a render at defaults is no longer
     *                    byte-identical to a pre-Callier one on monochrome.
     * 11  STAGES         12b (Callier)
     * 12  INTERACTIONS   Reads the profile's callier_q. Its effect is inherently
     *                    stock-dependent: a dye-only image has little silver
     *                    left to scatter, so the same setting does much less on
     *                    a colour negative than on a black-and-white stock.
     * 13  SCALAR/AVX2    External semantics identical.
     * 14  FULL/LITE      PENDING. Expected both -- stage 12b is pointwise.
     *
     * !! SAME CAVEAT AS exposureTimeS: this field is absent from the 2026-08-27
     *    source archive this pass was performed against, so items 5, 6, 7, 11
     *    and 12 come from the project handoff's control table rather than from
     *    the consuming stage. UNVERIFIED until stage 12b is read.
     *
     * !! RELATED OPEN ITEM: film::FilmProfile::callier_q was recorded elsewhere
     *    in the project as "dead data". If stage 12b consumes it, that note is
     *    now wrong and should be corrected; if 12b uses something else, the
     *    field/stage relationship needs re-verifying. Recorded, not resolved.
     */
    double scannerSpecular;

    /**
     *  1  NAME           scannerFixedPattern
     *  2  TYPE           double
     *  3  AE CONTROL     slider
     *  4  UNIT           dimensionless FRACTION on 0..1 -- the share of grain
     *                    VARIANCE that is frame-locked scanner fixed-pattern
     *                    noise rather than emulsion grain
     *  5  MIN            0
     *  6  MAX            1
     *  7  DEFAULT        0.0   (AlgoControl.cpp, getAlgoControlsDefault) --
     *                    a pure emulsion field, which is the film behaviour and
     *                    the pre-v49 path bit for bit
     *  8  STEP           0.01
     *  9  PURPOSE        Separates the two statistically distinct things that
     *                    look like grain in a scanned motion picture. EMULSION
     *                    grain is different silver on every frame, because
     *                    every frame is a different piece of film; SENSOR
     *                    fixed-pattern noise is identical on every frame it
     *                    digitises. Before v49 the engine modelled only the
     *                    first, and modelled it as fully independent frame to
     *                    frame, which is right for the emulsion and leaves the
     *                    scanner unrepresented.
     * 10  OUTPUT EFFECT  At 0 the grain field decorrelates completely between
     *                    frames. At 1 it is the SAME field on every frame. In
     *                    between, the frame-to-frame correlation of the grain
     *                    equals this value, flat across all lags -- which is
     *                    what distinguishes it from the emulsion's
     *                    grain_frame_correlation, whose correlation DECAYS
     *                    with lag. Total grain variance is unchanged at every
     *                    setting.
     * 11  STAGES         8 (grain), via make_grain_field / temporal_white
     * 12  INTERACTIONS   Combines with film::TemporalSpec::grain_frame_correlation
     *                    as sqrt(1-f)*E + sqrt(f)*S with E and S independent
     *                    and unit variance, so neither control can change how
     *                    MUCH grain there is -- only how it behaves in time.
     * 13  SCALAR/AVX2    External semantics identical; the kernel weights are
     *                    computed from the same closed form in both.
     * 14  FULL/LITE      Both. The composition is pointwise on the white field.
     *
     * !! A CONTROL AND NOT A DATABASE COLUMN, DELIBERATELY. Fixed-pattern noise
     *    is a property of the INSTRUMENT: the same negative on two scanners
     *    carries two different patterns and on none carries neither. A column
     *    on FilmProfile would state that a 1936 emulsion has a sensor -- the
     *    same category error corrected for callier_q under queue C22.
     *
     * !! DEFAULT 0 BECAUSE THE CORPUS HOLDS NO MEASUREMENT. Queue M1b/P88 found
     *    no scanner spectral response in 177 documents and the one candidate --
     *    a reflective document scanner's mathematically recovered responsivity
     *    -- was refused on three grounds. The mechanism ships built and inert,
     *    which is this project's standard treatment for a law whose data has
     *    not arrived.
     */
    double scannerFixedPattern;

    /**
     *  1  NAME           sceneKelvin
     *  2  TYPE           double
     *  3  AE CONTROL     slider, or a colour-temperature control if the panel
     *                    offers one
     *  4  UNIT           KELVIN, absolute thermodynamic temperature. Confirmed
     *                    by the code: the value appears in the denominator of
     *                    Planck's c2/(lambda T) term, which is dimensionless
     *                    only if the field is in kelvin
     *  5  MIN            2000 advisory - NOT ENFORCED as a clamp. The spectral
     *                    path has a positivity guard that redirects to the
     *                    proxy path; THE PROXY PATH HAS NO GUARD, and a zero or
     *                    negative value there divides by zero in the Planck
     *                    term. The host must not pass <= 0
     *  6  MAX            12000 advisory - NOT ENFORCED
     *  7  DEFAULT        5500.0   (AlgoControl.cpp, getAlgoControlsDefault) -
     *                    nominal daylight
     *  8  STEP           50 K [proposed] - around 5500 K a 50 K step is below
     *                    the just-noticeable chromatic difference, so the
     *                    control reads smooth; 1 K would need 10,000 drag
     *                    positions to cross the range
     *  9  PURPOSE        The colour temperature of the light on the subject.
     * 10  OUTPUT EFFECT  Combined with the stock's balance_kelvin it produces
     *                    the per-layer exposure mismatch: daylight on tungsten
     *                    stock goes blue because of Planck's law, not because
     *                    anything is tinted. The resulting gains are normalised
     *                    so green is exactly 1.0, which deliberately prevents
     *                    this stage from doubling as an exposure control and
     *                    fighting the anchor solve.
     * 11  STAGES         03 only
     * 12  INTERACTIONS   GATED ENTIRELY BY wbStrength: at the shipped default
     *                    wbStrength = 0 this field is never read at all.
     *                    Also disabled on every monochrome stock. Ratioed
     *                    against the profile's balance_kelvin.
     * 13  SCALAR/AVX2    Same semantics, and the Planck evaluation stays in
     *                    double in the AVX2 build - deliberately, because
     *                    blackbody radiance intermediates span roughly sixty
     *                    decades, which is beyond float.
     * 14  FULL/LITE      PENDING. Expected both, at full quality - stage 03 is
     *                    on the Lite keep-at-full-quality list because colour
     *                    balance is part of the film's identity.
     *
     * !! See the CCT/Duv note in the file header. This is the only
     *   colour-temperature axis that exists; there is no Duv component and no
     *   coarse/fine split.
     */
    double sceneKelvin;

    /**
     *  1  NAME           wbStrength
     *  2  TYPE           double
     *  3  AE CONTROL     slider
     *  4  UNIT           dimensionless INTERPOLATION COEFFICIENT. Confirmed by
     *                    the code: it multiplies a gain difference and is added
     *                    to 1, so at 0 the gain is exactly 1.0 and at 1 it is
     *                    the full computed gain. It is a lerp parameter, not a
     *                    strength in any physical sense
     *  5  MIN            0 - values <= 0 disable the stage by short circuit
     *                    rather than being clamped
     *  6  MAX            1 advisory - NOT ENFORCED. A value of 10 extrapolates
     *                    the gain ten times past the physical ratio
     *  7  DEFAULT        0.0   (AlgoControl.cpp, getAlgoControlsDefault) - no
     *                    correction, so a tungsten stock shot in daylight stays
     *                    blue, which is the physically honest default
     *  8  STEP           0.01 [proposed] - a 0..1 axis with 100 positions
     *  9  PURPOSE        How much of the stock/scene colour-temperature
     *                    mismatch to correct. 0 records it exactly as the stock
     *                    would; 1 renders as if the right conversion filter had
     *                    been on the lens. Intermediate values model a partial
     *                    correction, which is what an 80A on the wrong stock
     *                    actually looks like.
     * 10  OUTPUT EFFECT  At the default of 0, STAGE 03 IS A PURE COPY. This is
     *                    worth knowing before profiling or before concluding
     *                    that sceneKelvin does nothing.
     * 11  STAGES         03 only
     * 12  INTERACTIONS   It gates sceneKelvin completely. Itself disabled on
     *                    monochrome stocks - a gate that is load-bearing: when
     *                    it was omitted from a prototype the error on one
     *                    monochrome stock was 1.468, and was exactly zero at
     *                    the default wbStrength of 0, so it passed every casual
     *                    check.
     * 13  SCALAR/AVX2    Same semantics; the AVX2 build folds the blend into
     *                    three broadcasts instead of three scalars, same
     *                    expression.
     * 14  FULL/LITE      PENDING. Expected both, at full quality (stage 03).
     */
    double wbStrength;

    /**
     *  1  NAME           greyTarget
     *  2  TYPE           double
     *  3  AE CONTROL     slider
     *  4  UNIT           dimensionless DISPLAY-LINEAR VALUE on 0..1 - the value
     *                    an 18 per cent scene grey should land on. It is a
     *                    reflectance-like target, not a density and not a gain:
     *                    the solve treats it as the number the curve output
     *                    must equal
     *  5  MIN            0.02 advisory - NOT ENFORCED. Zero or negative flows
     *                    straight into a division and then into the bisection
     *                    solve
     *  6  MAX            0.60 advisory - NOT ENFORCED
     *  7  DEFAULT        0.18   (AlgoControl.cpp, getAlgoControlsDefault)
     *  8  STEP           0.005 [proposed] - the range is only 0.58 wide and the
     *                    tone scale is visibly sensitive to it; 0.005 gives
     *                    ~116 positions across the useful range
     *  9  PURPOSE        The tone-scale contract. The printer-light anchor
     *                    solve hits this value exactly, which is what makes it
     *                    a contract rather than a gain.
     * 10  OUTPUT EFFECT  Moves the whole tone scale by moving the anchor the
     *                    curve is solved against. Because the solve is exact,
     *                    changing this does not merely brighten - it re-places
     *                    the toe and shoulder relative to the scene.
     * 11  STAGES         08 and 08b via the anchor solve, and 13, which
     *                    re-solves against it because the dupe chain has moved
     *                    the neutral density
     * 12  INTERACTIONS   Enters the anchor solve jointly with couplerScale.
     *                    Depends on printStock resolving. Its stage 13 use is
     *                    skipped on reversal stocks and when no print stock
     *                    resolves. NOT connected to the ALGO_MID_GREY constant
     *                    used by exposureStops, despite the shared 0.18 value.
     * 13  SCALAR/AVX2    Same semantics. The anchor solve is HighPrecType
     *                    (double) in BOTH builds and is not vectorised in
     *                    either - a deliberate retention, not an oversight.
     * 14  FULL/LITE      PENDING. Expected both, at full quality - the tone
     *                    scale is the first thing the Lite design refuses to
     *                    approximate.
     */
    double greyTarget;

    /**
     *  1  NAME           blackPointStretch
     *  2  TYPE           double
     *  3  AE CONTROL     slider
     *  4  UNIT           dimensionless FRACTION on 0..1 - how much of the
     *                    stock's own Dmax is stretched down to output zero.
     *                    Stage 14 normalises transmittance as
     *                       out = (10^-D - s * tMin) / (tMax - s * tMin)
     *                    with tMax = 10^-Dmin, tMin = 10^-Dmax and s = this
     *                    control
     *  5  MIN            0.0 advisory - NOT ENFORCED, and nothing breaks below
     *                    it either: a negative value simply lifts the floor
     *                    further, which is a lift and not a film property
     *  6  MAX            1.0 advisory - NOT ENFORCED. Above 1.0 the black point
     *                    passes Dmax and the divisor shrinks toward zero; at
     *                    exactly tMax / tMin it divides by zero. The host
     *                    should hold the slider at 1.0
     *  7  DEFAULT        1.0   (AlgoControl.cpp, getAlgoControlsDefault)
     *  8  STEP           0.01 [proposed] - the range is 1.0 wide and the effect
     *                    is confined to the deepest shadows, so 100 positions
     *                    is ample
     *  9  PURPOSE        Decides whether a stock's Dmax is a RENDERED PROPERTY
     *                    or a normalisation constant. At 1.0 every stock's Dmax
     *                    becomes output zero, so a POLAROID's weak 1.6 D black
     *                    and VELVIA's 3.0 D black look identical. At 0.0 each
     *                    stock floors at its own 10^-(Dmax-Dmin) and Dmax is
     *                    visible for the first time.
     * 10  OUTPUT EFFECT  \warning AT 1.0 IT DESTROYS REAL SHADOW INFORMATION,
     *                    and that is measured rather than argued. On the
     *                    2026-09-09 VELVIA regression frame 13.06 per cent of
     *                    the BLUE record arrives at stage 14 at or above Dmax
     *                    and is mapped to exactly zero before the encoder sees
     *                    it. At 0.0 that is 0.00 per cent, the whole-frame mean
     *                    moves 0.7 of one 8-bit code, and red clipping is
     *                    untouched.
     *                    \warning IT ALSO MOVES ALL 191 STOCKS, by each stock's
     *                    own Dmax: VELVIA floors at 4.0/3.2/2.2 of 255,
     *                    negatives through SCAN_DI at 1.5, the POLAROID
     *                    materials at 33 to 46. 14 of the 44 reversal stocks
     *                    floor at or under 4 codes; the rest visibly lift.
     * 11  STAGES         14, and 08 / 08b / 13 through the anchor solve, which
     *                    MUST see the same value - the solve aims at a display
     *                    number that stage 14 produces, so if the two
     *                    expressions disagree a neutral stops landing where it
     *                    was solved to
     * 12  INTERACTIONS   Enters the anchor solve jointly with greyTarget.
     *                    \warning AND THE ONE MEASURED COST THE HOST SHOULD
     *                    SURFACE: at 0.0 the 18 per cent mid-grey anchor misses
     *                    its 12 per cent bound on the four lowest-Dmax stocks -
     *                    POLAROID_410 0.1669, POLAROID_42 0.1631, POLAROID_47
     *                    0.1474, POLAROID_51 0.1429, against a worst of 0.0876
     *                    at 1.0. Traced to stage 12b: Callier multiplies a
     *                    monochrome patch's NET density by 1.62 (0.4992 ->
     *                    0.7591 D measured on POLAROID_42), so the scalar
     *                    solver's residual against the full pixel pass is
     *                    amplified by that factor and 0.0 no longer compresses
     *                    the result. That is solver accuracy on four stocks,
     *                    not a contradiction in the normalisation.
     *                    \warning WHAT IT IS NOT: a fix for the cause. 9.73 of
     *                    those 13.06 points come from stage 12's dye_matrix, a
     *                    unit-row-sum SATURATION operator that is not bounded
     *                    by the channel's own range - at Dmax with the other
     *                    two records low VELVIA's returns 4.14 D against a Dmax
     *                    of 3.31, which no curve can produce.
     * 13  SCALAR/AVX2    Same semantics and the same expression. tMin is folded
     *                    once per channel in both builds, so the vector twin
     *                    broadcasts an already-scaled constant and costs
     *                    nothing per pixel.
     * 14  FULL/LITE      PENDING. Expected both, at full quality - it is one
     *                    multiply on a per-channel constant.
     */
    double blackPointStretch;

    // -- effect scales: each multiplies a PROFILED value, never replaces it --

    /**
     *  1  NAME           grainScale
     *  2  TYPE           double
     *  3  AE CONTROL     slider
     *  4  UNIT           dimensionless multiplier on the stock's calibrated RMS
     *                    granularity. It scales added OPTICAL DENSITY: the
     *                    grain field is zero-mean and is added as
     *                    gain * field * sqrt(developed density + fog)
     *  5  MIN            0.0, ENFORCED at all three reading stages
     *  6  MAX            4.0 advisory - NOT ENFORCED
     *  7  DEFAULT        1.0   (AlgoControl.cpp, getAlgoControlsDefault) - the
     *                    stock's calibrated granularity, not an aesthetic
     *                    choice
     *  8  STEP           0.01 [proposed] - grain is judged at 100 per cent
     *                    zoom and small amplitude changes are visible, so a
     *                    coarse step would feel steppy
     *  9  PURPOSE        Grain amplitude. 0 disables grain entirely, which is
     *                    genuinely useful for isolating other stages.
     * 10  OUTPUT EFFECT  Scales the density noise added at stage 11, and the
     *                    per-generation dupe grain at stage 13.
     *
     *                    ASYMMETRY WORTH KNOWING: at stage 14 it is a GATE
     *                    ONLY. Print grain is added with a literal gain of 1
     *                    regardless of this control's value; grainScale > 0
     *                    merely permits it. So halving grainScale halves camera
     *                    and dupe grain but leaves print grain untouched.
     * 11  STAGES         11 (true multiplier), 13 (true multiplier), 14 (gate
     *                    only)
     * 12  INTERACTIONS   Hard-ANDed with printGrain and with the reversal test
     *                    at stage 14 - grainScale = 0 overrides printGrain =
     *                    true. Its stage 13 use needs generations > 0. Combined
     *                    with seed and frameIndex for the field realisation.
     *                    The grain field type depends on reseau through the
     *                    driver's mosaic decision.
     * 13  SCALAR/AVX2    Same semantics and identical gates. The scale is
     *                    broadcast as float in the AVX2 build.
     * 14  FULL/LITE      PENDING. Expected both, at full quality of KIND but
     *                    reduced in cost. Grain is the honest exception in the
     *                    Lite design: its character lives at Nyquist, so a
     *                    low-resolution grain field upsampled reads as
     *                    blotches. Lite must reduce grain by CHANNEL COUNT and
     *                    OCTAVE COUNT, never by resolution.
     * 15  TEMPORAL GRAIN, and why the DEFAULT IS A STILL-FRAME CALIBRATION.
     *                    Queue C7, closed 2026-09-02. Honjo 1989 section 4
     *                    states that at 24 fps the eye integrates over about
     *                    0.2 s, i.e. about five frames. Grain is re-rolled per
     *                    frame and is zero-mean, so five independent samples
     *                    average down by 1/sqrt(5) and the granularity a viewer
     *                    perceives in PLAYBACK is about 0.447 of what the same
     *                    emulsion shows in a frozen frame - this control's
     *                    default of 1.0 is therefore about 2.24x too strong in
     *                    motion, and that is deliberate.
     *
     *                    IT IS DELIBERATE BECAUSE EVERY GRANULARITY FIGURE THIS
     *                    ENGINE IS CALIBRATED AGAINST IS A STILL MEASUREMENT -
     *                    rms through a 48 um aperture, Wiener spectra, Selwyn
     *                    constants, all made on a stationary sample. A default
     *                    that silently divided them by 2.24 would stop
     *                    reproducing the numbers the calibration cites, and the
     *                    parity tests would then be checking against a quantity
     *                    no document states. It is also not reversible by a user
     *                    who does not know the rule was applied.
     *
     *                    A HOST THAT WANTS MOTION-CORRECT GRAIN NEEDS NOTHING
     *                    NEW: it sets grainScale to 1/sqrt(fps * 0.2), clamped
     *                    to [1, 8] frames, which film_sim.temporal_grain_scale()
     *                    computes. 0.4564 at 24 fps, 0.4472 at 25. No field was
     *                    added to FilmProfile and no stage changed, so this note
     *                    is the whole of C7 on the C++ side.
     */
    double grainScale;

    /**
     *  1  NAME           halationScale
     *  2  TYPE           double
     *  3  AE CONTROL     slider
     *  4  UNIT           dimensionless multiplier on the stock's per-channel
     *                    halation GAIN, which is a scatter-return fraction
     *  5  MIN            0.0, ENFORCED (MAX_VALUE floor at stage 05)
     *  6  MAX            4.0 advisory - NOT ENFORCED
     *  7  DEFAULT        1.0   (AlgoControl.cpp, getAlgoControlsDefault)
     *  8  STEP           0.01 [proposed]
     *  9  PURPOSE        Halation amplitude. Raising it toward 2 is a
     *                    legitimate response to a render that reads weaker than
     *                    a reference scan, because the profiled gain rests on
     *                    an assumed highlight overshoot rather than a measured
     *                    one.
     * 10  OUTPUT EFFECT  Scales the amplitude of the halo returned around
     *                    highlights.
     *
     * !! CORRECTION TO EARLIER DOCUMENTATION: this control scales the GAINS
     *   ONLY. It does NOT scale the three lobe radii - those are converted from
     *   micrometres through pxPerMm and are untouched by this field. Previous
     *   revisions of this header said it scaled both. Halo SIZE is not
     *   adjustable from the controls at all.
     * 11  STAGES         05 only
     * 12  INTERACTIONS   None with other controls. Disabled when every profile
     *                    channel gain is zero, when pxPerMm is zero, or when
     *                    every lobe radius falls below a quarter pixel.
     * 13  SCALAR/AVX2    Same semantics; float in the AVX2 build, double in
     *                    scalar.
     * 14  FULL/LITE      PENDING. Expected both, at REDUCED SAMPLING in Lite -
     *                    halation is one of the four irreducibly
     *                    neighbourhood-coupled operations, so Lite lowers the
     *                    blur pyramid engagement threshold rather than dropping
     *                    the stage.
     */
    double halationScale;

    /**
     *  1  NAME           couplerScale
     *  2  TYPE           double
     *  3  AE CONTROL     slider
     *  4  UNIT           dimensionless multiplier on the profile's coupler
     *                    STRENGTH terms. It scales amplitude only - the
     *                    diffusion distances radius_um and edge_um are
     *                    untouched
     *  5  MIN            0 - but enforced in TWO DIFFERENT SHAPES. At stage 09
     *                    the field itself is floored. In the tone path (08,
     *                    08b, 13) the PRODUCT strength * couplerScale is
     *                    floored instead, so a negative couplerScale against a
     *                    negative profile strength yields a positive product
     *                    that survives the floor
     *  6  MAX            3.0 advisory - NOT ENFORCED
     *  7  DEFAULT        1.0   (AlgoControl.cpp, getAlgoControlsDefault)
     *  8  STEP           0.01 [proposed]
     *  9  PURPOSE        The LATERAL half of the coupler chemistry - edge
     *                    effects and micro-contrast. The vertical half is
     *                    interimage, which is profile data with its own
     *                    iteration count and is not exposed here.
     * 10  OUTPUT EFFECT  Two effects that users will not expect to be one
     *                    control. At stage 09 it scales lateral DIR diffusion,
     *                    a local micro-contrast effect. At stages 08/08b/13 it
     *                    enters the ANCHOR SOLVE, so it also moves the printer
     *                    light and shifts overall tone. Changing it is not a
     *                    purely local edit.
     * 11  STAGES         09 directly; 08 and 08b through the anchor solve; 13
     *                    through the neutral mid density
     * 12  INTERACTIONS   Enters the anchor solve jointly with greyTarget.
     *                    Interacts with generations and dupeStock at stage 13.
     *                    Discarded entirely on monochrome stocks in the tone
     *                    path, and the long-range term is disabled on
     *                    monochrome at stage 09.
     * 13  SCALAR/AVX2    Same semantics. Note the SPLIT PRECISION, deliberate
     *                    in both builds: the 08/13 anchor path consumes it in
     *                    double even in the AVX2 build, while the stage 09
     *                    per-pixel path consumes it in float there.
     * 14  FULL/LITE      PENDING. Expected both. Its tone-path half is part of
     *                    the film's identity and must stay exact; its stage 09
     *                    spatial half is a candidate for reduced sampling.
     */
    double couplerScale;

    /**
     *  1  NAME           misregScale
     *  2  TYPE           double
     *  3  AE CONTROL     slider
     *  4  UNIT           dimensionless multiplier on the profile's
     *                    misregistration_um. The arithmetic is
     *                    um * (px/mm) * 0.001 * misregScale, giving PIXELS of
     *                    RMS channel displacement, used as the sigma of a
     *                    Gaussian draw
     *  5  MIN            0.0, ENFORCED (MAX_VALUE floor at stage 10)
     *  6  MAX            4.0 advisory - NOT ENFORCED
     *  7  DEFAULT        1.0   (AlgoControl.cpp, getAlgoControlsDefault)
     *  8  STEP           0.01 [proposed]
     *  9  PURPOSE        Channel misregistration. 0 gives perfect registration.
     * 10  OUTPUT EFFECT  Displaces the colour records relative to each other by
     *                    a per-frame random draw. Technicolor three-strip
     *                    profiles carry tens of micrometres here, which is why
     *                    their edges fringe.
     * 11  STAGES         10 only
     * 12  INTERACTIONS   None with other controls. Disabled entirely on
     *                    monochrome stocks. Every realised draw is applied as
     *                    film_sim's exact phase ramp (no minimum shift since
     *                    2026-10-06). Combined with seed and frameIndex, so
     *                    the displacement re-rolls every frame.
     * 13  SCALAR/AVX2    Same semantics; the draws stay in double in both
     *                    builds, the displacement in pixels is float in AVX2.
     * 14  FULL/LITE      PENDING. The shift rides in the scan MTF's
     *                    frequency-domain pass, so it adds no pass of its own.
     */
    double misregScale;

    /**
     *  1  NAME           flare
     *  2  TYPE           double
     *  3  AE CONTROL     slider
     *  4  UNIT           dimensionless FRACTION OF TOTAL LIGHT on 0..1 - an
     *                    energy split, NOT stops. Confirmed by the code: the
     *                    direct image is weighted (1 - flare) and the scattered
     *                    veil by flare, so the two sum to unity
     *  5  MIN            -1.0 as the sentinel; effective minimum 0 because any
     *                    resolved value <= 0 makes the stage a pure copy. NOT
     *                    otherwise clamped
     *  6  MAX            0.5 advisory - NOT ENFORCED, and exceeding 1.0 makes
     *                    the direct weight NEGATIVE, inverting the image.
     *                    The host must enforce this one
     *  7  DEFAULT        -1.0   (AlgoControl.cpp, getAlgoControlsDefault) -
     *                    i.e. use the stock's era-appropriate value
     *  8  STEP           0.005 [proposed] - the useful range is only 0 to 0.5
     *                    and the black floor moves visibly across it
     *  9  PURPOSE        Veiling flare fraction of the taking lens. 0 is a
     *                    perfect modern lens.
     * 10  OUTPUT EFFECT  Lifts the black floor and compresses global contrast.
     *                    Nothing in the emulsion model substitutes for it -
     *                    this is a lens property, and without it a period
     *                    render looks too clean in a way no emulsion parameter
     *                    can fix.
     * 11  STAGES         03b only
     * 12  INTERACTIONS   None with other controls. It runs before stage 04, so
     *                    its black-floor lift is subsequently multiplied by the
     *                    vignette and coating field.
     * 13  SCALAR/AVX2    Same semantics. The AVX2 composite uses FMA, so it
     *                    differs from the scalar multiply-then-add by rounding
     *                    only. The frame-mean veil this control multiplies is
     *                    accumulated in double in BOTH builds, deliberately -
     *                    it sets the black floor of the whole frame.
     * 14  FULL/LITE      PENDING. Expected both, at REDUCED SAMPLING in Lite -
     *                    03b is a pyramid blur and Lite lowers its engagement
     *                    threshold.
     *
     *   SENTINEL: the test is strictly flare < 0, so -1 means "use
     *   profile.default_flare" and 0.0 means "perfect lens, effect off". Note
     *   the sentinel path can itself disable the stage, if the selected stock's
     *   own default_flare is zero.
     */
    double flare;

    /**
     *  1  NAME           printGrain
     *  2  TYPE           bool
     *  3  AE CONTROL     checkbox
     *  4  UNIT           none - boolean switch
     *  5  MIN            false, by type
     *  6  MAX            true, by type
     *  7  DEFAULT        true   (AlgoControl.cpp, getAlgoControlsDefault)
     *  8  STEP           n/a
     *  9  PURPOSE        Whether the print stock contributes its own, finer,
     *                    grain on top of the camera negative's.
     * 10  OUTPUT EFFECT  Adds a second grain population after the print curve.
     *                    Its amplitude comes from the print stock's own
     *                    grain_rms at a fixed gain of 1 - see grainScale item
     *                    10 for that asymmetry.
     * 11  STAGES         14 only
     * 12  INTERACTIONS   One term of a five-way AND. The other four can
     *                    neutralise it: the stock must not be reversal,
     *                    grainScale must be > 0, a print stock must resolve,
     *                    and that print stock's grain_rms must be non-zero.
     *                    Combined with seed and frameIndex for the realisation.
     * 13  SCALAR/AVX2    Same semantics, identical five-way condition.
     * 14  FULL/LITE      PENDING. Expected both - stage 14 is on the Lite
     *                    keep-at-full-quality list.
     */
    bool printGrain;

    /**
     *  1  NAME           reseau
     *  2  TYPE           bool
     *  3  AE CONTROL     checkbox
     *  4  UNIT           none - boolean switch
     *  5  MIN            false, by type
     *  6  MAX            true, by type
     *  7  DEFAULT        true   (AlgoControl.cpp, getAlgoControlsDefault)
     *  8  STEP           n/a
     *  9  PURPOSE        Permits the additive-mosaic path on reseau stocks
     *                    (Dufaycolor, Lumiere Autochrome and relatives).
     * 10  OUTPUT EFFECT  Switches those stocks between a true mosaic record and
     *                    a plain three-record path. It auto-disables when the
     *                    render is too small to represent the grid: below
     *                    ALGO_RESEAU_MIN_PITCH_PX = 3 pixels per cell the
     *                    output would be aliasing noise rather than a mosaic.
     *
     * !! CORRECTION TO EARLIER DOCUMENTATION: the auto-disable is silent. The
     *   previous comment promised it "auto-disables and warns"; no warning is
     *   emitted anywhere - the stage simply falls through to the three-record
     *   copy. Either the warning should be added or the promise removed; this
     *   is recorded rather than decided.
     * 11  STAGES         07 (record formation), 14b (reconstruction), and 11
     *                    indirectly, because the driver's mosaic decision
     *                    selects the grain field type
     * 12  INTERACTIONS   Requires profile.has_reseau - on every other stock the
     *                    control does nothing at all. Gated by the pxPerMm
     *                    threshold above, so it interacts with filmFormat and
     *                    render size.
     * 13  SCALAR/AVX2    Same semantics; identical predicates in both trees.
     * 14  FULL/LITE      PENDING. Expected Full only - 14b is on the Lite drop
     *                    list. Note that dropping 14b while stage 07 still
     *                    forms a mosaic record would be wrong, so the two must
     *                    be decided together, not stage by stage.
     */
    bool reseau;

    // -- schema v4 / v5 additions -------------------------------------------

    /**
     *  1  NAME           vignette
     *  2  TYPE           double
     *  3  AE CONTROL     slider
     *  4  UNIT           STOPS of corner light loss. Confirmed by the code: the
     *                    corner cosine is raised to ALGO_VIGNETTE_EXPONENT = 4
     *                    and the setup solves so that the corner lands at
     *                    exactly 2^-stops
     *  5  MIN            -1.0 as the sentinel; effective minimum 0 because any
     *                    resolved value <= 0 disables the vignette half of
     *                    stage 04. NOT otherwise clamped
     *  6  MAX            4.0 advisory - NOT ENFORCED
     *  7  DEFAULT        -1.0   (AlgoControl.cpp, getAlgoControlsDefault) -
     *                    i.e. use the stock's era default
     *  8  STEP           0.05 [proposed] - a twentieth of a stop; corner
     *                    falloff is judged against the frame edge and finer
     *                    steps are not distinguishable
     *  9  PURPOSE        Lens corner falloff.
     * 10  OUTPUT EFFECT  Darkens the corners on a cos^4 geometry locked to the
     *                    frame.
     *
     *                    This is a LENS property, not an emulsion one, and the
     *                    distinction is load-bearing: cos^4 is geometry and
     *                    applies in every era - modern glass still loses
     *                    0.3-0.5 stop wide open. Coating unevenness cannot
     *                    produce a corner-locked defect at all, because film is
     *                    coated as a wide web and slit afterwards.
     * 11  STAGES         04 only
     * 12  INTERACTIONS   Shares stage 04 with coatingScale; the stage is
     *                    skipped only if BOTH are inactive, and the two
     *                    multiply into the same field plane. The vignette
     *                    itself is frame-invariant - only the coating half uses
     *                    frameIndex and seed.
     * 13  SCALAR/AVX2    !! THE ONE DOCUMENTED PRECISION DIFFERENCE IN THIS
     *                    STRUCTURE'S CONSUMERS. The sentinel and the setup
     *                    (cosCorner, tanCorner2) are double in both builds, but
     *                    the per-pixel falloff loop is HighPrecType (double) in
     *                    the scalar path and deliberately narrowed to AlgoType
     *                    (float) in the AVX2 path, annotated there as a Rule D1
     *                    alignment change. External semantics are identical;
     *                    the numerical difference is bounded by float epsilon
     *                    on a smooth field.
     * 14  FULL/LITE      PENDING. Expected both - a pointwise multiply, cheap
     *                    in either mode.
     *
     *   SENTINEL: the test is strictly vignette < 0, so -1 means "use
     *   profile.default_vignette" and 0.0 means "no vignette". As with flare, a
     *   stock whose own default is zero disables the effect through the
     *   sentinel path.
     *
     *   DO NOT CONFUSE with the float named vignette inside a different struct
     *   in film_profiles.hpp, which is documented there as "source's own scale
     *   - NOT stops" and is not read by stage 04.
     */
    double vignette;

    /**
     *  1  NAME           coatingScale
     *  2  TYPE           double
     *  3  AE CONTROL     slider
     *  4  UNIT           dimensionless multiplier on three different
     *                    CoatingSpec amplitudes: the standard deviation of a
     *                    multiplicative transmission field (dimensionless), a
     *                    blend weight (dimensionless 0..1), and an additive
     *                    optical density. In no case does it scale a LENGTH -
     *                    the correlation distances and the edge-fog width are
     *                    untouched
     *  5  MIN            0.0, ENFORCED at all three reading stages
     *  6  MAX            3.0 advisory - NOT ENFORCED on the control. One
     *                    product cap exists at stage 06b only, where
     *                    buckle_mtf_loss * coatingScale is capped at
     *                    ALGO_DEFOCUS_MAX_LOSS = 0.9; stages 04 and 10b have no
     *                    ceiling of any kind
     *  7  DEFAULT        1.0   (AlgoControl.cpp, getAlgoControlsDefault)
     *  8  STEP           0.01 [proposed]
     *  9  PURPOSE        Scales all three CoatingSpec defects together: the
     *                    web-coherent coating field, gate buckling, and
     *                    narrow-gauge edge fog. 0 disables them.
     * 10  OUTPUT EFFECT  A low-frequency multiplicative unevenness across the
     *                    frame (stage 04), corner softening from buckle (06b),
     *                    and an additive fog band at the film edge on narrow
     *                    gauges (10b).
     * 11  STAGES         04, 06b, 10b
     * 12  INTERACTIONS   Shares stage 04's output field with vignette. Combined
     *                    with seed and frameIndex for the web offset - so
     *                    setting coatingScale = 0 also removes the only
     *                    consumer of frameIndex in stage 04. Each of the three
     *                    stages is independently disabled when its own profile
     *                    amplitude is zero.
     * 13  SCALAR/AVX2    Same semantics. Note stage 04's coating-field
     *                    synthesis is held in DOUBLE even in the AVX2 build,
     *                    and the refusal to narrow it is explicit: the web
     *                    offset grows without bound along a clip, and in single
     *                    precision the low bits of such an argument are gone.
     *                    Stages 06b and 10b consume it in float in the AVX2
     *                    build.
     * 14  FULL/LITE      PENDING. Expected split: 04 both, 06b and 10b Full
     *                    only - both are on the Lite drop list. That means the
     *                    same control participates in Lite for one of its three
     *                    effects and not the other two, which must be stated in
     *                    the panel or the control will appear to misbehave in
     *                    Preview.
     */
    double coatingScale;

    /**
     *  1  NAME           frameIndex
     *  2  TYPE           int32_t
     *  3  AE CONTROL     none - host-supplied, not user-facing
     *  4  UNIT           FRAME COUNT, clip-relative, dimensionless integer. It
     *                    plays three arithmetic roles: multiplied by frame
     *                    pitch it is a LENGTH in mm of web position; divided by
     *                    frame rate it is a TIME; fed to the RNG it is pure
     *                    counter salt
     *  5  MIN            none enforced. See the contradiction note below
     *  6  MAX            none enforced. One unguarded signed multiplication
     *                    exists at stage 13 (frameIndex * 9), so extreme values
     *                    can overflow
     *  7  DEFAULT        0   (AlgoControl.cpp, getAlgoControlsDefault)
     *  8  STEP           1 - integral, host-supplied
     *  9  PURPOSE        Which frame of the clip is being rendered. Set it from
     *                    the layer time and the frame rate, NOT from a running
     *                    counter - the engine has no state and renders frames
     *                    out of order.
     * 10  OUTPUT EFFECT  Slides the coating field's machine-direction structure
     *                    by one frame pitch per frame, re-rolls the per-frame
     *                    random draws, and positions the film window for the
     *                    defect layer. The field is a pure function of (seed,
     *                    absolute web position), so frames render independently
     *                    and out of order with no seams.
     * 11  STAGES         03c (flicker time, since 2026-10-10), 04, 09b, 10, 11,
     *                    13, 14, 15, 16.
     * 12  INTERACTIONS   Paired with frameRate at every temporal stage. Folded
     *                    with seed in the RNG counter. Deliberately NOT mixed
     *                    into the stage 09b particulate field seed - mixing the
     *                    frame in there would re-roll every particle each frame
     *                    and turn film-locked dirt into boiling noise. Stage 16
     *                    does key on it, for the one-frame sparkle population.
     *                    Made inert by coatingScale = 0 at stage 04 and by
     *                    filmDamageEnabled = false at 09b/15/16. With a
     *                    zero frame pitch (sheet film) every frame lands on the
     *                    same patch of web, which is correct.
     * 13  SCALAR/AVX2    Same semantics; the RNG and film-coordinate headers
     *                    are byte-identical, and the web offset stays double in
     *                    both builds for the reason given under coatingScale.
     * 14  FULL/LITE      PENDING. Both - it is an input, not a quality knob.
     *
     * !! RANGE CONTRADICTION, NOT SILENTLY RESOLVED: an earlier revision of this
     *   header documented "RANGE >= 0". The implementation contradicts it -
     *   AlgoControl.cpp states "May be negative", AlgoCounterRng.hpp states
     *   frameIndex is signed and may legitimately be negative near a clip
     *   boundary, and stage 16 uses std::floor specifically to handle negative
     *   indices correctly. The implementation is therefore taken as
     *   authoritative here and the range is recorded as SIGNED, negatives
     *   supported. The stale "RANGE >= 0" claim has been removed rather than
     *   carried forward. If the intent really is non-negative, the code must be
     *   changed, not the comment.
     */
    int32_t frameIndex;

    // -- determinism --------------------------------------------------------

    /**
     *  1  NAME           seed
     *  2  TYPE           int32_t
     *  3  AE CONTROL     integer field + randomise button
     *  4  UNIT           dimensionless integer identifier. It is only ever
     *                    bit-manipulated - reinterpreted as uint32_t, XORed
     *                    with a per-stage salt, shifted into a 64-bit counter -
     *                    and never enters physical arithmetic
     *  5  MIN            none - the whole int32_t range is valid. Negatives
     *                    wrap into the upper half of uint32_t, which is well
     *                    defined and harmless because the mixer treats all
     *                    64-bit values alike
     *  6  MAX            none, same reason
     *  7  DEFAULT        12345   (AlgoControl.cpp, getAlgoControlsDefault)
     *  8  STEP           1 [proposed] - an identifier; the panel should offer a
     *                    randomise button rather than a drag
     *  9  PURPOSE        Master seed for the stochastic stages. Identical
     *                    inputs and seed give a bit-identical render.
     * 10  OUTPUT EFFECT  Changes the grain and coating realisation and the
     *                    per-frame draws. Changes nothing physical - not
     *                    amplitude, not scale, not colour.
     * 11  STAGES         04, 10, 11, 13, 14. It does NOT reach 09b, 15 or 16 -
     *                    those use damage.damageSeed instead
     * 12  INTERACTIONS   Folded with frameIndex in the RNG counter. Explicitly
     *                    independent of damage.damageSeed, so re-rolling the
     *                    grain does not re-roll the dirt. Neutralised wherever
     *                    its stage is disabled.
     * 13  SCALAR/AVX2    Same semantics, and the counter-based RNG header is
     *                    byte-identical, so the drawn integer sequences are
     *                    BIT-IDENTICAL between the two builds. Only the
     *                    floating-point consumption of the drawn values
     *                    differs.
     * 14  FULL/LITE      PENDING. Both. Note a consequence the Lite work must
     *                    face: Preview and Full draw from the same streams but
     *                    consume different numbers of them if octave or channel
     *                    counts change, so the same seed will not give the same
     *                    realisation in the two modes. That is acceptable but
     *                    must be stated, not discovered.
     */
    int32_t seed;

    // -- spectral taking matrix and grain temporal mode (2026-10-06) ----------

    /**
     *  1  NAME           spectralTaking
     *  2  TYPE           bool
     *  3  AE CONTROL     checkbox
     *  4  UNIT           none - boolean switch
     *  5  MIN            false, by type
     *  6  MAX            true, by type
     *  7  DEFAULT        false  (SpectralTakingDef; AlgoControl.cpp,
     *                    getAlgoControlsDefault)
     *  8  STEP           n/a
     *  9  PURPOSE        Replace the stock's AUTHORED taking_matrix by the one
     *                    DERIVED from its measured layer sensitivities under
     *                    the scene illuminant (AlgoSpectralTakingMatrix: each
     *                    layer's response to three smooth primaries, rows
     *                    normalised so a neutral stays neutral).
     * 10  OUTPUT EFFECT  Off: unchanged. On: cross-channel mixing in EXPOSURE,
     *                    before the characteristic curve, on every stock that
     *                    carries three-layer curves; stocks without them keep
     *                    the authored matrix (the derivation refuses).
     * 11  STAGES         02b (resolved once per frame in Algorithm_Main and
     *                    passed to the stage and to its pass-through test)
     * 12  INTERACTIONS   Uses sceneKelvin for the illuminant and the profile's
     *                    taking filter (fetchLayers). Off by default because
     *                    dye_matrix and InterimageSpec already carry mixing
     *                    downstream; turning it on without a measured
     *                    reference applies the same physics twice.
     *                    spectralUpsampling (2026-10-13) refines this path
     *                    from the 3x3 matrix to a per-pixel Jakob-Hanika
     *                    spectrum when it is ALSO on; it has no effect alone.
     * 13  SCALAR/AVX2    Identical: the matrix is derived in the shared
     *                    AlgoSpectralSensitivity.cpp, in double, rounded once
     *                    to float exactly like the authored matrix.
     * 14  FULL/LITE      PENDING. Both.
     *
     *  Twin: film_sim.RenderSettings.spectral_taking (added to C++ 2026-10-06,
     *  owner decision G4 -- the three engines execute the same maximum flow).
     */
    bool spectralTaking;

    /**
     *  1  NAME           grainTemporalMode
     *  2  TYPE           GrainTemporalModeCtrl (AlgoControlEnums.hpp)
     *  3  AE CONTROL     popup: Still | Motion | Frozen
     *  4  UNIT           none - enumeration
     *  5  MIN            eGRAIN_TEMPORAL_STILL
     *  6  MAX            eGRAIN_TEMPORAL_FROZEN; out-of-range values clamp to
     *                    the default on entry (AlgoControlsClamped)
     *  7  DEFAULT        eGRAIN_TEMPORAL_STILL  (GrainTemporalModeCtrlDef)
     *  8  STEP           1, by enumeration
     *  9  PURPOSE        How the camera negative's grain field moves between
     *                    frames: Still (fresh field per frame, full still
     *                    amplitude - the physically faithful default), Motion
     *                    (perceptual matching: amplitude x 1/sqrt(clamp(
     *                    frameRate x 0.2 s, 1, 8))), Frozen (frame index 0 for
     *                    the whole clip).
     * 10  OUTPUT EFFECT  Still: unchanged. Motion at 24 fps: negative grain
     *                    amplitude x 0.4564. Frozen: the same grain on every
     *                    frame.
     * 11  STAGES         11 only. Print grain (14) and duplication grain (13)
     *                    keep the frame index, as in the Python reference.
     * 12  INTERACTIONS   Motion reads frameRate; a non-positive frameRate
     *                    leaves the amplitude unscaled (the reference refuses
     *                    such a request outright - R-T3, never a silent
     *                    default). Multiplies grainScale.
     * 13  SCALAR/AVX2    Identical: both stages call AlgoGrainTemporalFrame /
     *                    AlgoGrainTemporalScale below.
     * 14  FULL/LITE      PENDING. Both.
     *
     *  Twin: film_sim.RenderSettings.grain_temporal_mode (added to C++
     *  2026-10-06, owner decision G6).
     */
    GrainTemporalModeCtrl grainTemporalMode;

    // -- spectral upsampling (2026-10-13, owner-approved) ---------------------

    /**
     *  1  NAME           spectralUpsampling
     *  2  TYPE           bool
     *  3  AE CONTROL     checkbox (greyed while spectralTaking is off)
     *  4  UNIT           none - boolean switch
     *  5  MIN            false, by type
     *  6  MAX            true, by type
     *  7  DEFAULT        false  (SpectralUpsamplingDef; AlgoControl.cpp,
     *                    getAlgoControlsDefault)
     *  8  STEP           n/a
     *  9  PURPOSE        Refine the DERIVED taking (spectralTaking) from one
     *                    3x3 matrix per stock to a per-pixel spectrum: each
     *                    pixel's linear-light Rec.709 (sRGB-primary) exposure
     *                    is upsampled to the Jakob-Hanika 2019 sigmoid-
     *                    polynomial spectrum and integrated against the
     *                    stock's measured layer sensitivities under the scene
     *                    illuminant (AlgoSpectralUpsamplingLut +
     *                    AlgoStage02b_SpectralUpsampling).
     * 10  OUTPUT EFFECT  Effective ONLY when spectralTaking is also on
     *                    (effective switch: spectralTaking &&
     *                    spectralUpsampling), and only on the stocks the
     *                    derived matrix serves (three-layer curves). Neutrals
     *                    are unchanged; saturated colours follow the film's
     *                    own sensitivity overlaps instead of a linear mix.
     *                    Off, or on with spectralTaking off: bit-identical to
     *                    the render without this control.
     * 11  STAGES         02b (table built once per frame in Algorithm_Main into
     *                    MemHandler::Scr_SpectralLut).
     * 12  INTERACTIONS   Requires spectralTaking; reads sceneKelvin (the same
     *                    blackbody illuminant as the derived matrix). The
     *                    spectrum is a plausible reconstruction, not the
     *                    scene's true spectrum: three numbers cannot recover a
     *                    metamer the camera already merged.
     * 13  SCALAR/AVX2    Same table (built in the shared
     *                    AlgoSpectralSensitivity.cpp, in double, rounded once
     *                    to float); per-pixel lookup in AlgoType (double) in
     *                    the scalar build, float32 with gathers in AVX2.
     * 14  FULL/LITE      PENDING. Both.
     *
     *  Twin: film_sim.RenderSettings.spectral_upsampling.
     */
    bool spectralUpsampling;

    // -- Fuji fourth (CL) layer: NO CONTROL (2026-10-22e, owner decision) ----
    //  The member `bool fourthLayer` (2026-10-21) is REMOVED: the owner's
    //  decision is that the fourth layer is part of the film, always on.
    //  Stage 8b applies it on every stock that carries log_s_c and
    //  cl_red_inhibition > 0 (Fuji colour negatives; not on monochrome or
    //  reversal). Availability bit 56 stays RESERVED and is never set.
    //  Twin: film_sim.fourth_layer_active(profile).

    // -- aim exposure (2026-10-22, owner-approved batch, G-209-AIM) -----------

    /**
     *  1  NAME           aimExposure
     *  2  TYPE           bool
     *  3  AE CONTROL     checkbox (greyed on stocks without a grey-card aim)
     *  4  UNIT           none - boolean switch
     *  5  MIN            false, by type
     *  6  MAX            true, by type
     *  7  DEFAULT        false  (AimExposureDef; AlgoControl.cpp,
     *                    getAlgoControlsDefault)
     *  8  STEP           n/a
     *  9  PURPOSE        Expose the frame the way the maker's data sheet
     *                    defines correct exposure. Without it, metered 18 %
     *                    grey sits at log E = log10(taking row sum) on the
     *                    stored curve, which on every stock with a published
     *                    grey-card aim lands 0.35-1.35 D ABOVE the aim (verify
     *                    G-209-AIM). With it, the stock's final curves (after
     *                    every profile resolver) are translated along log E by
     *                    its aim_exposure_shift_stops (generator-computed: the
     *                    red curve's log E at the midpoint of the grey-card
     *                    aim for the stock's own EI, minus the metered log E,
     *                    in stops; -3.15 to -6.37 on the 17 stocks), so
     *                    metered grey reads the aim density.
     * 10  OUTPUT EFFECT  The anchor solve sees the translated curves and
     *                    re-times the print, as a lab times an aim-exposed
     *                    negative: overall level holds, the scene sits lower
     *                    on the curve (more highlight latitude, less shadow
     *                    separation, small colour-balance shifts from the
     *                    stock's own crossover).
     * 11  STAGES         frame setup (AlgoResolveAimExposure in
     *                    AlgorithmMain.cpp); read by the anchor solve, 08
     *                    and everything that evaluates the curves.
     * 12  INTERACTIONS   Independent of exposureStops (which still moves the
     *                    scene against a fixed print timing). Shift read from
     *                    the stock as shipped, applied to the curves after
     *                    process variant, development, age, storage and batch.
     *                    Inert on stocks without an aim (shift 0) and on a
     *                    curve set with a measured table.
     * 13  SCALAR/AVX2    Shared frame-setup code; the translated toe_x /
     *                    shoulder_x are computed in double, stored as float.
     * 14  FULL/LITE      PENDING. Both.
     *
     *  Twin: film_sim.RenderSettings.aim_exposure.
     */
    bool aimExposure;

    // -- physical film damage -----------------------------------------------

    /**
     *  1  NAME           filmDamageEnabled
     *  2  TYPE           bool
     *  3  AE CONTROL     checkbox that greys out the whole damage group
     *  4  UNIT           none - boolean switch
     *  5  MIN            false, by type
     *  6  MAX            true, by type
     *  7  DEFAULT        false  (AlgoControl.cpp, getAlgoControlsDefault)
     *                    Clean film is the default and damage is opt-in, which
     *                    is what the project requirements always specified.
     *                    Changed from true on 2026-09-04 - see the resolved
     *                    conflict note below
     *  8  STEP           n/a
     *  9  PURPOSE        Hard gate for the entire FilmDamage block. false means
     *                    every damage generator is skipped at zero cost and the
     *                    damage sub-struct is not read at all.
     * 10  OUTPUT EFFECT  With the flag clear, the engine is numerically
     *                    identical to a build with no defect layer - verified:
     *                    whole-chain agreement against the Python reference is
     *                    unchanged to the digit. Checked ONCE per frame, not
     *                    per pixel, so a clean render pays exactly one branch
     *                    for the whole subsystem.
     * 11  STAGES         03c, 09b, 15, 16 and the storage resolver at frame
     *                    setup (storageSeverity) - since 2026-10-10; before
     *                    that 09b, 15 and 16 only
     * 12  INTERACTIONS   Outermost of a three-level gate chain: this flag, then
     *                    damage.damageStrength > 0, then at least one class
     *                    level > 0. It transitively gates frameRate and
     *                    frameIndex at the damage stages. It has no
     *                    interaction with seed, because the damage stages use
     *                    damageSeed.
     * 13  SCALAR/AVX2    Same semantics, identical gate chain, including the
     *                    same asymmetry - stage 15 copies then returns, while
     *                    03c, 09b and 16 are skipped by their AlgoPass*
     *                    predicate or return after an unconditional copy.
     * 14  FULL/LITE      PENDING. Expected Full only for the pixel stages.
     *
     * !! DEFAULT CONFLICT - RESOLVED 2026-09-04, IN FAVOUR OF THE REQUIREMENT
     *   For several revisions the implementation set this TRUE while the
     *   project requirements said a clean render must be the default and
     *   damage must be opt-in. An earlier revision of this header asserted
     *   DEFAULT false while the code assigned true; the revision before this
     *   one recorded the disagreement and deliberately left the code alone,
     *   per the rule that documented defaults must match the implementation.
     *
     *   The owner asked on 2026-09-04 for the defect layer to be explicitly
     *   switchable, which is the decision this was waiting on.
     *   `AlgoControl.cpp` now assigns FALSE and the DEFAULT line above says
     *   false. A default render is the pure film-stock simulation and is
     *   numerically identical to a build with no defect layer.
     *
     *   TO ENABLE THE DEFECT LAYER, one line - `filmDamageEnabled = true` -
     *   and the working set already in `damage` takes effect.
     *
     *   Two consequences, both good: a caller wanting clean film gets it for
     *   free again; and the divergence from film_sim.RenderSettings, which has
     *   no damage group at all, is gone, so the verification harness's manual
     *   clear is now redundant rather than load-bearing.
     */
    bool filmDamageEnabled;

    /**
     *  1  NAME           damage
     *  2  TYPE           FilmDamage (17-field sub-struct, documented above)
     *  3  AE CONTROL     a control group in the panel, greyed out when
     *                    filmDamageEnabled is false
     *  4  UNIT           n/a - aggregate. Its members carry their own units
     *  5  MIN            n/a - aggregate
     *  6  MAX            n/a - aggregate
     *  7  DEFAULT        getFilmDamageDefault()
     *                    (AlgoControl.cpp, getAlgoControlsDefault)
     *  8  STEP           n/a - aggregate
     *  9  PURPOSE        Groups the damage parameters so there is one object to
     *                    hand around and one thing to serialise with a preset.
     * 10  OUTPUT EFFECT  Per member. Nine of seventeen members currently affect
     *                    the image; eight are inert. See each member's block.
     * 11  STAGES         09b, 15, 16
     * 12  INTERACTIONS   Read only when filmDamageEnabled is true.
     * 13  SCALAR/AVX2    Identical layout - AlgoControl.hpp is byte-identical
     *                    between the scalar and AVX2 deliveries, so the control
     *                    SET cannot differ between them by construction. Only
     *                    behavioural differences need documenting, and they are
     *                    documented per field.
     * 14  FULL/LITE      PENDING. Expected Full only.
     */
    FilmDamage damage;
};


/// Defaults mirroring film_sim.RenderSettings, with ONE deliberate divergence:
/// the damage group, which the reference model does not have. See
/// filmDamageEnabled for the consequences of that divergence and for the
/// unresolved conflict over its default value.
AlgoControls getAlgoControlsDefault (void) noexcept;

/// Damage sub-defaults on their own, for a "reset this group" button.
FilmDamage   getFilmDamageDefault   (void) noexcept;

/// \brief Return a copy of \p in with every control forced into its documented
///        range (AlgoControlEnums.hpp Min/Max), sentinels preserved.
///
/// ADDED 2026-10-02 after the owner observed black-and-white stocks rendering
/// as a near-binary image when the host forwarded Effect Control panel values.
/// The engine's historical contract was "pre-validated; no field is
/// range-checked here", so an out-of-range value went straight into the
/// stages. Measured on KODAK_TRI_X_400TX: grainScale 100 (a percent passed as
/// a factor) clips 82 % of pixels to black or white; scannerSpecular 85.3 or 2
/// flattens a silver image to a single grey; greyTarget 18 renders all white.
/// Colour stocks degrade far less, which is why the defect looked B&W-only.
///
/// Rules: a non-finite number becomes the default; a negative value on a
/// control whose "off / use the stock's own value" sentinel is negative
/// becomes that sentinel; everything else is clamped to [Min, Max]; an
/// enumerator outside its list becomes its default; filmProfile is clamped to
/// [0, profileCount - 1] (it was an unchecked index into the database);
/// dupeStock must be an intermediate stock (DUPE_FINE_GRAIN or
/// KODAK_VISION3_DI_2254) -- any other selection, including eSTOCKS_OWN,
/// becomes DupeStockCtrlDef, because a release print or the SCAN_DI transform
/// used as the duplicating stock compounds contrast over the generation chain.
/// Values already in range are returned unchanged, bit for bit.
AlgoControls AlgoControlsClamped (const AlgoControls& in,
                                  const int32_t       profileCount) noexcept;


/// \brief Frame index the camera negative's GRAIN field is drawn for
///        (grainTemporalMode, 2026-10-06). Frozen pins frame 0; Still and
///        Motion use the host's frame. Shared by the Scalar and AVX2 stage 11.
///        Twin: film_sim.simulate, stage 11 ("frozen" -> frame_index = 0).
inline int32_t AlgoGrainTemporalFrame (const AlgoControls& c,
                                       const int32_t       frameIndex) noexcept
{
    return (c.grainTemporalMode == GrainTemporalModeCtrl::eGRAIN_TEMPORAL_FROZEN)
         ? 0 : frameIndex;
}

/// \brief Amplitude factor on the camera negative's grain (grainTemporalMode).
///        Motion: 1 / sqrt(clamp(frameRate * GrainTemporalIntegrationS, 1,
///        GrainTemporalMaxFrames)); otherwise exactly 1.0, so Still and Frozen
///        leave grainScale bit-for-bit untouched. A non-positive frameRate in
///        Motion also returns 1.0 (the reference raises instead -- R-T3 -- and a
///        plugin cannot; leaving the amplitude unscaled is the no-surprise
///        equivalent). Twin: film_sim.temporal_grain_scale.
inline double AlgoGrainTemporalScale (const AlgoControls& c) noexcept
{
    if ((c.grainTemporalMode != GrainTemporalModeCtrl::eGRAIN_TEMPORAL_MOTION)
        || !(c.frameRate > 0.0))
        return 1.0;
    double n = c.frameRate * GrainTemporalIntegrationS;
    if (n < 1.0)                    n = 1.0;
    if (n > GrainTemporalMaxFrames) n = GrainTemporalMaxFrames;
    return 1.0 / std::sqrt(n);
}
