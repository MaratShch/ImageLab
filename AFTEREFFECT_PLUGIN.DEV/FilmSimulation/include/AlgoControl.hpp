#pragma once

#include <cstdint>

// eFILM_PROFILE: the generated profile index enumeration. Every algorithm
// parameter reaches the engine through this structure, and the film selection is
// an algorithm parameter, so it lives here rather than in the call signature.
#include "film_enum.hpp"

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
 * the distinction is not cosmetic. Audit of the whole engine found that NO
 * numeric control in this structure has an enforced upper bound. Stages floor
 * their inputs with MAX_VALUE(x, 0) and nothing else. The only ceiling anywhere
 * caps a *product* rather than a control (ALGO_DEFOCUS_MAX_LOSS at
 * Algo_06_Sim.cpp, and ALGO_DUST_DENSITY_MAX at Algo_09_Sim.cpp). The engine's
 * own contract states this: AlgorithmMain.hpp - "Pre-validated; no field is
 * range-checked here." So every "MAX ... advisory" line below means: the host
 * panel must enforce it, because the engine will not.
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
 *                       AlgoControls::filmDamageEnabled. NINE of the seventeen
 *                       are now consumed - by stages 09b, 15 and 16. EIGHT
 *                       remain unconsumed and are marked as such in their own
 *                       blocks. (Earlier revisions of this header stated the
 *                       whole struct was inert. That was true when written and
 *                       is no longer true.)
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
 *   Consumed (9): damageStrength, damageSeed, dustLevel, debrisLevel,
 *                 fibreLevel, dirtClumping  -> stage 09b
 *                 weaveAmount               -> stage 15
 *                 gateDirt, damageEvents    -> stage 16
 *   Unconsumed (8): scratchTransport, scratchHandling, processingQuality,
 *                 dryingMarks, storageSeverity, colourVeil, flickerStops,
 *                 scannerArtifacts
 *   The unconsumed eight remain in the layout deliberately, so the panel and
 *   any serialised preset can be built once against a stable structure. Each
 *   says so in its own item 9.
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
     *  3  AE CONTROL     slider
     *  4  UNIT           intended dimensionless multiplier on a per-metre
     *                    scratch rate. UNVERIFIABLE - no arithmetic consumes
     *                    the field, so no unit can be derived from the code
     *  5  MIN            none enforced - the field has no reader
     *  6  MAX            4.0 advisory - NOT ENFORCED, no reader
     *  7  DEFAULT        0.40   (AlgoControl.cpp, getFilmDamageDefault)
     *  8  STEP           0.01 [proposed], consistent with the other levels
     *  9  PURPOSE        UNCONSUMED IN THIS BUILD. Intended: longitudinal
     *                    transport scratches - long, straight, parallel to film
     *                    travel, continuing across frame boundaries. The
     *                    defining motion-picture defect ("rain", "tramlines"),
     *                    and machine-fixed: it holds a fixed position on screen
     *                    for a whole reel while the image moves past it.
     * 10  OUTPUT EFFECT  NONE TODAY. No stage reads this field; setting it has
     *                    no effect on the rendered image at any value.
     *                    Intended orientation, when implemented: horizontal on
     *                    a still frame and vertical on every common cine
     *                    format, because film travels along the long axis of a
     *                    still frame and the short axis of a cine one - derived
     *                    from the format, never authored.
     * 11  STAGES         none currently. Intended stage 09b.
     * 12  INTERACTIONS   None in force. Intended: multiplied by damageStrength
     *                    like the other levels.
     * 13  SCALAR/AVX2    No difference - unconsumed in both trees.
     * 14  FULL/LITE      PENDING and moot until the field is consumed.
     */
    double scratchTransport;

    /**
     *  1  NAME           scratchHandling
     *  2  TYPE           double
     *  3  AE CONTROL     slider
     *  4  UNIT           intended dimensionless multiplier on a burst rate.
     *                    UNVERIFIABLE - no reader
     *  5  MIN            none enforced - no reader
     *  6  MAX            4.0 advisory - NOT ENFORCED, no reader
     *  7  DEFAULT        0.30   (AlgoControl.cpp, getFilmDamageDefault)
     *  8  STEP           0.01 [proposed]
     *  9  PURPOSE        UNCONSUMED IN THIS BUILD. Intended: random handling
     *                    scratches - short, curved, 0.3-4 mm, individually very
     *                    faint but numerous, generated in bursts because a
     *                    single wipe leaves several roughly parallel marks.
     *                    Locked to the film, not to the machine.
     * 10  OUTPUT EFFECT  NONE TODAY.
     * 11  STAGES         none currently. Intended stage 09b.
     * 12  INTERACTIONS   None in force.
     * 13  SCALAR/AVX2    No difference - unconsumed in both trees.
     * 14  FULL/LITE      PENDING and moot until the field is consumed.
     */
    double scratchHandling;

    // =======================================================================
    //  PROCESSING AND DRYING
    // =======================================================================

    /**
     *  1  NAME           processingQuality
     *  2  TYPE           double
     *  3  AE CONTROL     slider
     *  4  UNIT           intended dimensionless 0..1 quality axis.
     *                    UNVERIFIABLE - no reader
     *  5  MIN            none enforced - no reader
     *  6  MAX            1.0 advisory - NOT ENFORCED, no reader
     *  7  DEFAULT        0.30   (AlgoControl.cpp, getFilmDamageDefault)
     *  8  STEP           0.01 [proposed] - a 0..1 axis with 100 positions
     *  9  PURPOSE        UNCONSUMED IN THIS BUILD. Intended: inverse quality of
     *                    the development bath - 0 a professional continuous
     *                    machine, 1 a hand tank in a bathroom. One control for
     *                    development mottle, air bells, surge marks, chemical
     *                    stains and reticulation together, because all five
     *                    express the same underlying variable: agitation
     *                    quality.
     * 10  OUTPUT EFFECT  NONE TODAY.
     * 11  STAGES         none currently.
     * 12  INTERACTIONS   None in force.
     * 13  SCALAR/AVX2    No difference - unconsumed in both trees.
     * 14  FULL/LITE      PENDING and moot until the field is consumed.
     */
    double processingQuality;

    /**
     *  1  NAME           dryingMarks
     *  2  TYPE           double
     *  3  AE CONTROL     slider
     *  4  UNIT           intended dimensionless multiplier. UNVERIFIABLE - no
     *                    reader
     *  5  MIN            none enforced - no reader
     *  6  MAX            2.0 advisory - NOT ENFORCED, no reader
     *  7  DEFAULT        0.25   (AlgoControl.cpp, getFilmDamageDefault)
     *  8  STEP           0.01 [proposed]
     *  9  PURPOSE        UNCONSUMED IN THIS BUILD. Intended: drying-stage
     *                    defects - water spots, tide lines, squeegee bands.
     * 10  OUTPUT EFFECT  NONE TODAY. Intended behaviour is strongly
     *                    gauge-dependent and that dependence is derived, not
     *                    authored: a single 5 mm drying mark covers 2.3 per
     *                    cent of a 35 mm still frame and 25.6 per cent of a
     *                    16 mm frame.
     * 11  STAGES         none currently.
     * 12  INTERACTIONS   None in force.
     * 13  SCALAR/AVX2    No difference - unconsumed in both trees.
     * 14  FULL/LITE      PENDING and moot until the field is consumed.
     */
    double dryingMarks;

    // =======================================================================
    //  AGE AND STORAGE
    // =======================================================================

    /**
     *  1  NAME           storageSeverity
     *  2  TYPE           double
     *  3  AE CONTROL     slider
     *  4  UNIT           intended dimensionless 0..1 severity axis.
     *                    UNVERIFIABLE - no reader
     *  5  MIN            none enforced - no reader
     *  6  MAX            1.0 advisory - NOT ENFORCED, no reader
     *  7  DEFAULT        0.20   (AlgoControl.cpp, getFilmDamageDefault)
     *  8  STEP           0.01 [proposed] - a 0..1 axis with 100 positions
     *  9  PURPOSE        UNCONSUMED IN THIS BUILD. Intended: 0 a cold vault,
     *                    1 a warm attic for forty years. One control for dye
     *                    fading, colour crossover, age fog, fungal growth and
     *                    base deterioration, scaling the profile's own
     *                    AgingSpec figures.
     * 10  OUTPUT EFFECT  NONE TODAY.
     * 11  STAGES         none currently.
     * 12  INTERACTIONS   None in force. Intended to be distinct from
     *                    colourVeil - see that field's item 9 for why merging
     *                    them is wrong.
     * 13  SCALAR/AVX2    No difference - unconsumed in both trees.
     * 14  FULL/LITE      PENDING and moot until the field is consumed.
     */
    double storageSeverity;

    /**
     *  1  NAME           colourVeil
     *  2  TYPE           double
     *  3  AE CONTROL     slider
     *  4  UNIT           intended additive density offset in one dye record.
     *                    UNVERIFIABLE - no reader
     *  5  MIN            none enforced - no reader
     *  6  MAX            2.0 advisory - NOT ENFORCED, no reader
     *  7  DEFAULT        0.15   (AlgoControl.cpp, getFilmDamageDefault)
     *  8  STEP           0.01 [proposed]
     *  9  PURPOSE        UNCONSUMED IN THIS BUILD. Intended: a flat additive
     *                    level shift in ONE dye record, with no neutral point
     *                    anywhere in the tonal range.
     *
     *                    SEPARATE from the crossover storageSeverity drives,
     *                    and deliberately so. Crossover is tone dependent -
     *                    shadows lean one way, highlights the other, mid tones
     *                    pass through neutral. A veil has no neutral point at
     *                    all. Measurement on aged ORWOCOLOR found both on
     *                    different parts of one roll, and the second cannot be
     *                    produced by turning up the first. Merging them into
     *                    one "fading" slider is the commonest error in
     *                    colour-fade emulation.
     * 10  OUTPUT EFFECT  NONE TODAY. Intended: level only - it does NOT reduce
     *                    contrast in the affected layer. An earlier analysis
     *                    claimed it did and was withdrawn on re-measurement.
     * 11  STAGES         none currently.
     * 12  INTERACTIONS   None in force.
     * 13  SCALAR/AVX2    No difference - unconsumed in both trees.
     * 14  FULL/LITE      PENDING and moot until the field is consumed.
     */
    double colourVeil;

    // =======================================================================
    //  MACHINE-SIDE -- motion picture
    // =======================================================================

    /**
     *  1  NAME           gateDirt
     *  2  TYPE           double
     *  3  AE CONTROL     slider
     *  4  UNIT           dimensionless multiplier, applied to two different
     *                    quantities: a particle COUNT
     *                    (ALGO_GATE_INITIAL_COUNT = 4.0 particles) and a
     *                    per-frame accretion RATE
     *                    (ALGO_GATE_ACCRETION_PER_FRAME = 4.0e-3 /frame). It
     *                    carries no length or area unit itself
     *  5  MIN            0.0, ENFORCED (MAX_VALUE floor at stage 16)
     *  6  MAX            2.0 advisory - NOT ENFORCED on the control. Two
     *                    indirect caps exist downstream: the persistent-mark
     *                    window ALGO_GATE_WINDOW = 96 frames and the sparkle
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
     *  4  UNIT           intended RMS amplitude in photographic STOPS.
     *                    UNVERIFIABLE from code - no reader exists
     *  5  MIN            none enforced - no reader
     *  6  MAX            0.5 advisory - NOT ENFORCED, no reader
     *  7  DEFAULT        0.15   (AlgoControl.cpp, getFilmDamageDefault)
     *  8  STEP           0.01 [proposed] - one hundredth of a stop is below the
     *                    visible threshold for frame-to-frame flicker, so the
     *                    control will read as continuous
     *  9  PURPOSE        UNCONSUMED IN THIS BUILD. Intended: temporal exposure
     *                    flicker. Hand-cranked cameras and early intermittent
     *                    mechanisms did not deliver equal exposure to
     *                    successive frames.
     * 10  OUTPUT EFFECT  NONE TODAY. Stage 03c, its intended consumer, is a
     *                    pass-through stub that voids all five of its arguments
     *                    and copies input to output.
     *
     *                    Intended: acts on EXPOSURE, before the characteristic
     *                    curve. A brightness change applied after development
     *                    is a grade, and a grade does not move highlights
     *                    through the shoulder the way a real exposure change
     *                    does.
     * 11  STAGES         none currently. Intended stage 03c.
     * 12  INTERACTIONS   None in force.
     * 13  SCALAR/AVX2    No difference - stage 03c is the same stub in both
     *                    trees.
     * 14  FULL/LITE      PENDING and moot until the field is consumed.
     *
     * !! DOCUMENTATION DEFECT FOUND AND NOT SILENTLY FIXED: AlgoTemporalFlicker.hpp
     *   refers to AlgoControls fields named flickerBaseHz and
     *   flickerColourSpread. Neither exists anywhere in either tree. That
     *   header comment is wrong and should be corrected or the fields added -
     *   an owner decision, not a documentation one, so it is recorded here
     *   rather than resolved.
     */
    double flickerStops;

    // =======================================================================
    //  SCANNER -- a separate layer, not a film defect
    // =======================================================================

    /**
     *  1  NAME           scannerArtifacts
     *  2  TYPE           double
     *  3  AE CONTROL     slider
     *  4  UNIT           intended dimensionless multiplier. UNVERIFIABLE - no
     *                    reader
     *  5  MIN            none enforced - no reader
     *  6  MAX            2.0 advisory - NOT ENFORCED, no reader
     *  7  DEFAULT        0.20   (AlgoControl.cpp, getFilmDamageDefault)
     *  8  STEP           0.01 [proposed]
     *  9  PURPOSE        UNCONSUMED IN THIS BUILD. Intended: digitisation
     *                    artefacts - illumination shading, fixed-pattern
     *                    banding, Newton's rings. Switchable independently of
     *                    everything above, because "clean scan of damaged film"
     *                    and "good film, cheap scanner" are both common
     *                    requests and merging the two makes neither
     *                    expressible.
     * 10  OUTPUT EFFECT  NONE TODAY. Worth keeping honest about when it is
     *                    implemented: in the reference dataset the strongest
     *                    high-frequency texture across 81 crops of damaged film
     *                    was the SCANNER's weave pattern, not any film defect.
     *                    Effects that look like film damage frequently are not.
     * 11  STAGES         none currently.
     * 12  INTERACTIONS   None in force.
     * 13  SCALAR/AVX2    No difference - unconsumed in both trees.
     * 14  FULL/LITE      PENDING and moot until the field is consumed.
     */
    double scannerArtifacts;
};


// ===========================================================================
// Live controls -- 25 fields, all consumed by the renderer
// ===========================================================================
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
     * 10  OUTPUT EFFECT  Selects the whole parameter set. The index is
     *                    dereferenced with NO range check, deliberately: the
     *                    enumerator comes from the effect panel and is treated
     *                    as pre-validated, and re-testing it on the hot path
     *                    was judged duplicated work. CONSEQUENCE, stated
     *                    plainly: an out-of-range value is an unchecked
     *                    out-of-bounds read, so the HOST must guarantee the
     *                    range.
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
     * 11  STAGES         15 and 16 in practice. It is also passed to 03c and
     *                    09b, both of which explicitly discard it - 03c because
     *                    it is a stub, 09b because none of its classes has a
     *                    per-second rate.
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

    /**
     *  1  NAME           filmFormat
     *  2  TYPE           const char*
     *  3  AE CONTROL     dropdown, populated from FORMAT_GEOM
     *  4  UNIT           a name key, dimensionless. The quantity it selects is
     *                    LENGTH IN MILLIMETRES - frame width, height and pitch
     *  5  MIN            n/a (string). nullptr and "" are both accepted and
     *                    mean "use the stock's default_format"
     *  6  MAX            n/a. An unrecognised key falls back to the profile
     *                    default; if that also fails to resolve, the geometry
     *                    is zeroed rather than guessed
     *  7  DEFAULT        "super35"   (AlgoControl.cpp, getAlgoControlsDefault)
     *  8  STEP           n/a - enumeration by list
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
    const char* filmFormat;

    /**
     *  1  NAME           printStock
     *  2  TYPE           const char*
     *  3  AE CONTROL     dropdown
     *  4  UNIT           a name key, dimensionless. The quantities it selects
     *                    are the print stock's mtf_f50 in CYCLES PER MILLIMETRE
     *                    and its characteristic curves in OPTICAL DENSITY
     *  5  MIN            n/a. "" (the default) means "use the stock's own
     *                    default_print"
     *  6  MAX            n/a. An unmatched name degrades silently to the
     *                    profile default, and failing that to no print stock at
     *                    all
     *  7  DEFAULT        ""   (AlgoControl.cpp, getAlgoControlsDefault)
     *  8  STEP           n/a - enumeration by list
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
    const char* printStock;

    /**
     *  1  NAME           dupeStock
     *  2  TYPE           const char*
     *  3  AE CONTROL     dropdown
     *  4  UNIT           a name key, dimensionless. Selects mtf_f50 in cycles
     *                    per millimetre, dmin in optical density, and an rms
     *                    grain figure
     *  5  MIN            n/a
     *  6  MAX            n/a. Unmatched degrades to the print stock, then to
     *                    nothing - and nothing forces the duplication chain to
     *                    zero passes whatever generations says
     *  7  DEFAULT        "DUPE_FINE_GRAIN"
     *                    (AlgoControl.cpp, getAlgoControlsDefault)
     *  8  STEP           n/a - enumeration by list
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
    const char* dupeStock;

    /**
     *  1  NAME           generations
     *  2  TYPE           int32_t
     *  3  AE CONTROL     integer slider
     *  4  UNIT           dimensionless COUNT of interpositive / dupe-negative
     *                    PAIRS. The stage runs 2 x generations passes
     *  5  MIN            0, ENFORCED (CLAMP_VALUE at stage 13)
     *  6  MAX            4, ENFORCED (CLAMP_VALUE against
     *                    ALGO_DUPE_MAX_GENERATIONS). THIS IS THE ONLY CONTROL
     *                    IN THE WHOLE STRUCTURE WITH AN ENFORCED UPPER BOUND.
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
     *  2  TYPE           int32_t
     *  3  AE CONTROL     dropdown, populated from the selected stock's own
     *                    `process_variants` list -- the entries are named by
     *                    the manufacturer ("EI 1600 (Push 1)", "Cs2 two-bath
     *                    kit") and the list is EMPTY on 164 of the 170 stocks,
     *                    so the control should hide itself rather than show a
     *                    dropdown with one dead entry
     *  4  UNIT           index into that vector. Not an enum: the list is
     *                    per-stock and changes when the stock changes
     *  5  MIN            -1, the OFF sentinel
     *  6  MAX            size of the selected stock's process_variants, minus
     *                    one. Out of range is treated as -1 rather than
     *                    clamped, because a stale preset pointing at a variant
     *                    the new stock does not have should render the stock as
     *                    shipped, not render its variant 0
     *  7  DEFAULT        -1 -- the development the stored curves represent
     *  8  STEP           1
     *  9  PURPOSE        Selects a DIFFERENT DEVELOPMENT of the same emulsion:
     *                    a push, a cross-process, an alternate chemistry kit.
     *                    Where the manufacturer plotted that development
     *                    separately the record carries its own TRACED curve
     *                    set, so this is not a contrast tweak -- it is a second
     *                    measurement of the same film.
     * 10  OUTPUT EFFECT  Replaces the profile's characteristic curves, and its
     *                    exposure index where the variant states one, before
     *                    anything reads either. -1 reproduces pre-field renders
     *                    BIT FOR BIT: the base profile is returned by
     *                    reference and nothing is copied.
     *                    \warning ONLY 5 OF THE 24 RECORDED VARIANTS CHANGE A
     *                    PIXEL. Four carry curves (PORTRA 800 at EI 1600 and
     *                    3200, ULTRA COLOR 400UC as E-190 prints it and at
     *                    EI 800) and CINESTILL 800T's Cs2 kit carries a gamma
     *                    scale. The other nineteen are the AGFAPAN developer
     *                    records, which differ only in exposure index -- a
     *                    field no stage reads -- so selecting one is a
     *                    deliberate no-op rather than an invented effect.
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
    int32_t processVariant;

    /**
     *  1  NAME           scannerSpecular
     *  2  TYPE           double
     *  3  AE CONTROL     slider
     *  4  UNIT           dimensionless FRACTION on 0..1 -- how specular the
     *                    scanner's illumination is. 0 is fully diffuse, 1 fully
     *                    specular
     *  5  MIN            0
     *  6  MAX            1
     *  7  DEFAULT        0 -- fully diffuse
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
     *                    the field was added under.
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
     *                    monochrome stocks. A realised draw below
     *                    ALGO_SCAN_MIN_SHIFT_PX is discarded, so the effect is
     *                    statically undecidable at small pxPerMm - it depends
     *                    on the RNG draw. Combined with seed and frameIndex, so
     *                    the displacement re-rolls every frame.
     * 13  SCALAR/AVX2    Same semantics; the draws stay in double in both
     *                    builds, the displacement in pixels is float in AVX2.
     * 14  FULL/LITE      PENDING. Expected both - it is a pointwise resample,
     *                    not a neighbourhood pass.
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
     * 11  STAGES         04, 09b, 10, 11, 13, 14, 15, 16. Also passed to 03c,
     *                    which discards it.
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

    // -- physical film damage -----------------------------------------------

    /**
     *  1  NAME           filmDamageEnabled
     *  2  TYPE           bool
     *  3  AE CONTROL     checkbox that greys out the whole damage group
     *  4  UNIT           none - boolean switch
     *  5  MIN            false, by type
     *  6  MAX            true, by type
     *  7  DEFAULT        true   (AlgoControl.cpp, getAlgoControlsDefault)
     *                    !! SEE THE DEFAULT CONFLICT BELOW - this is the
     *                    implemented value, and it is not the value the project
     *                    requirements specify
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
     * 11  STAGES         09b, 15, 16 - the only three consumers
     * 12  INTERACTIONS   Outermost of a three-level gate chain: this flag, then
     *                    damage.damageStrength > 0, then at least one class
     *                    level > 0. It transitively gates frameRate and
     *                    frameIndex at those three stages. It has no
     *                    interaction with seed, because the damage stages use
     *                    damageSeed.
     * 13  SCALAR/AVX2    Same semantics, identical gate chain, including the
     *                    same asymmetry - stage 15 copies then returns, while
     *                    09b and 16 return after an unconditional copy.
     * 14  FULL/LITE      PENDING. Expected Full only, since all three consuming
     *                    stages are on the Lite drop list.
     *
     * !! DEFAULT CONFLICT - REPORTED, NOT RESOLVED
     *   The implementation sets this to TRUE, and AlgoControl.cpp states the
     *   choice deliberately: "This default is now DAMAGED FILM, not clean
     *   film." A default render therefore shows embedded dust, coarse debris
     *   and the occasional fibre.
     *
     *   The project requirements state the opposite - that a clean render must
     *   be the default and damage must be opt-in - and an earlier revision of
     *   this header asserted DEFAULT false while the code assigned true.
     *
     *   This documentation records the IMPLEMENTED value, per the rule that
     *   documented defaults must match the implementation rather than the
     *   intent. It does not change the code. The conflict is an owner decision:
     *   either the default returns to false, or the requirement is amended.
     *
     *   Two consequences of the current default, both easy to be caught by:
     *   a caller wanting the pure film-stock simulation must now explicitly
     *   clear this flag; and this is a deliberate divergence from
     *   film_sim.RenderSettings, which has no damage group at all, so the
     *   verification harness clears the flag before comparing and must keep
     *   doing so.
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
