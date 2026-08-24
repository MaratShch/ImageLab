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
 * TWO GROUPS, AND THE DISTINCTION IS LOAD-BEARING
 *
 *   AlgoControls        every field is read by the renderer today. It mirrors
 *                       film_sim.RenderSettings one-for-one, so a C++ render
 *                       with getAlgoControlsDefault() is directly comparable
 *                       against the Python reference. That comparability is
 *                       what let Algo 02 be verified to 1e-15; keep the two in
 *                       step or the reference stops being a reference.
 *
 *   FilmDamage          specified, NOT yet consumed. A named sub-struct of
 *                       AlgoControls, gated by AlgoControls::filmDamageEnabled.
 *                       Nesting keeps it one object to pass and one place to
 *                       serialise, while the flag keeps the inert fields
 *                       visibly inert: a reader of the live pipeline sees the
 *                       gate is false and knows the whole block is skipped.
 *
 * WHAT IS *NOT* HERE
 *   Film properties. Those live in FilmProfile (see film_profiles.hpp) and are
 *   data, not controls -- 89 stocks of measured and cited values. A control
 *   never replaces a profile number; it scales or overrides it, and every such
 *   field says so explicitly below.
 *
 * SENTINEL CONVENTION
 *   Fields documented as "<0 = use the stock's own default" exist because the
 *   profile carries an era-appropriate value that a user may want to override
 *   without losing it. Passing 0.0 means "genuinely none"; passing -1.0 means
 *   "whatever this stock would have done".
 */


// ===========================================================================
// Damage controls -- SPECIFIED, NOT YET CONSUMED
// ===========================================================================
/**
 * Physical film damage. Every field here is currently inert: no renderer stage
 * reads this struct. It exists so the AE/Premiere panel and the C++ port can
 * be built against a stable layout instead of re-versioning later.
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
 *   An earlier revision of this struct expressed defect rates per second of
 *   running time. That is wrong twice over: it cannot express a still frame at
 *   all, and it conflates a property of the film with a property of the
 *   timeline. Frame rate still enters, but only for the TEMPORAL classes -- how
 *   long a defect persists -- never for how much of it there is.
 *
 * WHAT IS A CONTROL AND WHAT IS A CONSTANT
 *   These seventeen fields are the whole user-facing surface. The defect model
 *   specifies roughly two hundred parameters; the other ~185 are MEASURED FACTS
 *   about film, not choices, and live as named constants in the stage headers
 *   with their measured value and evidence grade in the comment -- exactly as
 *   ALGO_HALATION_KNEE_FRACTION and ALGO_MTF_ADJACENCY_INNER already do.
 *
 *   Examples of what is deliberately NOT here: the dust size exponent
 *   (gamma = 2.6), the clumping field spectral slope (beta = 1.0), scratch width
 *   (26 um), scratch straightness (0.98), the 3.5:1 longitudinal orientation
 *   bias, the median 3.5 per cent contrast amplitude, the weave X:Y ratio and
 *   its 0.8 Hz corner, the T1/T2/T3 population shares. Every one of those is a
 *   measurement or a physical constant. Exposing them would be 200 sliders
 *   nobody touches.
 *
 * ZERO DISABLES, AND IT COSTS NOTHING
 *   Every scale below follows the engine's existing convention: 0 switches the
 *   class off completely, and the corresponding generator is not run and its
 *   buffer not written. Same rule as grainScale, halationScale, coatingScale.
 *
 * THE PROFILE SUPPLIES THE ERA, THE CONTROL SUPPLIES THE INTENT
 *   AgingSpec in each film profile already carries dust_area_ppm,
 *   mottle_amplitude, scratch_rate_base_per_m, dye_fade_c/m/y and dmin_lift.
 *   Those set the era-typical LEVEL; the controls below MULTIPLY them. So a 1943
 *   Agfacolor is dirtier than a VISION3 500T at identical settings, and the
 *   three particulate controls stay correlated by default through the shared
 *   profile figure -- which is what the requirements ask for -- while still
 *   allowing that correlation to be broken deliberately.
 *
 * STATELESSNESS REQUIREMENT
 *   Every generator must be a pure function of (damageSeed, frameIndex,
 *   stageId, ordinal) via a counter-based RNG, with a bounded birth-frame
 *   scan for objects that persist. Any frame must be renderable alone, out of
 *   order, on any thread -- the same rule the coating field already follows.
 *
 *   The five-level seed hierarchy the requirements demand needs no new fields:
 *   L0 stock is the profile index, L1 roll is damageSeed, L2 segment derives
 *   from frameIndex over the segment length, L3 frame is frameIndex, and L4 is
 *   the per-defect ordinal.
 */
struct FilmDamage
{
    // =======================================================================
    //  MASTER
    // =======================================================================

    /// Global severity multiplier on every class below, so one control dials the
    /// whole look without touching the balance between classes.
    /// RANGE 0..4. DEFAULT 1.
    /// AE control: slider.
    double damageStrength;

    /// Seed for every damage generator. Deliberately independent of
    /// AlgoControls::seed so that re-rolling the grain does not also re-roll the
    /// dirt, and vice versa.
    ///
    /// This is also the ROLL seed (level L1): holding it fixed while changing
    /// frameIndex gives different frames OF THE SAME ROLL -- same stains, same
    /// scratches, same fading, different dust.
    /// DEFAULT 20250803.
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

    /// Fine particulate. Scales the profile's era-typical areal density; the
    /// measured central estimate for amateur hand-processed film is 2 /mm2, with
    /// 0.1-0.5 /mm2 for professionally processed, well-stored material.
    /// RANGE 0..4. DEFAULT 0.
    /// AE control: slider.
    double dustLevel;

    /// Coarse opaque lint and chemistry fragments, 0.3-1.5 mm. Rare -- a few per
    /// 35 mm still frame -- but individually conspicuous and fully opaque.
    /// RANGE 0..4. DEFAULT 0.
    double debrisLevel;

    /// Hair and textile fibres. Distinguished from a scratch by near-constant
    /// width, free ends and a curl; a fibre lies ON the film, a scratch is IN it.
    /// RANGE 0..4. DEFAULT 0.
    double fibreLevel;

    /// Clumpiness of all three particulate classes: how much the LOCAL rate
    /// varies across the frame, as a scale on the measured coefficient of
    /// variation of 0.88.
    ///
    /// 0 gives a uniform Poisson scatter, which is the single most common and
    /// most visible failure of existing film-emulation products -- real dirt
    /// arrives in patches, some regions carrying five times the average. 1 is the
    /// measured film behaviour. Above 1 exaggerates it.
    /// RANGE 0..2. DEFAULT 1.
    /// AE control: slider.
    double dirtClumping;

    // =======================================================================
    //  SCRATCHES -- two classes with opposite geometry and opposite temporal
    //  behaviour, which is why they are not one control
    // =======================================================================

    /// Longitudinal transport scratches: long, straight, parallel to film travel,
    /// continuing across frame boundaries. The defining motion-picture defect
    /// ("rain", "tramlines"), and machine-fixed -- it holds a fixed position on
    /// screen for a whole reel while the image moves past it.
    ///
    /// Runs HORIZONTALLY on a still frame and VERTICALLY on every common cine
    /// format, because film travels along the long axis of a still frame and the
    /// short axis of a cine one. That rotation is derived from the format, not
    /// authored.
    /// RANGE 0..4. DEFAULT 0.
    double scratchTransport;

    /// Random handling scratches: short, curved, 0.3-4 mm, individually very
    /// faint but numerous, and generated in bursts because a single wipe leaves
    /// several roughly parallel marks. Locked to the film, not to the machine.
    /// RANGE 0..4. DEFAULT 0.
    double scratchHandling;

    // =======================================================================
    //  PROCESSING AND DRYING
    // =======================================================================

    /// Inverse quality of the development bath: 0 is a professional continuous
    /// machine, 1 is a hand tank in a bathroom. Drives development mottle, air
    /// bells, surge marks, chemical stains and reticulation together, because all
    /// five express the same underlying variable -- agitation quality.
    ///
    /// Air bells, surge and reticulation should be near zero for machine-processed
    /// professional cine stock; that follows from setting this low rather than
    /// from separate switches.
    /// RANGE 0..1. DEFAULT 0.
    double processingQuality;

    /// Drying-stage defects: water spots, tide lines and squeegee bands. A single
    /// 5 mm drying mark covers 2.3 per cent of a 35 mm still frame and 25.6 per
    /// cent of a 16 mm frame, so this control is far more aggressive on small
    /// gauges -- which is correct and is derived, not authored.
    /// RANGE 0..2. DEFAULT 0.
    double dryingMarks;

    // =======================================================================
    //  AGE AND STORAGE
    // =======================================================================

    /// Storage severity: 0 is a cold vault, 1 is a warm attic for forty years.
    /// Drives dye fading and colour crossover, age fog, fungal growth and base
    /// deterioration together, scaling the profile's own AgingSpec figures.
    /// RANGE 0..1. DEFAULT 0.
    double storageSeverity;

    /// Per-layer colour veil: a flat additive level shift in ONE dye record, with
    /// no neutral point anywhere in the tonal range.
    ///
    /// SEPARATE from the crossover that storageSeverity drives, and deliberately
    /// so. Crossover is tone dependent -- shadows lean one way, highlights the
    /// other, mid tones pass through neutral. A veil has no neutral point at all.
    /// Measurement on aged ORWOCOLOR found both on different parts of one roll,
    /// and you cannot produce the second by turning up the first. Merging them
    /// into one "fading" slider is the commonest error in colour-fade emulation.
    ///
    /// Level only: it does NOT reduce contrast in the affected layer. An earlier
    /// analysis claimed it did and was withdrawn on re-measurement.
    /// RANGE 0..2. DEFAULT 0.
    double colourVeil;

    // =======================================================================
    //  MACHINE-SIDE -- motion picture
    // =======================================================================

    /// Gate dirt: the particulate population lodged in the projector or telecine
    /// gate rather than riding on the film. It holds a FIXED SCREEN POSITION for
    /// hundreds to hundreds of thousands of frames, while film-borne dust appears
    /// for exactly one frame and vanishes.
    ///
    /// Separate from dustLevel because the same physical dust splits into two
    /// behaviourally opposite populations, and a simulation implementing only one
    /// of them is immediately identifiable. Still photography has no equivalent
    /// of this distinction, which is why it is easy to miss.
    ///
    /// Uncalibrated: the reference scanner was measurably clean, so no gate-dirt
    /// statistics could be derived. Treat the default as a starting point.
    /// RANGE 0..2. DEFAULT 0.
    double gateDirt;

    /// Film weave: the small per-frame displacement of the image within the gate.
    /// Scales the measured amplitude envelope.
    ///
    /// Weave is the CARRIER that makes every machine-fixed defect convincing.
    /// Film-borne defects move with the image, so they do not move relative to
    /// it; gate-borne defects do not move with the image, so they appear to shift
    /// by exactly the weave amount. That inverse relationship is why a real gate
    /// scratch shimmers instead of sitting still, and a mathematically static
    /// line reads as a digital overlay.
    /// RANGE 0..2. DEFAULT 0.
    double weaveAmount;

    /// Event-localised damage: splices, light leaks, static discharge marks and
    /// cinch marks. These appear, evolve over a few frames to a few hundred, and
    /// disappear -- as opposed to every other class here, which is either
    /// per-frame or permanent.
    ///
    /// A light leak adds EXPOSURE, not brightness, so it passes through the
    /// stock's characteristic curve and lifts shadows dramatically while barely
    /// touching highlights. That is why the defect layer needs access to the
    /// active stock model and cannot be a post-process.
    /// RANGE 0..2. DEFAULT 0.
    double damageEvents;

    /// Temporal exposure flicker, RMS amplitude in stops. Hand-cranked cameras
    /// and early intermittent mechanisms did not deliver equal exposure to
    /// successive frames.
    ///
    /// Acts on EXPOSURE, before the characteristic curve: a brightness change
    /// applied after development is a grade, and a grade does not move highlights
    /// through the shoulder the way a real exposure change does.
    /// RANGE 0..0.5. DEFAULT 0.
    double flickerStops;

    // =======================================================================
    //  SCANNER -- a separate layer, not a film defect
    // =======================================================================

    /// Digitisation artefacts: illumination shading, fixed-pattern banding and
    /// Newton's rings. Switchable independently of everything above, because
    /// "clean scan of damaged film" and "good film, cheap scanner" are both
    /// common requests and merging the two makes neither expressible.
    ///
    /// Worth keeping honest about: in the reference dataset, the strongest
    /// high-frequency texture across 81 crops of damaged film was the SCANNER's
    /// weave pattern, not any film defect. Effects that look like film damage
    /// frequently are not.
    /// RANGE 0..2. DEFAULT 0.
    double scannerArtifacts;
};


// ===========================================================================
// Live controls -- all 21 are consumed by the renderer
// ===========================================================================
struct AlgoControls
{
    // -- film selection -----------------------------------------------------

    /// Which film stock to simulate. Index into the std::vector returned by
    /// film::GetFilmDatabase(), and equivalently line (value + 1) of
    /// film_names.txt.
    ///
    /// WARNING on persistence: profiles are stored alphabetically, so inserting a
    /// stock renumbers every enumerator after it. A saved project must store the
    /// NAME, never this number.
    film::eFILM_PROFILE filmProfile;

    /// Frames per second OF FILM, following any layer time stretch. Consumed by the
    /// temporal stages - exposure flicker, negative-side defects, gate weave and
    /// gate-side defects - all of which specify their rates per SECOND and divide by
    /// this to reach a per-frame rate.
    ///
    /// It belongs here rather than in the call signature because it changes what the
    /// algorithm computes: a defect rate tuned at 24 fps must not run twice as fast
    /// on a 50 fps timeline.
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

    /// Film gauge key, e.g. "super35", "ff35", "16mm", "8mm", "imax15".
    /// Selects frame width in millimetres, which with the render width in
    /// pixels gives px_per_mm -- the mechanism that makes every spatial
    /// quantity (grain, halation, MTF, registration) resolution independent.
    /// Empty string = the stock's own default_format.
    /// AE control: dropdown, populated from FORMAT_GEOM.
    const char* filmFormat;

    /// Print stock key, "" = the stock's own default_print. Nobody looks at a
    /// negative; this second emulsion is what produces correct highlight
    /// rolloff and shadow crush. AE control: dropdown.
    const char* printStock;

    /// Duplication stock for the generation chain. AE control: dropdown.
    const char* dupeStock;

    /// Intermediate interpositive/dupe-negative PAIRS. 0 = camera negative
    /// straight to print. Each generation adds grain and softens detail; the
    /// grain-to-detail ratio worsens monotonically, which is the measurable
    /// signature of a dupe. RANGE 0..4. DEFAULT 0.
    /// AE control: integer slider.
    int32_t generations;

    // -- exposure and tone --------------------------------------------------

    /// Camera exposure offset in stops, applied before the characteristic
    /// curve. This is the exposure the cinematographer chose, not a
    /// brightness trim: it moves the scene along the curve, so the toe and
    /// shoulder do the work. RANGE -4..+4. DEFAULT 0. AE: slider.
    double exposureStops;

    /// Shutter open time in SECONDS, for reciprocity failure. 0 = NOT STATED,
    /// which disables the correction entirely and reproduces every render made
    /// before this field existed, bit for bit. RANGE 0 (off), else 1e-5..3600.
    /// DEFAULT 0. AE: slider, or a text field, since the useful range spans
    /// eight decades.
    ///
    /// NOT a duplicate of exposureStops. That one says how much light the
    /// cinematographer let in and moves the scene along the curve; this one says
    /// over how long, which changes how the EMULSION RESPONDS to the same amount
    /// of light. Beyond a stock's onset the film loses speed (low-intensity
    /// reciprocity failure) and, on colour stock, loses it unequally per layer,
    /// which is why a 10 s frame goes blue on EKTACHROME rather than merely
    /// dark. Below about 1e-4 s the high-intensity branch does the same thing
    /// from the other end.
    ///
    /// SECONDS RATHER THAN SHUTTER ANGLE, deliberately. Angle divided by frame
    /// rate can only ever produce 1/1000 s to 1/24 s, and every sheet in the
    /// corpus prints "no correction needed" across exactly that span -- so an
    /// angle control would be a knob that provably never does anything. The
    /// corrections live where a still photographer works (multi-second
    /// exposures) or where a strobe does (tens of microseconds), and only a time
    /// in seconds reaches either. A shutter-angle convenience that WRITES this
    /// field is a reasonable thing to add on top; it is not a substitute for it.
    ///
    /// WHAT IT DOES NOT MODEL, stated because the limit is a data limit and not
    /// an implementation one: the correction is per CHANNEL and GLOBAL, not per
    /// pixel. Real reciprocity failure is intensity dependent -- the darkest
    /// parts of a frame fail first, so a long exposure loses shadow separation
    /// as well as speed -- but all six measured tables in the database are
    /// functions of time alone, and no source on file carries an intensity axis
    /// to calibrate against. See AlgoReciprocity.hpp.
    double exposureTimeS;

    /// How DIRECTIONAL the reader's optics are, for Callier's coefficient.
    /// 0 = a diffuse integrating sphere; 1 = a condenser or point source, which
    /// sees the stock's full Callier Q. RANGE 0..1. DEFAULT 0, which reproduces
    /// every render made before this field existed, bit for bit.
    ///
    /// ⚠ IT DOES NOTHING ON COLOUR STOCK, BY CONSTRUCTION rather than by
    /// omission: Callier is silver scattering and a chromogenic dye image has
    /// essentially none, so all 93 colour profiles carry Q = 1.0. It moves the
    /// 66 monochrome ones, where a condenser steepens the tone scale by up to
    /// 22 % while mid grey stays put (the print re-times itself, as a lab would).
    ///
    /// ⚠ THE FILM HALF OF THE PRODUCT IS A CLASS ESTIMATE: the two monochrome Q
    /// values are a generator rule, not a document. The geometry half is exact.
    /// That is why the default is 0 rather than some "typical scanner" value.
    /// See AlgoCallier.hpp. AE control: slider.
    double scannerSpecular;

    /// Colour temperature of the light on the subject, kelvin. Combined with
    /// the stock's balance_kelvin this produces the per-layer exposure
    /// mismatch -- daylight on tungsten stock goes blue because of Planck's
    /// law, not because anything is tinted. RANGE 2000..12000. DEFAULT 5500.
    /// AE: slider (or colour-temperature control).
    double sceneKelvin;

    /// How much of that mismatch to correct. 0 = record it exactly as the
    /// stock would; 1 = as if the right conversion filter had been on the
    /// lens. Intermediate values model a partial correction, which is what an
    /// 80A on the wrong stock actually looks like. RANGE 0..1. DEFAULT 0.
    double wbStrength;

    /// Display-linear value that 18 % scene grey should land on. The printer-
    /// light anchor solve hits this exactly, so it is the tone-scale contract
    /// rather than a gain. RANGE 0.02..0.60. DEFAULT 0.18. AE: slider.
    double greyTarget;

    // -- effect scales: each multiplies a PROFILED value, never replaces it --

    /// Grain amplitude multiplier. 0 disables grain entirely (useful for
    /// isolating other stages). 1 = the stock's calibrated RMS granularity.
    /// RANGE 0..4. DEFAULT 1. AE: slider.
    double grainScale;

    /// Halation multiplier. The stock's gains and three-lobe radii are
    /// measured or datasheet-derived; this scales them. Raise toward 2 if a
    /// render reads weaker than a reference scan -- the profiled gain rests
    /// on an assumed highlight overshoot. RANGE 0..4. DEFAULT 1.
    double halationScale;

    /// DIR coupler multiplier -- the LATERAL half of the coupler chemistry
    /// (edge effects, micro-contrast). The vertical half is interimage, which
    /// is profile data with its own iteration count. RANGE 0..3. DEFAULT 1.
    double couplerScale;

    /// Channel misregistration multiplier. 0 = perfect registration.
    /// Technicolor three-strip profiles carry tens of micrometres here, which
    /// is why their edges fringe. RANGE 0..4. DEFAULT 1.
    double misregScale;

    /// Veiling flare fraction of the taking lens. <0 = use the stock's
    /// era-appropriate default_flare. 0 = a perfect modern lens. Flare lifts
    /// the black floor and compresses global contrast, and nothing in the
    /// emulsion model substitutes for it. RANGE -1 (auto), else 0..0.5.
    /// DEFAULT -1.
    double flare;

    /// Whether the print stock contributes its own (finer) grain.
    /// DEFAULT true. AE: checkbox.
    bool printGrain;

    /// Allow the additive-mosaic path on reseau stocks (Dufaycolor, Lumiere).
    /// Auto-disables and warns when the render is too small to represent the
    /// grid -- below ~3 px per cell the output is aliasing noise, not a
    /// mosaic. DEFAULT true. AE: checkbox.
    bool reseau;

    // -- schema v4 / v5 additions -------------------------------------------

    /// Lens corner falloff in stops; <0 = the stock's era default_vignette.
    /// A LENS property, not an emulsion one: cos^4(theta) is geometry and
    /// applies in every era -- modern glass still loses 0.3-0.5 stop wide
    /// open. Coating unevenness cannot produce a corner-locked defect at all,
    /// because film is coated as a wide web and slit afterwards.
    /// RANGE -1 (auto), else 0..4. DEFAULT -1. AE: slider.
    double vignette;

    /// Scales all three CoatingSpec defects together: the web-coherent
    /// coating field, gate buckling, and narrow-gauge edge fog. 0 disables
    /// them. RANGE 0..3. DEFAULT 1. AE: slider.
    double coatingScale;

    /// Frame number within the clip. Only the coating field uses it, to slide
    /// its machine-direction structure by one frame pitch per frame. The
    /// field is a pure function of (seed, absolute web position), so frames
    /// render independently and out of order -- no state, no seams. Set it
    /// from the layer time and the frame rate, NOT from a running counter.
    /// RANGE >= 0. DEFAULT 0.
    int32_t frameIndex;

    // -- determinism --------------------------------------------------------

    /// Master seed. Identical inputs and seed give a bit-identical render;
    /// changing it changes the grain and coating realisation but nothing
    /// physical. DEFAULT 12345. AE: integer field + randomise button.
    int32_t seed;

    // -- physical film damage -----------------------------------------------

    /// Hard gate for the entire FilmDamage block below. false = every damage
    /// generator is skipped at zero cost and `damage` is not read at all.
    /// DEFAULT false: a clean render must be the default, and damage is opt-in.
    /// Checked ONCE per frame, not per pixel.
    /// AE control: checkbox that greys out the whole damage group.
    bool filmDamageEnabled;

    /// Physical damage parameters. Inert until filmDamageEnabled is true AND
    /// the renderer stages that consume them exist. Nested rather than passed
    /// separately so there is one object to hand around and one thing to
    /// serialise with a preset.
    FilmDamage damage;
};


/// Defaults mirroring film_sim.RenderSettings exactly, so a C++ render with
/// these values is directly comparable against the Python reference.
/// Includes damage defaults with filmDamageEnabled = false.
AlgoControls getAlgoControlsDefault (void) noexcept;

/// Damage sub-defaults on their own, for a "reset this group" button.
FilmDamage   getFilmDamageDefault   (void) noexcept;
