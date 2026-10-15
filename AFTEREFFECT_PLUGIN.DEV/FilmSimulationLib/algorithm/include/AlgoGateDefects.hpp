#pragma once

// ---------------------------------------------------------------------------
//  AlgoGateDefects.hpp
//
//  Stage 16 of the film simulation pipeline: machine-side defects.
//
//      gate dirt        debris lodged in the gate, fixed in the FRAME for a long
//                       run, slowly accreting and occasionally shedding
//      one-frame dirt   the loose particles riding through on this frame only -
//                       what everyone calls sparkle
//      events           splices and edge damage passing through the aperture
//
//  WHY THIS IS A DIFFERENT STAGE FROM 9b, AND NOT A DUPLICATE OF IT
//
//  Stage 9b renders the same MATERIAL - ordinary dust and debris. What differs is
//  where it sits, and that changes everything the viewer perceives:
//
//                       9b, film-borne              16, machine-side
//      position         locked to the film          locked to the frame
//      survives print   yes, and every dupe         no
//      moves with weave yes                         no
//      polarity         bright on the positive      dark on the positive
//      lifetime         forever                     one frame, or a long run
//
//  In still photography this class does not exist as a separate thing - there is
//  no gate the film passes through twice. In motion picture it is a first-class
//  defect and cannot be left out, because it is most of what an audience actually
//  sees.
//
//  POLARITY IS OPPOSITE, AND THIS IS THE PART WORTH GETTING RIGHT
//
//  A speck embedded in the negative blocks printing light, so that patch of print
//  is never exposed and comes out CLEAR - a bright mark. That is stage 9b, and it
//  is why those particles are added as density before the print.
//
//  A speck in the projector gate blocks projection light, so that patch of screen
//  goes DARK. This stage works on the finished positive in transmittance, and
//  applies
//
//      T' = T * (1 - alpha)
//
//  which darkens. Two populations of the same dirt with opposite polarity is not
//  an inconsistency - it is what a real projected print looks like, and rendering
//  both is a large part of why film reads as film.
//
//  THE THREE TEMPORAL CLASSES, AND WHY THE SPLIT MATTERS MORE THAN THE AMOUNT
//
//      loose dust on the film at scan or projection   50 - 80 %   one frame
//      embedded processing dust                       15 - 40 %   stage 9b
//      dirt lodged in the gate                         5 - 20 %   long run
//
//  This stage owns the first and third. A common and very visible mistake is to
//  give one-frame dirt a little frame-to-frame correlation "to make it less
//  noisy". That is precisely backwards: the complete absence of correlation IS the
//  sparkle, and smoothing it destroys the one cue that says the film is moving.
//
//  PERSISTENT GATE DIRT WITHOUT ANY PERSISTENT STATE
//
//  Gate dirt accretes over a screening and occasionally sheds, so the obvious
//  implementation keeps a running population and steps it once per frame. This
//  engine cannot do that: Algorithm_Main is a pure function, frames may be
//  rendered out of order or in isolation, and a host may ask for frame 900 without
//  ever having rendered frame 899.
//
//  So the population is not accumulated - it is DERIVED. Each candidate slot in a
//  fixed pool draws a birth frame and a lifetime from its own stream, keyed on the
//  reel and the slot ordinal. A slot is present at frame f exactly when
//
//      birth <= f < birth + lifetime
//
//  which is a closed-form test costing two draws and no history. Birth times are
//  the arrivals of a Poisson accretion process and lifetimes are exponential with
//  the shed rate, so the population statistics are the ones the model calls for,
//  and the count still grows through a reel and resets at the change. Scrubbing
//  backwards, rendering frame 900 alone, and rendering the reel in order all give
//  the same dirt.
//
//  GATE DIRT DOES NOT MOVE WITH THE PICTURE
//
//  This stage runs AFTER the weave at 15, so it is not displaced by it, while
//  everything film-borne was. The image jitters underneath stationary gate dirt.
//  That inverse relationship is subtle, entirely free here because it falls out of
//  the stage order, and is one of the strongest cues that the dirt is a physical
//  object in a machine rather than a graphic laid over the picture.
//
//  WHERE IT SITS IN THE FRAME
//
//  Concentrated at the aperture edges and corners, where the plate touches the
//  film and scrapes it, decaying inward over a couple of millimetres. Placement is
//  therefore a one-dimensional process along the aperture perimeter with an
//  inward decay, not a uniform scatter over the frame.
//
//  HONEST GRADING OF THE NUMBERS BELOW
//
//  The dataset these constants come from CANNOT calibrate this class. The scanner
//  used for the measurements was demonstrably free of static particulate, which is
//  a clean negative result but leaves gate dirt uncalibrated. The mechanism and
//  the temporal taxonomy are solid; every rate, size and bias below is an
//  engineering estimate within the documented plausible ranges, and is marked as
//  such at each constant. That is a materially weaker footing than the fine-dust
//  constants at stage 9b, which were solved against measurement, and it should not
//  be presented otherwise.
//
//  What IS anchored is the per-frame event rate, which comes from each stock's own
//  TemporalSpec - populated for all 93 profiles, unlike AgingSpec - so era drives
//  the baseline exactly as it does for weave.
// ---------------------------------------------------------------------------

// Project-wide primitives, included unconditionally as required by the project
// coding standard.
#include "Common.hpp"
#include "CompileTimeUtils.hpp"

// The single source of the engine's numeric types and alignment policy.
#include "AlgoTypes.hpp"

// Buffer layout and the geometry fields that travel with it.
#include "AlgoMemHandler.hpp"

// User-facing controls, pre-validated by the caller.
#include "AlgoControl.hpp"

// Shared defect machinery: hashing, Poisson, power law, blob coverage.
#include "AlgoDefectField.hpp"

// Stock parameters, including TemporalSpec.
#include "film_profiles.hpp"

#include <cstdint>   // int32_t, uint32_t


// ---------------------------------------------------------------------------
//  Length of one reel, in seconds of running time.
//
//  1200, twenty minutes - a 2000 foot 35 mm reel at 24 frames per second. The
//  gate-dirt population resets here, because a reel change means the projectionist
//  has opened the gate and, at minimum, disturbed what was in it.
//
//  Used only to decide when the population restarts; nothing else depends on it.
// ---------------------------------------------------------------------------
constexpr HighPrecType ALGO_GATE_REEL_SECONDS = 1200.0;

// ---------------------------------------------------------------------------
//  Candidate slots examined per frame - the WINDOW, not the population.
//
//  96, and the distinction matters. An earlier version used a pool of 32 slots
//  numbered from the start of the reel, so once 32 particles had ever arrived the
//  gate could accrete nothing more, and the population collapsed as the survivors
//  shed. Measured through the pipeline it ran 61, 32, 0, 2, 3 marks across a reel:
//  monotonically DOWN, where the model requires it to grow.
//
//  The fix is to examine a sliding window of arrival ordinals ending at the
//  current frame rather than a fixed pool starting at the reel head. A particle
//  can only be alive if it arrived recently enough, so the window need only be
//  long enough to cover the exponential lifetime tail: at the rates below it spans
//  24000 frames, nearly five mean lifetimes, so under one per cent of survivors
//  fall outside it.
//
//  The per-frame cost is 96 closed-form presence tests - two draws and a compare
//  each - which is trivial beside rasterising the twenty or so that are alive.
// ---------------------------------------------------------------------------
constexpr int32_t ALGO_GATE_WINDOW = 96;

// ---------------------------------------------------------------------------
//  Particles left in the gate at the start of a reel, at level 1.0.
//
//  4, inside the documented 0 - 20 range: what the projectionist did not clean out
//  when the reel was changed. Below the steady-state figure below, so that the
//  population GROWS through a reel rather than decaying towards equilibrium -
//  which is the behaviour the model requires and the behaviour a real gate has,
//  since it is cleaned between screenings rather than reaching a balance.
// ---------------------------------------------------------------------------
constexpr HighPrecType ALGO_GATE_INITIAL_COUNT = 4.0;

// ---------------------------------------------------------------------------
//  Accretion and shed rates, per frame, at level 1.0.
//
//  Accretion 4e-3 and shed 2e-4, both inside the documented ranges of 1e-4 to 1e-2
//  and 1e-4 to 1e-3.
//
//  The pair is chosen for the population it implies, not independently. Mean
//  lifetime is one over the shed rate, 5000 frames; steady-state population is the
//  accretion rate times that lifetime, 4e-3 x 5000 = 20 particles - the top of the
//  documented 0 - 20 range. Starting from 4 the count therefore climbs towards 20
//  with a time constant of 5000 frames, about three and a half minutes at 24
//  frames per second, and resets at the reel change.
//
//  That is the whole point of the class: a gate visibly gets dirtier as a
//  screening goes on.
//
//  Engineering estimates. See the grading note in the file header.
//
//  ⚠ RETUNED 2026-09-04, AND THE OLD PAIR IS THE REASON THE DEFECT LAYER READ AS
//  A DIRTY MONITOR RATHER THAN AS FILM. Shed 2e-4 is a mean identity lifetime of
//  FIVE THOUSAND FRAMES - three minutes and twenty-eight seconds at 24 fps. For
//  all that time a gate mark held the same pixel, the same outline, the same
//  opacity, because every one of those is drawn once from the slot's own stream
//  and never touched again. A ten-second clip is 240 frames, so nothing in the
//  gate population changed at all over an entire shot: the population model was
//  right and its TIME CONSTANT was two orders of magnitude too slow to be seen
//  as anything but a stain on the glass.
//
//  Both rates now sit at the TOP of their documented ranges (1e-4 to 1e-2 for
//  accretion, 1e-4 to 1e-3 for shed) rather than in the middle. Mean identity
//  lifetime becomes 1000 frames - 42 seconds - and the steady state is
//  1e-2 x 1000 = 10 particles, still inside the documented 0 - 20 and still
//  above the initial count, so the gate still visibly dirties through a reel.
//
//  ⚠ AND THE LIFETIME ALONE WAS NEVER GOING TO BE ENOUGH. Even at 42 seconds a
//  mark is pixel-identical while it lives, which is exactly what the eye reads
//  as a display defect. The per-frame jitter below is the other half of the fix
//  and is the part that matters most on a short clip.
// ---------------------------------------------------------------------------
constexpr HighPrecType ALGO_GATE_ACCRETION_PER_FRAME = 1.0e-2;
constexpr HighPrecType ALGO_GATE_SHED_PER_FRAME      = 1.0e-3;

// ---------------------------------------------------------------------------
//  The stock's own dirt-event rate at which the three constants above describe
//  the gate population.
//
//  3.0 events per frame, which is what TemporalSpec carries for 1930s and 1940s
//  material - the dirtiest thing in the database.
//
//  ⚠ THIS EXISTS BECAUSE THE GATE POPULATION WAS NOT SCALED BY THE STOCK AT ALL,
//  AND THAT IS A DEFECT RATHER THAN A TASTE DECISION. The one-frame class below
//  multiplies by profile.temporal.dirt_events_per_frame, which spans 0.05 to 3.0
//  across the database - a factor of sixty. The gate class multiplied by the USER
//  LEVEL only, so a pristine 2010s VISION3 negative and a 1937 nitrate print were
//  given the SAME twenty gate particles. Measured on the live database: 68 stocks
//  carry 0.1 events per frame, so on the most common stock in the corpus the gate
//  produced roughly twenty standing marks against 0.075 one-frame specks - a
//  ratio of 267 to 1, when the project's own temporal taxonomy puts 50 to 80 per
//  cent of all dirt in the ONE-FRAME population.
//
//  Dividing by this reference restores the taxonomy without touching either
//  class's own statistics: at 3.0 the gate behaves exactly as the constants
//  above describe, at 0.1 it produces about a third of one standing particle,
//  and the one-frame population dominates on modern stock as it must.
// ---------------------------------------------------------------------------
constexpr HighPrecType ALGO_GATE_RATE_REFERENCE = 3.0;

// ---------------------------------------------------------------------------
//  PER-FRAME JITTER OF A LIVING GATE MARK.
//
//  ⚠ WHAT THIS FIXES, STATED AS THE FAILURE IT IS. A gate particle used to be a
//  dead stamp: position, aspect, angle, harmonic phases, radius and opacity were
//  all drawn once when the slot was created and reproduced bit for bit on every
//  frame the particle survived. Rendered as a sequence that is not a photographic
//  defect at all - it is a scratch on the lens of the projector, or a dead spot on
//  a monitor, and the eye names it as such immediately.
//
//  A real particle wedged at the aperture edge is being scraped by film moving
//  past it at 456 mm per second. It shifts within its lodging, it rocks, it partly
//  lifts and re-seats, and its effective opacity changes as it does. It stays in
//  the SAME REGION - that is what makes it gate dirt rather than loose dirt - but
//  it is never in the same place twice.
//
//  All three quantities are pure functions of (slot counter, frameIndex), so the
//  statelessness requirement is untouched: any frame still renders alone, out of
//  order, on any thread.
//
//  WANDER  60 micrometres of positional wobble, peak, in each axis. Chosen as
//          half the median particle diameter (0.12 mm): enough that the mark is
//          visibly alive, small enough that it stays the same piece of dirt in
//          the same corner of the frame rather than wandering across it.
//  ANGLE   0.35 radians of rocking, peak - about twenty degrees. An irregular
//          lump seated on a moving surface rocks; it does not spin.
//  ALPHA   modulated between 0.55 and 1.00 of its own drawn opacity, so the mark
//          breathes rather than pulsing between visible and invisible.
//  BLINK   0.10 - one frame in ten the film lifts the particle clear of the
//          aperture entirely and it is simply absent. This is what breaks the
//          last of the stamp: a defect that is present on every single frame of a
//          shot cannot read as physical however much it wobbles.
//
//  Engineering estimates, every one of them. Nothing measures the motion of dirt
//  inside a projector gate, and the file header's grading note applies.
// ---------------------------------------------------------------------------
constexpr HighPrecType ALGO_GATE_JITTER_UM        = 60.0;
constexpr HighPrecType ALGO_GATE_JITTER_ANGLE_RAD = 0.35;
constexpr HighPrecType ALGO_GATE_JITTER_ALPHA_MIN = 0.55;
constexpr HighPrecType ALGO_GATE_BLINK_PROB       = 0.10;

// ---------------------------------------------------------------------------
//  Inward decay of the edge bias, millimetres.
//
//  1.0, inside the documented 0 - 2 mm. Gate dirt collects where the aperture
//  plate contacts and scrapes the film, so its distance inward from the aperture
//  edge is exponential with this mean rather than uniform over the frame.
// ---------------------------------------------------------------------------
constexpr HighPrecType ALGO_GATE_EDGE_DECAY_MM = 1.0;

// ---------------------------------------------------------------------------
//  Share of gate dirt that ignores the edge bias entirely.
//
//  0.15. Not everything in a gate is at its edge - some debris lands in the
//  middle of the aperture - and a population entirely confined to the border reads
//  as a frame decoration rather than as dirt.
// ---------------------------------------------------------------------------
constexpr HighPrecType ALGO_GATE_INTERIOR_SHARE = 0.15;

// ---------------------------------------------------------------------------
//  Gate-dirt particle size: log-normal, median and dispersion, millimetres.
//
//  Median 0.12 mm with sigma of the logarithm 0.7, clamped to 0.03 - 0.8 mm.
//
//  Substantially larger than the fine dust at stage 9b, and that is the point: an
//  agglomeration that has been sitting in a gate collecting more debris is not a
//  20 micrometre speck. It is also why this class stays visible at low delivery
//  resolutions where fine dust averages away - 0.12 mm is three pixels at standard
//  definition and nineteen at 4K.
//
//  Engineering estimate.
// ---------------------------------------------------------------------------
constexpr HighPrecType ALGO_GATE_SIZE_MEDIAN_MM = 0.12;
constexpr HighPrecType ALGO_GATE_SIZE_SIGMA_LN  = 0.7;
constexpr HighPrecType ALGO_GATE_SIZE_MIN_MM    = 0.03;
constexpr HighPrecType ALGO_GATE_SIZE_MAX_MM    = 0.80;

// ---------------------------------------------------------------------------
//  Gate-dirt opacity range.
//
//  0.35 to 0.95, uniform. Higher than fine dust because this material is
//  agglomerated and thick, but not driven to one: a fully opaque mark reads as a
//  hole punched in the picture.
// ---------------------------------------------------------------------------
constexpr HighPrecType ALGO_GATE_ALPHA_MIN = 0.35;
constexpr HighPrecType ALGO_GATE_ALPHA_MAX = 0.95;

// ---------------------------------------------------------------------------
//  One-frame dirt: size distribution, millimetres.
//
//  Median 0.08 mm, sigma of the logarithm 0.8, clamped to 0.02 - 0.6 mm. Smaller
//  than gate dirt, because this material has not had time to agglomerate, but
//  still far larger than embedded fine dust - the loose population is what falls
//  onto film from the air and off the film's own edges, not what was pressed into
//  the emulsion during drying.
// ---------------------------------------------------------------------------
constexpr HighPrecType ALGO_SPARKLE_SIZE_MEDIAN_MM = 0.08;
constexpr HighPrecType ALGO_SPARKLE_SIZE_SIGMA_LN  = 0.8;
constexpr HighPrecType ALGO_SPARKLE_SIZE_MIN_MM    = 0.02;
constexpr HighPrecType ALGO_SPARKLE_SIZE_MAX_MM    = 0.60;

// ---------------------------------------------------------------------------
//  One-frame dirt opacity range.
//
//  0.25 to 0.90, uniform. Loose material sits ON the film rather than in it, so it
//  is less consistently dense than something wedged in a gate.
// ---------------------------------------------------------------------------
constexpr HighPrecType ALGO_SPARKLE_ALPHA_MIN = 0.25;
constexpr HighPrecType ALGO_SPARKLE_ALPHA_MAX = 0.90;

// ---------------------------------------------------------------------------
//  Share of the profile's per-frame dirt-event rate that is one-frame dirt.
//
//  0.75. The stock's TemporalSpec carries dirt_events_per_frame - 0.1 for modern
//  material, 3.0 for 1930s and 1940s - and the temporal taxonomy puts 50 to 80 per
//  cent of dirt in the one-frame population. The upper part of that range is used
//  because the gate share is modelled separately and explicitly below, so this
//  fraction covers loose dirt alone.
// ---------------------------------------------------------------------------
constexpr HighPrecType ALGO_SPARKLE_EVENT_SHARE = 0.75;

// ---------------------------------------------------------------------------
//  Largest one-frame dirt count drawn for a single frame.
//
//  64. A hard bound on the Poisson draw so that a pathological rate cannot make one
//  frame cost unboundedly more than its neighbours. At the highest era rate of 3.0
//  events per frame the mean is far below this, so it never binds in practice.
// ---------------------------------------------------------------------------
constexpr int32_t ALGO_SPARKLE_MAX_PER_FRAME = 64;

// ---------------------------------------------------------------------------
//  Splice events: mean interval in seconds at level 1.0, and duration in frames.
//
//  One splice every 90 seconds of running time, visible for 2 frames.
//
//  A splice is where two pieces of film were joined - with cement or tape - and it
//  passes through the aperture as a bar across the frame, usually with a jump in
//  density either side of it. Two frames because the join is thicker than the film
//  and unseats the following frame in the gate as well, which is why a splice is
//  seen rather than merely passed.
//
//  Engineering estimate; splice frequency depends entirely on how heavily a print
//  was repaired.
// ---------------------------------------------------------------------------
constexpr HighPrecType ALGO_SPLICE_INTERVAL_SECONDS = 90.0;
constexpr int32_t      ALGO_SPLICE_FRAMES           = 2;

// ---------------------------------------------------------------------------
//  Splice bar: thickness as a fraction of the frame's transport extent, and the
//  density of the bar itself.
//
//  0.035 of the frame, at 0.55 opacity. The bar runs ACROSS the film, at right
//  angles to the transport direction, because that is how film is cut and joined -
//  and because the transport axis is derived from the format geometry, that comes
//  out horizontal on a 35 mm still and vertical on every cine gauge with no
//  per-format code.
// ---------------------------------------------------------------------------
constexpr HighPrecType ALGO_SPLICE_THICKNESS_FRAC = 0.035;
constexpr HighPrecType ALGO_SPLICE_ALPHA          = 0.55;

// ---------------------------------------------------------------------------
//  Edge transition width for machine-side dirt, micrometres on the film.
//
//  25, floored at half a pixel. Deliberately wider than the 8 micrometres used for
//  film-borne particulate at stage 9b, and for a physical reason: gate dirt is not
//  in the film plane. It sits on the aperture plate, a fraction of a millimetre out
//  of focus, so the projection or scanning optics image it softer than anything on
//  the emulsion. Rendering it as sharp as embedded dust is a subtle tell that it
//  was composited rather than photographed.
// ---------------------------------------------------------------------------
constexpr HighPrecType ALGO_GATE_EDGE_UM     = 25.0;
constexpr HighPrecType ALGO_GATE_EDGE_MIN_PX = 0.5;

// ---------------------------------------------------------------------------
//  Aspect ratio and lobe depth for machine-side dirt.
//
//  Up to 2.5:1 elongation and a 0.28 harmonic depth. Agglomerated debris is lumpy
//  and irregular rather than round; circles read as digital.
// ---------------------------------------------------------------------------
constexpr HighPrecType ALGO_GATE_ASPECT_MAX = 2.5;
constexpr HighPrecType ALGO_GATE_LOBE_DEPTH = 0.28;

// ---------------------------------------------------------------------------
//  Generator stream tags.
//
//  One per population, so that changing the amount of one does not disturb another.
//  Turning the events up must not move the gate dirt.
// ---------------------------------------------------------------------------
constexpr uint32_t ALGO_GATE_TAG_SLOT    = 0x0A7ED17u;
constexpr uint32_t ALGO_GATE_TAG_SPARKLE = 0x05A2C1Eu;
constexpr uint32_t ALGO_GATE_TAG_SPLICE  = 0x05911CEu;
//: Per-frame jitter of a LIVING slot, 2026-09-04. Its own tag so that the
//: wobble stream cannot disturb the birth, lifetime and shape draws above -
//: turning the jitter off must leave every particle exactly where it was.
constexpr uint32_t ALGO_GATE_TAG_JITTER  = 0x0117E20u;


// ---------------------------------------------------------------------------
//  Stage 16: machine-side defects.
//
//  pSrcR/G/B      display-linear transmittance in
//  pDstR/G/B      out
//  sizeX/sizeY    active pixel extent
//  pitch          row stride in ELEMENTS
//  profile        stock being simulated; TemporalSpec supplies the era event rate
//  params         user controls; damage.gateDirt and damage.damageEvents
//  negWidthMm     frame width on the film, so sizes scale with gauge
//  negHeightMm    frame height on the film
//  pxPerMm        render resolution
//  frameIndex     clip-relative frame number; this stage is entirely temporal
//  frameRate      frames per second OF PROJECTION - correct here, because this
//                 damage happens at the machine rather than on the film
//  seed           per-call seed, combined with params.damage.damageSeed
// ---------------------------------------------------------------------------
void AlgoStage16_GateDefects
(
    const AlgoType* RESTRICT pSrcR,
    const AlgoType* RESTRICT pSrcG,
    const AlgoType* RESTRICT pSrcB,
    AlgoType* RESTRICT       pDstR,
    AlgoType* RESTRICT       pDstG,
    AlgoType* RESTRICT       pDstB,
    const int32_t            sizeX,
    const int32_t            sizeY,
    const int32_t            pitch,
    const film::FilmProfile& profile,
    const AlgoControls&      params,
    const AlgoType           negWidthMm,
    const AlgoType           negHeightMm,
    const AlgoType           pxPerMm,
    const int32_t            frameIndex,
    const AlgoType           frameRate,
    const uint32_t           seed
) noexcept;
