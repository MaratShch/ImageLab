#pragma once

// ---------------------------------------------------------------------------
//  AlgoNegativeDefects.hpp
//
//  Sub-stage 9b of the film simulation pipeline: negative-side defects.
//
//  THREE PARTICULATE CLASSES AND ONE ABRASION CLASS ARE NOW MODELLED
//
//      fine dust           the dominant class, and the one that carries the look
//      coarse debris       rare, large, opaque, hard-edged
//      hair and fibres     long, constant width, curved, with free ends
//      scratches           damage IN the film rather than on it: a transport
//                          tramline with a run length along the web, and single
//                          handling marks locked to the film
//
//  The remaining negative-side classes - processing mottle, drying marks, storage
//  fade, colour veil - are separate controls and are not applied here yet.
//
//  ⚠ THE SCRATCH CLASS IS THE ONE PLACE THIS STAGE HAS A TIME AXIS, and the rest
//  of this header's insistence that it has none is still correct. Dust, debris
//  and fibres are baked into the film and only TRANSLATE with the web; nothing is
//  born and nothing dies. A transport scratch is different in kind: it is abraded
//  by continuous contact while the film runs past a fixed object, so it exists
//  over a RUN of web - a length, expressed in frames through the frame pitch -
//  and outside that run the film is clean. That run length is what
//  TemporalSpec::scratch_persistence_frames is, and it is the only field in this
//  stage that reads the frame index for anything but the window position.
//  See the frameIndex note on the prototype below.
//
//  WHAT "NEGATIVE SIDE" MEANS, AND WHY IT IS A SEPARATE STAGE FROM 16
//
//  Damage divides by WHERE IT HAPPENED, and the division decides how it behaves
//  in time.
//
//  A negative-side defect is baked into the film emulsion: a particle embedded in
//  the gelatin while it was swollen and tacky during drying, a camera-gate
//  scratch, a processing mark. It was recorded once and is part of the image for
//  ever, so it survives printing, every subsequent generation duplicates it, and
//  it advances with the film.
//
//  A gate-side defect - stage 16 - happens at projection or telecine: loose dirt
//  on the gate, a hair in the light path. It is applied to the finished positive,
//  does not survive re-printing, and is tied to the machine rather than the film.
//
//  So this stage is applied BEFORE the print at 13, and that ordering is
//  load-bearing: the dupe chain and the print curve act on the damage exactly as
//  they act on the picture. Camera-original damage composited after the print
//  would be sharper and cleaner than the image around it, which reads as a
//  digital overlay instantly.
//
//  THE EMBEDDED / LOOSE SPLIT, AND WHY THIS STAGE ONLY GETS PART OF THE DUST
//
//  Measured dust divides into three temporal populations:
//
//      loose dust on the film during the scan   50 - 80 %   one frame only
//      embedded processing dust                 15 - 40 %   locked to the film
//      dirt lodged in the gate                   5 - 20 %   fixed for a long run
//
//  Only the embedded population is a property of the negative, so only that
//  population belongs here. The one-frame "sparkle" and the gate population are
//  machine-side and belong to stage 16. This stage therefore renders the embedded
//  fraction of the requested level, and the rest arrives when stage 16 is
//  written. Putting all of it here would make every particle survive the print
//  and duplicate through the generations, which is wrong for the majority of
//  real dirt.
//
//  Being explicit about that share also prevents a subtler error. If this stage
//  rendered the whole level, enabling dust at 1.0 would put roughly three times
//  the measured embedded density onto the negative, and it would all be
//  reproducible on re-scan - the exact opposite of the characteristic behaviour,
//  which is that most dirt moves between passes.
//
//  WHY THE DISTRIBUTION IS NOT UNIFORM, AND WHY THAT IS THE WHOLE POINT
//
//  Uniform random scatter is the single most recognisable failure of existing
//  film-emulation products. Real dirt arrives in patches: measured on blank film,
//  the local particle rate has a coefficient of variation near 0.9, so some
//  regions of a frame carry several times the average and others almost none.
//
//  Placement therefore runs through the clumped intensity field in
//  AlgoDefectField.hpp, and the field is keyed on INTEGER FILM COORDINATES rather
//  than on the frame index. A dirty patch of film stays dirty as it travels
//  through the gate, a particle straddling a frame line matches on both sides,
//  and nothing boils.
//
//  RANDOM, BUT NOT CONTROLLABLE, AND NOT REPEATING
//
//  Every particle draws its own size, opacity, shape harmonics, elongation,
//  orientation and colour from its own counter-based stream. Nothing about an
//  individual particle is exposed as a control: a user setting the amount of dust
//  is not choosing how any one speck looks, in the same way that loading a roll
//  of film is not choosing where the dirt lands.
//
//  Determinism and regularity are different things. Every draw is a pure function
//  of (seed, film cell, ordinal), so scrubbing the timeline and rendering frames
//  out of order are both stable, while no two particles and no two patches of
//  film are alike.
//
//  HOW A PARTICLE MEETS THE IMAGE
//
//  This stage works in DENSITY, which is what makes the whole thing cheap. An
//  opaque speck on the negative transmits a fraction (1 - alpha) of the light
//  that reaches it, and density is the negative base-ten logarithm of
//  transmittance, so the particle is exactly an additive density:
//
//      D' = D - log10(1 - alpha)
//
//  No new colour domain, no extra buffer, no separate compositing pass. Stacked
//  particles compose correctly by simple addition, because multiplying
//  transmittances is adding logarithms - so overlapping specks darken the way two
//  real specks would, without any special case.
//
//  Alpha is capped just below one rather than at one. A perfectly opaque particle
//  is an infinite density, which would poison every stage downstream; the cap
//  turns it into a large but finite value that the print stage clips normally.
//
//  DEFECTS ARE NOT ACHROMATIC, AND THIS SURPRISES PEOPLE
//
//  Measured on colour material, the ratio of the smallest to the largest
//  per-channel deviation of a detected speck has a median near 0.55. Particles
//  are coloured - lint is not grey, and a particle sitting in one emulsion layer
//  blocks that layer's light more than the others. Rendering dirt as a neutral
//  density is a visible tell, so each particle carries three channel weights.
//
//  EDGES ARE NEVER SHARPER THAN THE OPTICS
//
//  A particle boundary is rendered with a soft transition whose width is the
//  system point-spread function, floored at half a pixel. Nothing in a real
//  imaging chain has an edge sharper than its own PSF, and a hard-edged speck is
//  the second most reliable way to make dust look composited - the first being
//  uniform placement.
//
//  Shapes are irregular by construction: a radius modulated by low-order angular
//  harmonics, with elongation up to 3:1 for dust and a lobed, concave outline for
//  debris. Circles read as digital immediately.
//
//  ORIENTATION FOLLOWS THE TRANSPORT AXIS
//
//  The mildly elongated members of the measured population are not isotropic:
//  they split 54 per cent along the transport axis against 17 per cent across it,
//  inheriting the same longitudinal bias as the scratches. Elongated particles
//  therefore take a biased orientation, and because the film coordinate system
//  knows which image axis the film runs along, that bias comes out horizontal on
//  a 35 mm still and vertical on every cine format with no per-format code.
//
//  FORMAT SCALING IS AUTOMATIC
//
//  Everything is placed per square millimetre of film over the window this frame
//  occupies, so a 16 mm frame receives about eleven times fewer particles than a
//  35 mm still frame while each one covers about 3.3 times more of the frame
//  width. Both halves of that fall out of the geometry; neither is coded.
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

// Film-fixed coordinates: the window on the film and the transport rotation.
#include "AlgoFilmCoord.hpp"

// The clumped spatial process and the sampling primitives.
#include "AlgoDefectField.hpp"

// Stock parameters, including TemporalSpec and AgingSpec.
#include "film_profiles.hpp"

#include <cstdint>   // int32_t, uint32_t


// ===========================================================================
//  MEASURED CONSTANTS
//
//  Every figure below came from measurement on real film scans, and each carries
//  the number it was measured as. They are constants rather than controls on
//  purpose: they are properties of dirt, not decisions a user makes.
// ===========================================================================

// ---------------------------------------------------------------------------
//  Areal density of fine dust at level 1.0, particles per square millimetre.
//
//  2.0 is the conservative central estimate: measured with a detector that
//  requires a candidate to exceed its local surround by four times the local
//  standard deviation AND by a minimum absolute amount, so that image structure
//  cannot be counted as dirt. A permissive detector on the same frames reports up
//  to 14 per square millimetre, and part of that excess is scanner interpolation
//  rather than particles, which is why the conservative figure anchors level 1.0
//  and the permissive one only bounds the control.
// ---------------------------------------------------------------------------
constexpr HighPrecType ALGO_DUST_DENSITY_PER_MM2 = 2.0;

// ---------------------------------------------------------------------------
//  Hard ceiling on dust density, particles per square millimetre.
//
//  14.0, the most a permissive detector found on the dirtiest frame measured.
//  A level above this is not "dirtier film", it is a different material, so the
//  request is clamped rather than honoured.
// ---------------------------------------------------------------------------
constexpr HighPrecType ALGO_DUST_DENSITY_MAX = 14.0;

// ---------------------------------------------------------------------------
//  Fraction of the dust population that is embedded in the emulsion.
//
//  0.6, the centre of the measured 0.3 - 0.9 range. This is the share that
//  belongs on the negative; the remainder is loose or gate dirt and is stage 16's
//  to render.
// ---------------------------------------------------------------------------
constexpr HighPrecType ALGO_DUST_EMBEDDED_FRACTION = 0.6;

// ---------------------------------------------------------------------------
//  Dust size distribution: truncated power law p(d) proportional to d^-gamma.
//
//  gamma 2.6, fitted over the resolved 18 - 107 micrometre range. Limits 15 and
//  200 micrometres: below 15 the detector cannot separate particles from scanner
//  noise, and above 200 a single power law generates the large particles at the
//  wrong rate, which is why coarse debris is a separate population rather than
//  the tail of this one.
// ---------------------------------------------------------------------------
constexpr HighPrecType ALGO_DUST_SIZE_GAMMA  = 2.6;
constexpr HighPrecType ALGO_DUST_SIZE_MIN_UM = 15.0;
constexpr HighPrecType ALGO_DUST_SIZE_MAX_UM = 200.0;

// ---------------------------------------------------------------------------
//  Dust opacity: median level and dispersion, both SOLVED against the measured
//  output amplitudes rather than assumed.
//
//  ==================== READ THIS BEFORE CHANGING EITHER ====================
//
//  WHY THESE ARE NOT THE OBVIOUS 0.15 TO 1.0
//
//  The first implementation drew opacity as Beta(2,3) scaled onto 0.15 to 1.0,
//  which is the range a reasonable person would pick for "mostly translucent
//  specks, a few opaque ones". Rendered and measured, that produced peak
//  amplitudes with a median of 31 DN of 255 and a 95th percentile of 160 DN,
//  against measured values of 8.8 - 9.8 and 11.2 - 14.3. Three times too strong at
//  the median and eleven times too strong in the tail: the specks read as digital
//  hot pixels rather than as dirt, and it was the single thing most wrong with the
//  render.
//
//  THE DOMAIN ERROR THAT CAUSED IT
//
//  Opacity is drawn in the NEGATIVE DENSITY domain, because that is where a
//  particle physically sits and where this stage works. The measured amplitudes are
//  peak deviations in the FINISHED POSITIVE SCAN. Between the two sit the print
//  exposure, the print characteristic curve, both dye matrices and the scan
//  transfer - and the print gamma AMPLIFIES a density difference rather than
//  passing it through. So an opacity that is numerically sensible as an opacity is
//  several times too large as a cause of output amplitude. The numbers were right
//  for the wrong domain.
//
//  WHY TWO CONSTANTS AND NOT ONE GAIN
//
//  A single gain on the old range was tried first and cannot work. It fixes the
//  median and the tail at different values: a gain of 0.25 lands the 95th
//  percentile inside its measured range but leaves the median at 6.9, and a gain of
//  0.33 lands the median but pushes the tail to 19. The measured distribution is
//  NARROWER than the old range produced - measured p95 over median is about 1.4,
//  while the old range gave 2.1 - so the dispersion is a second, independent error
//  and needs its own parameter.
//
//  HOW THEY WERE SOLVED
//
//  Numerically, on SVEMA_FOTO_65 (named SVEMA_FN_64 at measurement time; renamed
//  2026-08-13, same emulsion) - which is in the database and is the
//  exact stock the amplitude measurements were made on, so the reference
//  configuration is not a stand-in. Dust is rendered alone on a flat field, the
//  population is detected with the same one-DN floor the original measurement used,
//  and the pair is swept until the median peak lands on 9.3 DN (the centre of
//  8.8 - 9.8) and the 95th percentile inside 11.2 - 14.3 simultaneously.
//
//  AND AT THE RESOLUTION THE MEASUREMENT WAS MADE AT, WHICH MATTERS MORE THAN IT
//  LOOKS
//
//  A first solve was run at 1024 pixels across the frame and had to be redone. At
//  that raster one pixel is 24.3 micrometres, while the measured population has a
//  median equivalent diameter of 20 to 34 micrometres - so the typical particle is
//  smaller than a pixel, and its peak is averaged down by however much of the pixel
//  it fails to cover. Calibrating there produces an opacity that is too high,
//  because it is compensating for a blur the original scan did not have.
//
//  The source scan resolved features down to about 18 micrometres, so its sampling
//  was of the order of 12 micrometres per pixel. 24.89 mm of super35 across 2048
//  pixels is 12.2 micrometres per pixel, which is that scale, and the solve was
//  redone there. The difference is not academic: the same constants give a median
//  of 11.0 DN at 12 micrometres per pixel and 9.4 DN at 24, so calibrating at the
//  wrong raster misplaces the amplitude by about fifteen per cent.
//
//  The remaining raster dependence is CORRECT and must not be calibrated away. A
//  real scanner at lower resolution genuinely averages sub-pixel dirt down; that is
//  why dust which is obvious in a 4K scan is nearly invisible in a 2K one. The
//  constants are anchored at the resolution the target was measured at, and every
//  other resolution then follows from the physics rather than from a second
//  constant.
//
//  WHAT THIS DOES NOT CLAIM
//
//  Not that every stock produces these amplitudes. A contrasty print stock renders
//  the same speck harder than a soft one, and that is real behaviour which must
//  survive. What is fixed here is the domain error, once, against the stock the
//  measurement came from.
//
//  The Beta(2,3) shape is retained: it is skewed towards the low end, matching the
//  observation that most particles are partially transmissive and only large ones
//  approach opaque. Only its LOCATION and SCALE are calibrated, so the shape
//  remains the measured one and the two solved numbers carry no shape information
//  of their own.
// ---------------------------------------------------------------------------

/// Median opacity of a fine-dust particle. Sets the amplitude level.
/// Solved: 9.31 DN of 255 median peak, against a measured 8.8 - 9.8.
constexpr HighPrecType ALGO_DUST_ALPHA_MID = 0.13;

/// Fractional half-width of the opacity distribution about that median. Sets the
/// amplitude dispersion. 0 would make every particle equally opaque; 1 would let
/// the weakest reach zero and the strongest twice the median.
/// Solved: 11.91 DN 95th percentile, against a measured 11.2 - 14.3.
constexpr HighPrecType ALGO_DUST_ALPHA_SPREAD = 0.22;

/// Floor under the drawn opacity. The distribution is symmetric about the median in
/// its argument but the mapping is not, so a low draw at a high spread could reach
/// zero or below; a particle with no opacity is not a particle.
constexpr HighPrecType ALGO_DUST_ALPHA_FLOOR = 0.01;

// ---------------------------------------------------------------------------
//  Diameters over which a particle is driven towards fully opaque, micrometres.
//
//  Large particles are not partially transmissive - they are chips of material -
//  so opacity ramps to one over this span rather than being drawn.
//
//  The onset is 200 and full opacity is 300, and the separation of those two
//  numbers from the dust SIZE limit is the point. An earlier version used a single
//  constant equal to the size limit itself, which forced the largest dust in every
//  frame to be perfectly opaque - the top of the size range and the saturation
//  threshold were the same number, so the tail of the population saturated by
//  construction. Measured amplitudes at the 95th percentile came out at 160 DN of
//  255 against a measured 11 - 14, and essentially all of that excess was this.
//
//  The measurement says particles of 0.3 mm and above are effectively opaque and
//  saturate. 0.3 mm is 300 micrometres, which is in the COARSE DEBRIS class, not
//  in this one - fine dust stops at 200. So the ramp is deliberately inert across
//  the whole of the dust range, and exists only so that the behaviour stays
//  correct if the size limit is ever raised into it.
// ---------------------------------------------------------------------------
constexpr HighPrecType ALGO_DUST_OPAQUE_ONSET_UM = 200.0;
constexpr HighPrecType ALGO_DUST_OPAQUE_FULL_UM  = 300.0;

// ---------------------------------------------------------------------------
//  Mean of Beta(2,3), the opacity distribution's own centre.
//
//  2/(2+3) = 0.4, exactly. Needed because the draw is re-centred on the calibrated
//  median, and re-centring requires knowing where the distribution already sits.
//  Written as the closed-form value rather than a decimal so that changing the Beta
//  parameters cannot leave a stale constant behind.
// ---------------------------------------------------------------------------
constexpr HighPrecType ALGO_DEFECT_BETA23_MEAN = 2.0 / 5.0;

// ---------------------------------------------------------------------------
//  Maximum aspect ratio within the dust class.
//
//  3:1. Anything more elongated than this is a fibre by definition and is
//  generated by the fibre class instead, which is why the two populations do not
//  overlap.
// ---------------------------------------------------------------------------
constexpr HighPrecType ALGO_DUST_ASPECT_MAX = 3.0;

// ---------------------------------------------------------------------------
//  Coarse debris: areal density, particles per square millimetre.
//
//  0.0046, from a count of 2 to 6 particles of 0.3 mm equivalent diameter or
//  larger per 864 square millimetre frame. Three orders of magnitude rarer than
//  fine dust and an order of magnitude larger.
// ---------------------------------------------------------------------------
constexpr HighPrecType ALGO_DEBRIS_DENSITY_PER_MM2 = 0.0046;

// ---------------------------------------------------------------------------
//  Coarse debris size: log-normal, median 0.35 mm, sigma of the logarithm 0.5.
//
//  Bounded at 1.5 mm because the largest single particle measured was 0.55 mm
//  equivalent diameter with a 0.77 mm maximum dimension, and the log-normal tail
//  would otherwise occasionally produce something larger than anything observed.
// ---------------------------------------------------------------------------
constexpr HighPrecType ALGO_DEBRIS_MEDIAN_MM   = 0.35;
constexpr HighPrecType ALGO_DEBRIS_SIGMA_LN    = 0.5;
constexpr HighPrecType ALGO_DEBRIS_MAX_MM      = 1.5;

// ---------------------------------------------------------------------------
//  Coarse debris opacity: 0.85 to 1.0.
//
//  Debris is fully opaque by the definition that separates it from fine dust -
//  it is a lint ball or a chemistry crystal, not a speck.
// ---------------------------------------------------------------------------
constexpr HighPrecType ALGO_DEBRIS_ALPHA_MIN = 0.85;
constexpr HighPrecType ALGO_DEBRIS_ALPHA_MAX = 1.0;

// ---------------------------------------------------------------------------
//  Fibres: areal density, fibres per square millimetre.
//
//  0.0012, from a count of about one fibre per affected frame over an 864 square
//  millimetre frame, on a minority of frames. External references show dense
//  fields of fifteen or more, which is the top of the control range rather than
//  the default.
// ---------------------------------------------------------------------------
constexpr HighPrecType ALGO_FIBRE_DENSITY_PER_MM2 = 0.0012;

// ---------------------------------------------------------------------------
//  Fibre length: log-normal, median 4 mm, sigma of the logarithm 0.6.
//
//  Clamped to 1 - 25 mm, the observed span. At 4 mm a fibre crosses about a
//  seventh of a 35 mm still frame, roughly a quarter of a 35 mm Academy frame and
//  about half a 16 mm frame - which is why fibres are so much more intrusive in
//  small gauges, and why the length is in millimetres of film rather than pixels.
// ---------------------------------------------------------------------------
constexpr HighPrecType ALGO_FIBRE_LENGTH_MEDIAN_MM = 4.0;
constexpr HighPrecType ALGO_FIBRE_LENGTH_SIGMA_LN  = 0.6;
constexpr HighPrecType ALGO_FIBRE_LENGTH_MIN_MM    = 1.0;
constexpr HighPrecType ALGO_FIBRE_LENGTH_MAX_MM    = 25.0;

// ---------------------------------------------------------------------------
//  Fibre width: 20 to 80 micrometres, near-constant along the length.
//
//  Constant width is the mechanical signature of a foreign object LYING ON the
//  film. A scratch is damage IN the film and its width varies along its run, so
//  the two are generated by different code and must not share a primitive.
// ---------------------------------------------------------------------------
constexpr HighPrecType ALGO_FIBRE_WIDTH_MIN_UM = 20.0;
constexpr HighPrecType ALGO_FIBRE_WIDTH_MAX_UM = 80.0;

// ---------------------------------------------------------------------------
//  Fibre persistence length, millimetres: the distance over which the centreline
//  forgets its direction.
//
//  2.5, the centre of the measured 1 - 5 mm range for the radius of curvature.
//  This single number is what makes a fibre look like a fibre: too large and it
//  is a straight line indistinguishable from a scratch, too small and it curls
//  into a ball of wool.
// ---------------------------------------------------------------------------
constexpr HighPrecType ALGO_FIBRE_PERSISTENCE_MM = 2.5;

// ---------------------------------------------------------------------------
//  Probability that a fibre ends in a hook, and the hook's turn per step.
//
//  0.3 of fibres carry a distinctive hook or loop at one end. It is the single
//  most recognisable feature that separates a fibre from a scratch by eye, so it
//  is worth the handful of lines it costs. The hook is applied over the last
//  fifth of the length, turning steadily.
// ---------------------------------------------------------------------------
constexpr HighPrecType ALGO_FIBRE_HOOK_PROBABILITY = 0.3;
constexpr HighPrecType ALGO_FIBRE_HOOK_TURN_RAD    = 0.55;
constexpr HighPrecType ALGO_FIBRE_HOOK_FRACTION    = 0.2;

// ---------------------------------------------------------------------------
//  Fibre opacity: 0.5 to 1.0, uniform.
// ---------------------------------------------------------------------------
constexpr HighPrecType ALGO_FIBRE_ALPHA_MIN = 0.5;
constexpr HighPrecType ALGO_FIBRE_ALPHA_MAX = 1.0;

// ---------------------------------------------------------------------------
//  Number of persistent-walk control points per millimetre of fibre.
//
//  4 gives a step of 250 micrometres, comfortably shorter than the 2.5 mm
//  persistence length, so the curvature statistics are well sampled.
//
//  This is the PHYSICS sampling rate, not the drawing rate. The two are separate
//  on purpose: raising it would change the walk itself, because a random walk with
//  more, smaller steps is a different curve, so it cannot be used to fix the
//  appearance of the stroke.
// ---------------------------------------------------------------------------
constexpr int32_t ALGO_FIBRE_STEPS_PER_MM = 4;

// ---------------------------------------------------------------------------
//  Subdivisions inserted between consecutive control points before stroking.
//
//  4, taking the drawn polyline to 16 points per millimetre - about 60 micrometres
//  per segment, which is under one pixel at any raster this engine renders and
//  therefore invisible.
//
//  This exists because the first implementation stroked the control polyline
//  directly and the fibres came out visibly faceted: at 250 micrometres per
//  segment and a typical 40 pixels per millimetre, each straight run was ten
//  pixels long, and a chain of ten-pixel straight lines does not read as a hair.
//  Subdividing with a smooth interpolant fixes the drawing without touching the
//  walk that produced it.
// ---------------------------------------------------------------------------
constexpr int32_t ALGO_FIBRE_SUBDIV = 4;

// ---------------------------------------------------------------------------
//  Maximum persistent-walk control points held for one fibre.
//
//  25 mm at 4 per mm is 100, so 128 is the next power of two above the longest
//  fibre the length clamp permits. Fixed size because the engine allocates nothing.
// ---------------------------------------------------------------------------
constexpr int32_t ALGO_FIBRE_MAX_POINTS = 128;

// ---------------------------------------------------------------------------
//  Maximum drawn polyline points for one fibre.
//
//  128 control points subdivided four ways, plus the closing point. Sized from the
//  two constants above rather than written as a literal, so that changing either
//  cannot silently overflow the store.
// ---------------------------------------------------------------------------
constexpr int32_t ALGO_FIBRE_MAX_DRAW =
    ((ALGO_FIBRE_MAX_POINTS - 1) * ALGO_FIBRE_SUBDIV) + 1;

// ===========================================================================
//  SCRATCHES - the fourth class, and the only one that is DAMAGE TO the film
//  rather than something lying on it.
//
//  WHY IT IS NOT A SECOND FIBRE, EVEN THOUGH IT SHARES THE STROKE PRIMITIVE
//
//  defectFibres already draws long, constant-width, curved marks, and its own
//  comment names the discriminator: end-to-end chord over traced arc length is
//  0.7 - 0.95 for a fibre and about 0.98 for a scratch. So a scratch is that
//  generator at a much higher straightness, with a transport-axis lock on one of
//  its two populations, a different width, a different amplitude and - for the
//  transport population - a RUN LENGTH along the web. The centreline walk, the
//  Catmull-Rom subdivision and the nearest-point stroke are therefore reused
//  rather than copied; only the statistics differ, which is the whole claim
//  being made about the two classes.
//
//  TWO POPULATIONS, ONE PER EXISTING CONTROL
//
//    scratchTransport   a TRAMLINE. A fixed abrader - a guide rail, a chip of
//                       dirt in a roller - touching the moving web. Locked to
//                       the transport axis, at a fixed position ACROSS the web,
//                       present for a run of frames and then gone.
//    scratchHandling    a single event. A wipe, a fingernail, a careless splice.
//                       Short, randomly placed and oriented, locked to the film
//                       and therefore permanent in it - which is not persistence
//                       in the sense below.
//
//  THE POLARITY QUESTION, AND WHY IT IS BOTH SIGNS
//
//  Stage 16's header states this stage's polarity convention: a negative-side
//  defect that BLOCKS printing light is added as DENSITY here, and comes out
//  bright on the finished positive. A scratch has two mechanisms and they sit on
//  opposite sides of that convention:
//
//    cut       the abrader removes emulsion, so less dye or silver survives, so
//              the negative is LESS dense there - a density DECREMENT.
//    burnish   the abrader polishes the base or the supercoat without reaching
//              the image, so the mark scatters light out of the printing beam
//              exactly as a particle would - a density INCREMENT.
//
//  Both are rendered, and the sign is what makes them different. ⚠ The stroke
//  therefore has to be able to SUBTRACT, which no other class here does, and the
//  result has to be floored at zero density - see defectRasteriseFibre.
// ===========================================================================

// ---------------------------------------------------------------------------
//  Scratch width, micrometres on the film.
//
//  26. A measured fact, listed in AlgoControl.hpp's "deliberately NOT here"
//  paragraph beside the dust size exponent and the clumping spectral slope, as
//  one of the ~185 parameters that are properties of film rather than user
//  choices.
//
//  ⚠ CONSTANT ALONG THE RUN HERE, AND THAT IS A SIMPLIFICATION THIS FILE
//  ALREADY ARGUED AGAINST. The fibre width comment says a scratch's width
//  "varies along its run", which is true and is one of the things that separates
//  a scratch from a hair by eye. Nothing in the tree measures that variation -
//  there is one number, 26 um - so inventing a modulation depth for it would be
//  inventing the very kind of figure this header refuses to invent elsewhere.
//  The single measured width is used, and the variation is left for whoever
//  measures it.
// ---------------------------------------------------------------------------
constexpr HighPrecType ALGO_SCRATCH_WIDTH_UM = 26.0;

// ---------------------------------------------------------------------------
//  Scratch straightness: end-to-end chord over traced arc length.
//
//  0.98, from AlgoControl.hpp's measured-constant list, and corroborated inside
//  this subsystem by defectFibres' own comment - "fibres come out at 0.7 to
//  0.95, scratches at 0.98 to 0.99". It is the single number that separates the
//  two classes, so it is what the scratch centreline is generated FROM rather
//  than something checked afterwards.
//
//  HOW A STRAIGHTNESS BECOMES A PERSISTENCE LENGTH
//
//  The fibre walk is a two-dimensional worm-like chain, whose mean square
//  end-to-end distance over a contour length L with persistence length Lp is
//
//      <R^2> = 4 Lp L [ 1 - (2 Lp / L)(1 - exp(-L / 2 Lp)) ]
//
//  Expanding for L much smaller than Lp gives <R^2> / L^2 = 1 - L / (6 Lp), so
//
//      straightness = sqrt(<R^2>) / L  =  1 - L / (12 Lp)
//      Lp = L / (12 (1 - straightness))
//
//  which at 0.98 is Lp = L / 0.24, i.e. a persistence length about 4.2 times the
//  scratch's own length. The expansion is first order and L/Lp is 0.24 here, so
//  it is good to about one per cent - well inside the precision of a straightness
//  quoted to two decimals.
//
//  ⚠ THE SAME FORMULA VALIDATES THE FIBRE CLASS, which is why it is trusted
//  here: ALGO_FIBRE_PERSISTENCE_MM is 2.5 and the median fibre is 4 mm long, so
//  1 - 4 / (12 x 2.5) = 0.87 - in the middle of the 0.7 - 0.95 that defectFibres
//  claims to produce. The two classes are therefore separated by the one measured
//  quantity and nothing else.
// ---------------------------------------------------------------------------
constexpr HighPrecType ALGO_SCRATCH_STRAIGHTNESS = 0.98;

// ---------------------------------------------------------------------------
//  Longitudinal orientation bias of scratches, as a ratio.
//
//  3.5:1 in favour of the transport axis. AlgoFilmCoord.hpp says exactly where
//  this belongs - "the measured bias itself - a 3.5:1 preference for the
//  transport axis among light-polarity curvilinear features - lives as a
//  constant in the scratch generator, not here" - so this is that constant,
//  finally in the place its own comment reserved for it.
//
//  Expressed as a SHARE for the draw, the same form the particulate classes use.
//  ⚠ It is NOT the same number as ALGO_DEFECT_ORIENT_ALONG_SHARE: that is 0.76,
//  from the particulate split of 54 per cent along against 17 per cent across,
//  which is 3.2:1. Two populations, two measurements, two constants - merging
//  them would silently replace one measurement with the other.
//
//  ⚠ AND IT APPLIES ONLY TO THE HANDLING POPULATION. A transport scratch is not
//  biased towards the transport axis, it IS the transport axis - it is drawn by a
//  stationary abrader on a moving web and cannot be at any other angle. Applying
//  a statistical bias to it would be modelling the machine as if it wobbled.
// ---------------------------------------------------------------------------
constexpr HighPrecType ALGO_SCRATCH_ORIENT_RATIO = 3.5;

constexpr HighPrecType ALGO_SCRATCH_ORIENT_ALONG_SHARE =
    ALGO_SCRATCH_ORIENT_RATIO / (ALGO_SCRATCH_ORIENT_RATIO + 1.0);

// ---------------------------------------------------------------------------
//  Median scratch contrast amplitude, as a fraction.
//
//  0.035 - the "median 3.5 per cent contrast amplitude" from AlgoControl.hpp's
//  measured-constant list. Used directly as the peak opacity of the mark, which
//  is a density of -log10(1 - 0.035) = 0.0155 D added or removed.
//
//  ⚠⚠ THE DOMAIN CAVEAT THE DUST CONSTANTS HAD TO BE SOLVED FOR APPLIES HERE
//  AND HAS NOT BEEN SOLVED. Read ALGO_DUST_ALPHA_MID's block: a measured
//  amplitude is a deviation in the FINISHED POSITIVE SCAN, while this constant
//  acts in the NEGATIVE DENSITY domain, and the print gamma between them
//  AMPLIFIES a density difference rather than passing it through. For dust that
//  mismatch made the first implementation three times too strong at the median.
//  The same solve has NOT been run for scratches - there is no measured scratch
//  amplitude distribution in the tree to solve against, only this one median - so
//  the measured figure is used unchanged and the expected direction of the error
//  is stated instead of being hidden: if it is wrong, it is too STRONG, by
//  something of the order of the print gamma. Solving it needs a scratch
//  amplitude measurement, not a different constant here.
//
//  Only a MEDIAN was measured, so no dispersion is drawn and every scratch
//  carries the same amplitude. That is visibly less varied than real damage and
//  is deliberate: a dispersion here would be a number nobody measured.
// ---------------------------------------------------------------------------
constexpr HighPrecType ALGO_SCRATCH_CONTRAST_MEDIAN = 0.035;

// ---------------------------------------------------------------------------
//  Share of scratches that CUT rather than burnish - i.e. that subtract density.
//
//  ⚠ 0.5, AND IT IS A MODELLING CHOICE WITH NO SOURCE BEHIND IT. Nothing in this
//  tree measures how often an abrasion reaches the image layer and how often it
//  only polishes the base. Both mechanisms are real and both are common, and with
//  no information at all the even split is the only defensible statement; it is
//  written here as a named constant precisely so that it is one line to change
//  when somebody measures it, rather than a 0.5 buried in a comparison.
//
//  It is stated as a choice and not as a fact. Do not cite it as measured.
// ---------------------------------------------------------------------------
constexpr HighPrecType ALGO_SCRATCH_CUT_SHARE = 0.5;

// ---------------------------------------------------------------------------
//  Expected number of transport scratches running on the web at once, at
//  scratchTransport = 1.0.
//
//  ⚠ 1.0 IS A DEFINITIONAL ANCHOR, NOT A MEASUREMENT, and the difference matters.
//  AgingSpec carries scratch_rate_base_per_m, which is what a real per-metre rate
//  would multiply - and every one of the 184 stocks ships that structure entirely
//  zero, documented as "fresh". So there is nothing to anchor a rate against, and
//  the honest thing is to define the control instead of pretending to derive it:
//  scratchTransport = 1.0 means ONE running tramline on the web, on average, at
//  any instant. The default of 0.40 therefore means "a tramline about forty per
//  cent of the time", which is what the control's own documentation describes.
//
//  When AgingSpec is populated this becomes the era baseline's multiplicand,
//  exactly as the dust density will.
// ---------------------------------------------------------------------------
constexpr HighPrecType ALGO_SCRATCH_TRANSPORT_COUNT = 1.0;

// ---------------------------------------------------------------------------
//  Arrival ordinals examined per frame for the transport population.
//
//  32. Same construction as stage 16's ALGO_GATE_WINDOW and for the same reason:
//  the population must be DERIVED from a bounded closed-form scan rather than
//  accumulated, because Algorithm_Main is a pure function and a host may ask for
//  frame 900 without ever having rendered 899.
//
//  The window has to cover the exponential run-length tail. Arrivals are spaced
//  one mean run length apart per unit of level, so 32 ordinals is 32 mean runs at
//  level 1.0 and still 8 at the advisory maximum of 4 - and exp(-8) is 3.4e-04 of
//  the population outside the window. Thirty-two closed-form presence tests per
//  frame, two draws each.
//
//  ⚠ NEGATIVE ORDINALS ARE EXAMINED, UNLIKE STAGE 16, and that is not an
//  oversight. A reel has a head, so an arrival before frame zero never happened
//  and stage 16 rightly skips it. A ROLL OF FILM DOES NOT: frame 0 of a clip is
//  an arbitrary point on the web, frameIndex may be negative, and skipping the
//  negative ordinals would make the head of every clip mysteriously clean.
// ---------------------------------------------------------------------------
constexpr int32_t ALGO_SCRATCH_RUN_WINDOW = 32;

// ---------------------------------------------------------------------------
//  Handling scratch areal density, marks per square millimetre, at
//  scratchHandling = 1.0.
//
//  ⚠ 0.01 IS THE SAME KIND OF DEFINITIONAL ANCHOR as the transport count above,
//  and for the same reason - AgingSpec is zero on every stock, so there is no
//  measured rate to scale. What it delivers is stated rather than claimed: 0.01
//  per square millimetre is about 4.6 marks on a super35 frame (464 mm2) and 8.6
//  on a 35 mm still frame (864 mm2), which is the "numerous, individually very
//  faint" population AlgoControl.hpp describes. As with every other class here
//  the figure is per square millimetre OF FILM, so a 16 mm frame gets
//  proportionally fewer without any per-format code.
// ---------------------------------------------------------------------------
constexpr HighPrecType ALGO_SCRATCH_HANDLING_PER_MM2 = 0.01;

// ---------------------------------------------------------------------------
//  Handling scratch length, millimetres: 0.3 to 4.
//
//  The range AlgoControl.hpp gives for the class - "short, curved, 0.3-4 mm".
//  Drawn UNIFORMLY over it, because the source states a range and no shape; a
//  log-normal here would be borrowing the fibre class's distribution and
//  presenting it as this class's.
// ---------------------------------------------------------------------------
constexpr HighPrecType ALGO_SCRATCH_HANDLING_LEN_MIN_MM = 0.3;
constexpr HighPrecType ALGO_SCRATCH_HANDLING_LEN_MAX_MM = 4.0;

// ---------------------------------------------------------------------------
//  Control points per millimetre of handling scratch, and the point stores.
//
//  8 per millimetre, a 125 micrometre step. The persistence length a straightness
//  of 0.98 implies is 4.2 times the mark's own length - at least 1.25 mm even for
//  the shortest mark - so the step is an order of magnitude below it and the
//  curvature statistics are properly sampled. Like ALGO_FIBRE_STEPS_PER_MM this
//  is the PHYSICS rate and not the drawing rate; raising it would change the walk
//  itself. Smoothing between the control points is ALGO_FIBRE_SUBDIV's job, and
//  defectSubdivide is shared.
//
//  The stores are sized from the length clamp rather than written as literals, so
//  that changing either constant cannot silently overflow them.
// ---------------------------------------------------------------------------
constexpr int32_t ALGO_SCRATCH_STEPS_PER_MM = 8;

constexpr int32_t ALGO_SCRATCH_MAX_POINTS =
    (static_cast<int32_t>(ALGO_SCRATCH_HANDLING_LEN_MAX_MM)
     * ALGO_SCRATCH_STEPS_PER_MM) + 1;

constexpr int32_t ALGO_SCRATCH_MAX_DRAW =
    ((ALGO_SCRATCH_MAX_POINTS - 1) * ALGO_FIBRE_SUBDIV) + 1;

// ---------------------------------------------------------------------------
//  Share of elongated particles oriented along the transport axis.
//
//  0.76, from a measured split of 54 per cent along against 17 per cent across.
//  The remainder is drawn isotropically, so the bias is a preference rather than
//  a rule - which is what the measurement shows.
// ---------------------------------------------------------------------------
constexpr HighPrecType ALGO_DEFECT_ORIENT_ALONG_SHARE = 0.76;

// ---------------------------------------------------------------------------
//  Angular spread around the preferred orientation, radians.
//
//  0.35 radians, about twenty degrees. Wide enough that the bias is statistical
//  rather than a set of parallel marks.
// ---------------------------------------------------------------------------
constexpr HighPrecType ALGO_DEFECT_ORIENT_SPREAD_RAD = 0.35;

// ---------------------------------------------------------------------------
//  Smallest per-channel weight a particle may carry, on COLOUR stock only.
//
//  Particles are NOT neutral - measured on colour material, the median ratio of
//  the smallest to the largest per-channel deviation of a detected speck is 0.55.
//  Lint is not grey, and a particle sitting in one emulsion layer blocks that
//  layer's light more than the others. Rendering dirt as a neutral density is a
//  visible tell.
//
//  0.45 rather than the measured 0.55, and for the SAME reason the opacity needed
//  calibrating: 0.55 is the ratio in the finished positive scan, while this
//  constant acts in the negative density domain, and the print chain between them
//  does not preserve the ratio. Solved the same way - rendered on ORWOCOLOR_NC21,
//  which is the colour material the ratio was measured on, and swept until the
//  detected output ratio landed on the measured value. 0.55 in this slot produced
//  0.59 in the output; 0.45 produces 0.56.
//
//  It has no effect on monochrome stock, where a particle is exactly neutral by
//  construction - one silver record cannot carry a colour. See defectChroma.
// ---------------------------------------------------------------------------
constexpr HighPrecType ALGO_DEFECT_CHROMA_MIN = 0.45;

// ---------------------------------------------------------------------------
//  Edge transition width of a rendered particle, micrometres on the film.
//
//  8, of the order of the system point-spread function. A particle must never be
//  rendered with an edge sharper than the optics that would have imaged it.
// ---------------------------------------------------------------------------
constexpr HighPrecType ALGO_DEFECT_EDGE_UM = 8.0;

// ---------------------------------------------------------------------------
//  Floor on the edge width in pixels.
//
//  0.5 px. At coarse rasters the physical PSF is a small fraction of a pixel, and
//  a transition narrower than half a pixel aliases into a hard staircase. This
//  floor is what antialiases sub-pixel dust instead of dropping or squaring it.
// ---------------------------------------------------------------------------
constexpr HighPrecType ALGO_DEFECT_EDGE_MIN_PX = 0.5;

// ---------------------------------------------------------------------------
//  Number of angular harmonics modulating a particle radius, and their depth.
//
//  Three harmonics at 0.16 of the radius for dust; debris uses the same count at
//  0.3 with a concave second harmonic, giving the lobed outline that distinguishes
//  a lint ball from a speck. Circles read as digital, and a polygon costs more and
//  looks no better once the PSF has softened the edge.
// ---------------------------------------------------------------------------
constexpr int32_t      ALGO_DEFECT_HARMONICS      = 3;
constexpr HighPrecType ALGO_DUST_LOBE_DEPTH       = 0.16;
constexpr HighPrecType ALGO_DEBRIS_LOBE_DEPTH     = 0.30;

// ---------------------------------------------------------------------------
//  Cap on the alpha actually applied.
//
//  0.999 gives a maximum added density of 3.0, since -log10(1 - 0.999) = 3. A
//  true 1.0 is an infinite density that would propagate as a not-a-number through
//  every stage after this one.
// ---------------------------------------------------------------------------
constexpr HighPrecType ALGO_DEFECT_ALPHA_CAP = 0.999;

// ---------------------------------------------------------------------------
//  Placement margins, millimetres of film.
//
//  A particle whose centre lies outside the window can still overlap it, so
//  placement covers the window plus a margin of the largest footprint the class
//  can produce. Omitting the margin produces a visible clean border, which is the
//  classic tell of a naive generator.
//
//  Dust: 0.2 mm, comfortably above the 200 micrometre size limit. Debris: 1.6 mm,
//  above the 1.5 mm size clamp. Fibres: 25 mm, the full length clamp, because a
//  fibre's centre can sit a whole length away from the window and still cross it.
// ---------------------------------------------------------------------------
//  Handling scratches: 4 mm, the full length clamp, by the same argument as the
//  fibres - a mark's placement point can sit a whole length outside the window
//  and still cross it. The transport population needs no margin constant: it
//  spans the window by construction and its extent is clipped to the window plus
//  one along-extent, which is geometry rather than a tunable.
// ---------------------------------------------------------------------------
constexpr HighPrecType ALGO_DUST_MARGIN_MM     = 0.2;
constexpr HighPrecType ALGO_DEBRIS_MARGIN_MM   = 1.6;
constexpr HighPrecType ALGO_FIBRE_MARGIN_MM    = 25.0;
constexpr HighPrecType ALGO_SCRATCH_MARGIN_MM  = ALGO_SCRATCH_HANDLING_LEN_MAX_MM;

// ---------------------------------------------------------------------------
//  Generator stream tags.
//
//  Each class gets its own tag so that changing the amount of one class does not
//  re-roll another. Turning the debris level up must not move the dust.
// ---------------------------------------------------------------------------
//  The two scratch populations get one tag each for the same reason: the
//  transport tag keys an ORDINAL along the web, the handling tag keys a placement
//  CELL on the film, and moving one control must not disturb the other.
// ---------------------------------------------------------------------------
constexpr uint32_t ALGO_DEFECT_TAG_DUST_FIELD    = 0x00D05701u;
constexpr uint32_t ALGO_DEFECT_TAG_DUST_CELL     = 0x00D05702u;
constexpr uint32_t ALGO_DEFECT_TAG_DEBRIS_CELL   = 0x00DEB101u;
constexpr uint32_t ALGO_DEFECT_TAG_FIBRE_CELL    = 0x00F1B201u;
constexpr uint32_t ALGO_DEFECT_TAG_SCRATCH_RUN   = 0x005C4A01u;
constexpr uint32_t ALGO_DEFECT_TAG_SCRATCH_CELL  = 0x005C4A02u;


// ---------------------------------------------------------------------------
//  Sub-stage 9b: negative-side defects.
//
//  pSrcR/G/B      density in
//  pDstR/G/B      density out
//  sizeX/sizeY    active pixel extent
//  pitch          row stride in ELEMENTS
//  profile        stock being simulated
//  params         user controls; the film-damage group and damageSeed
//  negWidthMm     frame width on the film, so defect sizes scale with gauge
//  negHeightMm    frame height on the film
//  framePitchMm   web advance per frame, for defects that drift along the film
//  pxPerMm        render resolution
//  frameIndex     clip-relative frame number, used to DRIFT defects along the
//                 web (frameIndex * framePitchMm)
//
//                 ⚠ THIS LINE ONCE READ "defects have lifetimes", WHICH WAS
//                 FALSE OF THE PARTICULATE CLASSES AND STILL IS. Corrected
//                 2026-09-11. Dust, debris and fibres have no birth and no
//                 death: each is baked into the film and translates with the
//                 web, which is exactly why this stage takes the FILM frame
//                 rate rather than the projection one.
//
//                 ⚠ QUEUE P44 IS NOW CLOSED, AND NOT BY WIRING THE FIELD INTO
//                 THE OLD SENTENCE. The claim cost something concrete: a
//                 2026-09-11 field audit read that comment and listed
//                 `TemporalSpec::scratch_persistence_frames` - populated on all
//                 184 stocks - as a field the stage "can use directly". It
//                 could not, because the particulate classes have no lifetime
//                 for it to set, and P44 said correctly that giving it one is a
//                 modelling decision rather than a wiring job.
//
//                 The decision taken is that the field is NOT a birth/death
//                 lifetime for a particle. Its docstring calls it the "mean
//                 lifetime of a RUNNING SCRATCH", and a running scratch is
//                 abraded by continuous contact as the web passes a fixed
//                 object - so it is a RUN LENGTH ALONG THE FILM, which the
//                 frame pitch turns into a length in millimetres. A new
//                 SCRATCH class consumes it that way: each transport scratch
//                 has a start frame and a run, is present while
//                 start <= frameIndex < start + run, and marks exactly the
//                 stretch of web its abrader touched. The particulate classes
//                 are untouched and remain lifetime-free.
//  frameRate      frames per second OF FILM - correct here, because this damage
//                 is baked into the film rather than happening at projection
//  seed           per-call seed, combined with params.damageSeed by the generator
// ---------------------------------------------------------------------------
void AlgoStage09b_NegativeDefects
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
    const AlgoType           framePitchMm,
    const AlgoType           pxPerMm,
    const int32_t            frameIndex,
    const AlgoType           frameRate,
    const uint32_t           seed
) noexcept;
