#pragma once
// ---------------------------------------------------------------------------
//  film_damage_constants.hpp
//
//  EVERY COMPILE-TIME CONSTANT OF THE FILM-DAMAGE LAYER, in one place
//  (owner request, 2026-10-20c). Stage 9b (negative defects), 9b processing
//  defects, 15 (gate weave), 16 (machine-side dirt, sparkle, splices, scanner
//  artifacts), the film coordinate frame and the Cox field they share, and the
//  storage-severity mapping of damage.storageSeverity.
//
//  The constants were MOVED here verbatim, each with the rationale block that
//  documents it, from the headers named in the section banners below; their
//  values and types are unchanged (renders bit-identical, verified). Those
//  headers include this file, so every existing #include keeps working.
//
//  NOT HERE, deliberately:
//    * control ranges, defaults and steps -> AlgoControlEnums.hpp (UI limits,
//      including the owner's Splice & Tear block);
//    * ALGO_DISPLAY_CLEAR_WHITE -> AlgoDisplayWhite.hpp (applies to every
//      render, not only to damage);
//    * the Wilhelm temperature table (ALGO_STORAGE_TEMP_*) -> AlgoStorageAge.hpp
//      (also drives storageYears, which is not a damage control).
//
//  Shared by the scalar and the AVX2 trees. HighPrecType / AlgoType come from
//  the including tree's AlgoTypes.hpp, exactly as before.
// ---------------------------------------------------------------------------

#include "Common.hpp"
#include "CompileTimeUtils.hpp"
#include "AlgoTypes.hpp"

#include <cstdint>   // int32_t, uint32_t


// ===========================================================================
//  from AlgoFilmCoord.hpp
// ===========================================================================

// ---------------------------------------------------------------------------
//  Smallest frame pitch treated as a real transport, millimetres.
//
//  Sheet and pack formats report exactly zero. The threshold is well below any
//  real pitch (Regular 8 is 3.81 mm, the smallest in the database) so it cannot
//  misclassify a genuine gauge.
// ---------------------------------------------------------------------------
constexpr AlgoType ALGO_FILM_MIN_PITCH_MM = static_cast<AlgoType>(0.5);

// ---------------------------------------------------------------------------
//  Longitudinal anisotropy strength for transported film.
//
//  1.0 means "apply the measured bias in full". The measured bias itself - a
//  3.5:1 preference for the transport axis among light-polarity curvilinear
//  features - lives as a constant in the scratch generator, not here; this field
//  only says whether the film HAS a transport direction at all.
// ---------------------------------------------------------------------------
constexpr AlgoType ALGO_FILM_ANISOTROPY_TRANSPORTED = static_cast<AlgoType>(1.0);


// ===========================================================================
//  from AlgoDefectField.hpp
// ===========================================================================

// ---------------------------------------------------------------------------
//  Measured coefficient of variation of the intensity field.
//
//  0.88, the centre of the 0.83 - 0.92 range obtained by inverting the observed
//  dispersion index at three cell areas. Grade E.
// ---------------------------------------------------------------------------
constexpr HighPrecType ALGO_DEFECT_FIELD_CV = 0.88;

// ---------------------------------------------------------------------------
//  Spectral slope of the modulating field, as the exponent beta in 1/f^beta.
//
//  1.0 - near-pink. This is what makes the coefficient of variation almost
//  independent of measurement scale, which is what was observed: CV changed by
//  less than a tenth across a sixteen-fold range of cell areas. A single-scale
//  blob field cannot do that. Grade E for the observation, P for the inference.
// ---------------------------------------------------------------------------
constexpr HighPrecType ALGO_DEFECT_FIELD_BETA = 1.0;

// ---------------------------------------------------------------------------
//  Octave structure of the value-noise field.
//
//  Finest lattice spacing 1 mm, six octaves, so the field carries structure at
//  1, 2, 4, 8, 16 and 32 mm. That spans the 1 - 20 mm range over which the
//  scale-free behaviour was measured, with one octave of headroom at each end.
//
//  Six is chosen so the coarsest octave exceeds the largest frame dimension in
//  the database that has a transport axis, and the finest is at the scale where
//  the measurement's smallest cell was taken.
// ---------------------------------------------------------------------------
constexpr HighPrecType ALGO_DEFECT_FIELD_BASE_MM = 1.0;
constexpr int32_t      ALGO_DEFECT_FIELD_OCTAVES = 6;

// ---------------------------------------------------------------------------
//  Cell size for stratified placement, millimetres square.
//
//  One millimetre, matching the finest lattice of the field and the smallest cell
//  the dispersion index was measured at. Small enough that the intensity is
//  effectively constant within a cell - which is what makes the stratified draw
//  equivalent to sampling the continuous process - and large enough that the
//  per-cell Poisson count stays in single figures at realistic densities.
// ---------------------------------------------------------------------------
constexpr HighPrecType ALGO_DEFECT_CELL_MM = 1.0;

// ---------------------------------------------------------------------------
//  Hard cap on particles generated from one cell.
//
//  A guard, not a model parameter. The Poisson draw below is unbounded in
//  principle, and a pathological intensity - a control driven far past its
//  documented range, say - could otherwise ask for millions of particles from one
//  square millimetre. Thirty-two is roughly ten standard deviations above the
//  highest per-cell mean any documented density produces.
// ---------------------------------------------------------------------------
constexpr int32_t ALGO_DEFECT_MAX_PER_CELL = 32;

// ---------------------------------------------------------------------------
//  Variance retained by one octave of Hermite-interpolated value noise.
//
//  A lattice of independent unit-variance values, smoothly interpolated, does NOT
//  have unit variance away from the lattice points. A point inside a cell is a
//  weighted mean of four corner values whose weights sum to one, so its variance
//  is sum(w^2), which is 1 only exactly at a corner.
//
//  Averaged over the cell this is computable in closed form. With
//  W = smoothstep(u) = 3u^2 - 2u^3 and u uniform on [0,1]:
//
//      E[W]   = 1/2
//      E[W^2] = 9/5 - 12/6 + 4/7 = 13/35
//      one axis : E[(1-W)^2 + W^2] = 1 - 2E[W] + 2E[W^2] = 26/35
//      two axes : (26/35)^2 = 676/1225 = 0.551837
//
//  Measured empirically over 400 000 samples of the full six-octave field:
//  0.5567. The small excess over 0.551837 is the sampling grid interacting with
//  the octave spacings.
//
//  This factor MUST be divided out, and getting it wrong is not cosmetic: without
//  it the field's sigma is 0.746 instead of 1.0, the delivered coefficient of
//  variation is 0.63 instead of the measured 0.88, and the index of dispersion
//  comes out roughly three times too low at every scale. Which is exactly the
//  under-clustered look this whole Cox machinery exists to avoid.
// ---------------------------------------------------------------------------
constexpr HighPrecType ALGO_DEFECT_INTERP_VARIANCE = 676.0 / 1225.0;

// ---------------------------------------------------------------------------
//  ALGO_DEFECT_BLOB_HARMONICS
//
//  Angular harmonics modulating a particle's radius, shared by every class that
//  draws a compact blob.
//
//  Three, starting at the second. The first would merely translate the shape,
//  which the particle's centre already expresses, so including it would waste a
//  term and make the centre ambiguous.
// ---------------------------------------------------------------------------
constexpr int32_t ALGO_DEFECT_BLOB_HARMONICS = 3;


// ===========================================================================
//  from AlgoNegativeDefects.hpp
// ===========================================================================

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
//  result has to be floored at zero density - see defectRasteriseStroke (2026-10-20; was defectRasteriseFibre).
// ===========================================================================

// ---------------------------------------------------------------------------
//  Scratch width, micrometres on the film - one median per population, a
//  log-normal spread between marks, and a smooth variation ALONG each mark.
//
//  ⚠ REPLACED 2026-10-20 (owner approval of the stochastic defect model). The
//  previous single figure, 26 um, was listed in AlgoControl.hpp as measured but
//  no document in the tree records how or on what; it is superseded, not lost
//  (doc/RESULT_2026-10-20_defect_optics.md §2 keeps it).
//
//  NEW FIGURES, PROVISIONAL, from the owner's reference collage
//  (DEFECTS/Film_Defect_Reference_Collage_9x9.png, «statistical reference
//  only»), measured on its 8.15 um/px raster: transport tramlines FWHM 54 um
//  and handling marks 47 um, which a Gaussian scan PSF of the measured dust
//  edge width de-blurs to about 45 and 38 um. Between-mark spread sigma_ln 0.22
//  (handling, measured; applied to both). Along one mark: transport CV 0.065
//  correlated over about 2.5 mm, handling CV 0.20 over 0.25 mm.
//
//  Log-normal because a width is a positive scale quantity produced by many
//  multiplicative causes (load, abrader tip, emulsion hardness); a normal would
//  allow negative widths at the tail.
// ---------------------------------------------------------------------------
constexpr HighPrecType ALGO_SCRATCH_TRANSPORT_WIDTH_UM  = 45.0;
constexpr HighPrecType ALGO_SCRATCH_HANDLING_WIDTH_UM   = 38.0;
constexpr HighPrecType ALGO_SCRATCH_WIDTH_SIGMA_LN      = 0.22;
constexpr HighPrecType ALGO_SCRATCH_TRANSPORT_WIDTH_CV  = 0.065;
constexpr HighPrecType ALGO_SCRATCH_TRANSPORT_WIDTH_CORR_MM = 2.5;
constexpr HighPrecType ALGO_SCRATCH_HANDLING_WIDTH_CV   = 0.20;
constexpr HighPrecType ALGO_SCRATCH_HANDLING_CORR_MM    = 0.25;

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
//  2026-10-20: THE DISPERSION IS NOW DRAWN. The 0.035 median stays the
//  HANDLING median. The reference collage (provisional, «statistical reference
//  only») gives the between-mark spread and the transport/handling ratio:
//  handling sigma_ln 0.50, transport sigma_ln 0.20, and transport tramlines
//  2.46 times the handling contrast (18.7 against 7.6 DN), so the transport
//  median is 0.035 x 2.46 = 0.0861. Log-normal: a contrast is a positive
//  multiplicative quantity, and the collage histogram is right-skewed.
//
//  Along one tramline the contrast varies too, as two processes: a fast one
//  (CV 0.09, correlated over 1.5 mm) and a slow trend (sigma_ln 0.30 over
//  about 20 mm; the collage shows a x1.9 change over 24 mm), both log-normal
//  multipliers keyed on the ABSOLUTE along-film position. On top of that, a
//  short-scale white texture of CV 0.14 (independent samples every
//  ALGO_DEFECT_TEXTURE_UM along the mark). Handling marks: CV 0.30 over
//  ALGO_SCRATCH_HANDLING_CORR_MM and the same texture.
// ---------------------------------------------------------------------------
constexpr HighPrecType ALGO_SCRATCH_CONTRAST_MEDIAN = 0.035;
constexpr HighPrecType ALGO_SCRATCH_CONTRAST_SIGMA_LN = 0.50;
constexpr HighPrecType ALGO_SCRATCH_TRANSPORT_CONTRAST_RATIO = 2.46;
constexpr HighPrecType ALGO_SCRATCH_TRANSPORT_CONTRAST_MEDIAN =
    ALGO_SCRATCH_CONTRAST_MEDIAN * ALGO_SCRATCH_TRANSPORT_CONTRAST_RATIO;
constexpr HighPrecType ALGO_SCRATCH_TRANSPORT_CONTRAST_SIGMA_LN = 0.20;
constexpr HighPrecType ALGO_SCRATCH_TRANSPORT_CONTRAST_CV       = 0.09;
constexpr HighPrecType ALGO_SCRATCH_TRANSPORT_CONTRAST_CORR_MM  = 1.5;
constexpr HighPrecType ALGO_SCRATCH_TRANSPORT_TREND_SIGMA_LN    = 0.30;
constexpr HighPrecType ALGO_SCRATCH_TRANSPORT_TREND_CORR_MM     = 20.0;
constexpr HighPrecType ALGO_SCRATCH_HANDLING_CONTRAST_CV        = 0.30;
constexpr HighPrecType ALGO_SCRATCH_TEXTURE_CV                  = 0.14;

// ---------------------------------------------------------------------------
//  Share of scratches that CUT rather than burnish - i.e. that subtract density.
//
//  ⚠ 2026-10-20: 0.5 -> 0.35 for handling marks and 0.25 for transport
//  tramlines, PROVISIONAL. The previous even split was a stated modelling choice
//  with no source. The reference collage (statistical reference only) shows the
//  polarity depending on the density under the mark - on thin areas 123 of 125
//  lines read light, on dense areas all read dark - which is the signature of a
//  CUT as a fractional removal of net density (it vanishes where there is no
//  image) against a burnish as an additive scatterer. The shares are the
//  collage's dark-line fractions after that density dependence is allowed for.
//  Not measured on film in hand; do not cite as measured.
// ---------------------------------------------------------------------------
constexpr HighPrecType ALGO_SCRATCH_HANDLING_CUT_SHARE  = 0.35;
constexpr HighPrecType ALGO_SCRATCH_TRANSPORT_CUT_SHARE = 0.25;

// ---------------------------------------------------------------------------
//  TRANSPORT SCRATCH EVENTS - 2026-10-20b (owner-approved temporal spec).
//
//  REPLACES ALGO_SCRATCH_TRANSPORT_COUNT (1.0 tramline on the web at level 1,
//  a definition) and ALGO_SCRATCH_RUN_WINDOW (32 jittered ordinals per frame).
//  Both counted in FILM FRAMES, so the event rate per second of screen time
//  scaled with the frame rate. The control is now FilmDamage::scratchTransport
//  = new events per minute of screen time, t = frameIndex / frameRate.
//
//  EVENT TIMING: a RENEWAL PROCESS WITH BOUNDED, INDEPENDENT GAPS. The gap
//  between consecutive new scratches is drawn independently for each event,
//  UNIFORM on [ALGO_SCRATCH_GAP_LO, ALGO_SCRATCH_GAP_HI] x m, m = 60 / rate s.
//  At the minimum non-zero rate, ScratchTransportFloor = 4.8 / min, m = 12.5 s
//  and every gap lies in exactly 5 - 20 s - the owner's specification
//  (2026-10-20): randomised per event, never shorter than 5 s, never longer
//  than 20 s, uniform because no evidence favours another shape. The same
//  4 : 1 band scales with the rate. Not a Poisson process (whose gaps are
//  unbounded): the owner required bounds.
//
//  STATELESS AND EXACT. Event k's time is the running sum of its gaps from a
//  per-seed random phase, so a frame finds the events alive at its time by
//  summing gaps from the origin - O(|t| x rate) hash draws, 7 200 for an hour
//  at 120 / min (x damageStrength up to 4) - with no history and no
//  approximation. Any frame renders alone, out of order, in any batch or on
//  any thread, and gives the same scratches; a fixed damageSeed reproduces the
//  sequence. frameIndex is CLIP-RELATIVE (AlgoFilmCoord), so t is normally
//  minutes; the walk is capped at 1e7 steps (347 h at 480 / min), beyond which
//  the schedule is no longer exact.
//  ⚠ A KEYFRAMED RATE is applied to the whole timeline at the value of the
//  current frame: event times are meanGap x sum of unit gaps, so animating
//  scratchTransport moves past events too. Animate the visibility instead.
//
//  LIFETIME: log-normal, sigma_ln ALGO_SCRATCH_LIFE_SIGMA_LN, MEAN equal to
//  scratchLifetime (or the stock's run length at ALGO_SCRATCH_NOMINAL_FPS),
//  truncated to [ALGO_SCRATCH_LIFE_LO, ALGO_SCRATCH_LIFE_HI] x mean by inverse-
//  CDF remapping. Log-normal: a contact duration is a positive product of
//  causes; the truncation keeps a scratch from flashing for a single frame
//  (which would read as flicker) or never ending. A CHOICE - nothing measures
//  scratch lifetimes in seconds.
//
//  At most ALGO_SCRATCH_LIVE_MAX tramlines are drawn at once (the newest);
//  reachable only at extreme rate x lifetime products.
// ---------------------------------------------------------------------------
constexpr HighPrecType ALGO_SCRATCH_GAP_LO           = 0.4;
constexpr HighPrecType ALGO_SCRATCH_GAP_HI           = 1.6;
constexpr HighPrecType ALGO_SCRATCH_LIFE_SIGMA_LN    = 0.5;
constexpr HighPrecType ALGO_SCRATCH_LIFE_LO          = 0.25;
constexpr HighPrecType ALGO_SCRATCH_LIFE_HI          = 3.0;
constexpr HighPrecType ALGO_SCRATCH_NOMINAL_FPS      = 24.0;
//  Median correction so that the TRUNCATED log-normal keeps the requested mean:
//  E[Y | 0.25 < Y < 3] = 0.98583 for Y log-normal with mean 1, sigma_ln 0.5.
constexpr HighPrecType ALGO_SCRATCH_LIFE_MEAN_CORR   = 1.0143724;
constexpr int32_t      ALGO_SCRATCH_LIVE_MAX         = 256;
//  defectVariability at which every along-run temporal amplitude below equals
//  its collage value; the amplitude scales as variability / this.
constexpr HighPrecType ALGO_DEFECT_VARIABILITY_REF   = 0.5;

// ---------------------------------------------------------------------------
//  Clear-base density, the floor a REMOVAL (cut, pinhole) works towards with
//  defectOptics on (2026-10-20b). 0.05: ГОСТ 25120 §2.1a (Amendment 1) - a
//  colourless triacetate base has optical density at most 0.05. An upper bound
//  used as the value, PROVISIONAL; a grey-dyed antihalation base is denser.
//  Never above the record's own D_min (AlgoDefectOpticsMake).
// ---------------------------------------------------------------------------
constexpr HighPrecType ALGO_CLEAR_BASE_DENSITY       = 0.05;

// ---------------------------------------------------------------------------
//  Minimum rasterised footprint, PIXELS (radius / half-width), 2026-10-20b.
//  A mark thinner than a pixel was sampled at pixel centres only, so where its
//  centre fell between two centres it vanished - and a wandering tramline or
//  a drifting speck then blinked in and out from frame to frame at coarse
//  rasters. Below this size the footprint is widened to it and the opacity
//  scaled down in proportion (area for a particle, width for a stroke), which
//  keeps the mark's integrated density and makes it resolution-consistent.
//  Inert at normal rasters (a 45 um tramline is 1.7 px wide at 2K super35).
// ---------------------------------------------------------------------------
constexpr HighPrecType ALGO_DEFECT_MIN_FOOTPRINT_PX  = 0.5;

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
//  Handling scratch length, millimetres.
//
//  2026-10-20: was UNIFORM on 0.3 - 4 mm (AlgoControl.hpp's range, no shape).
//  Now LOG-NORMAL, median 0.5 mm, sigma_ln 0.6, truncated to [0.1, 4.0] by
//  inverse-CDF remapping (no pile-up at the bounds). PROVISIONAL: the reference
//  collage gives a median near 0.5 mm with sigma_ln 0.45 - 0.8 (statistical
//  reference only). Log-normal because the length of a slide contact is a
//  product of independent factors (pressure, distance, debris size); the
//  collage histogram is right-skewed with no hard lower edge.
//  4 mm stays the upper bound, so the point stores and margins are unchanged.
// ---------------------------------------------------------------------------
constexpr HighPrecType ALGO_SCRATCH_HANDLING_LEN_MEDIAN_MM = 0.5;
constexpr HighPrecType ALGO_SCRATCH_HANDLING_LEN_SIGMA_LN  = 0.6;
constexpr HighPrecType ALGO_SCRATCH_HANDLING_LEN_MIN_MM    = 0.1;
constexpr HighPrecType ALGO_SCRATCH_HANDLING_LEN_MAX_MM    = 4.0;

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
//  The same floor on a REVERSAL colour stock (2026-10-22, owner-approved).
//
//  ALGO_DEFECT_CHROMA_MIN above was solved by rendering ORWOCOLOR_NC21, a
//  NEGATIVE, while the ratio it targets was measured on the owner's ORWO colour
//  frames -- ORWOCHROME UT-18 slides (ORWO_CHROM_UT18). On a slide a film-borne
//  particle prints DARK (it adds density to the positive), and the owner's
//  requirements document measures that polarity separately: median min/max
//  channel deviation ratio 0.63 for dark defects (31 frames), 0.55 for light.
//  Solved like the negative constant: rendered on ORWO_CHROM_UT18 (dust,
//  debris, fibres; grain off) and swept until the detected output median ratio
//  of dark particles lands on 0.63: 0.45 -> 0.551, 0.55 -> 0.619, 0.57 -> 0.628,
//  0.62 -> 0.653 (mid-grey field); 0.57 gives 0.60-0.67 from a light to a dark
//  field -- see RESULT_2026-10-22. PROVISIONAL: the
//  three 1:1 colour tiles of the reference collage (R8C7-C9) give a lower
//  median, 0.38 over 28 specks, recorded as an unresolved disagreement.
// ---------------------------------------------------------------------------
constexpr HighPrecType ALGO_DEFECT_CHROMA_MIN_REVERSAL = 0.57;

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
// ⚠ 2026-10-20c (owner-approved): debris 0.30 -> 0.12, and "a polygon ... looks
// no better" above is SUPERSEDED: at 8 um/px the owner's collage particles are
// angular, nearly convex flakes (solidity 0.94) with soft edges, not lobed beans.
// The outline is now polygon x roughness x these lobes (ALGO_OUTLINE_*).
constexpr HighPrecType ALGO_DEBRIS_LOBE_DEPTH     = 0.12;

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
//  ALGO_SCRATCH_CLIP_MARGIN_MM (2026-10-20; was one along-extent, which the
//  sampled centreline cannot afford on a large gauge).
//  A burst puts members up to 3 spacings (and a log-normal tail) beside its
//  seed, so the handling margin is the length clamp plus 2 mm.
// ---------------------------------------------------------------------------
constexpr HighPrecType ALGO_DUST_MARGIN_MM     = 0.2;
constexpr HighPrecType ALGO_DEBRIS_MARGIN_MM   = 1.6;
constexpr HighPrecType ALGO_FIBRE_MARGIN_MM    = 25.0;
constexpr HighPrecType ALGO_SCRATCH_MARGIN_MM  = ALGO_SCRATCH_HANDLING_LEN_MAX_MM + 2.0;

// ===========================================================================
//  2026-10-20: STOCHASTIC VARIATION AND READER OPTICS.
//
//  Owner-approved (2026-10-19, item 3 of the defect task). Every figure in this
//  block is PROVISIONAL: either read from the owner's reference collage
//  (DEFECTS/Film_Defect_Reference_Collage_9x9.png, used «as statistical
//  reference only», 8.15 um/px) or, where the collage cannot give it, a stated
//  choice. The basis of each is in doc/RESULT_2026-10-20_defect_optics.md.
//
//  Distribution choices, by physical property:
//    positions        log-Gaussian Cox process (unchanged); transport tramline
//                     across-web position uniform; companion offset and burst
//                     spacing log-normal (positive separations)
//    sizes, widths,
//    lengths, contrast log-normal (positive scale quantities)
//    orientation      mixture: transport-biased scatter + uniform (unchanged);
//                     burst members share the seed heading plus N(0, 0.017 rad)
//    shape            harmonic lobe amplitudes scaled by U(0.5, 1.5) each
//    opacity          Beta(2,3) (dust, unchanged), uniform (debris, fibres)
//    polarity         Bernoulli with the shares below
//    texture, wander,
//    along-mark change Gaussian: white (independent lattice samples) or smooth
//                     (lattice of the correlation length), in film millimetres
//
//  TIME. Every draw is keyed on film coordinates or on a transport ordinal,
//  never on the frame index, so a defect looks the same on every render of a
//  frame and a tramline's wander and contrast are continuous from one frame to
//  the next (keyed on the ABSOLUTE along-film position). No pattern repeats:
//  the lattice hash has no usable period.
// ===========================================================================

// ---- reader optics (FilmDamage::defectOptics, default OFF) -----------------
//  s_ref: the reader specular fraction at which the law reproduces the OFF
//  densities. The reference scans were made on a GCMC «UF15» film scanner whose
//  light source is undocumented; its flat, vignette-free field and the absence
//  of Newton's rings point weakly to a diffuse LED panel. 0.25, bracket 0.1 -
//  0.5. NOT 0.853, which is Streiffert's 1947 densitometer, not a scanner.
constexpr HighPrecType ALGO_DEFECT_OPTICS_S_REF          = 0.25;
//  Cap of the scattering part as read at s_ref, as a fraction of its ceiling
//  -log10(1 - s_ref). Keeps the inversion finite; the excess becomes absorption.
constexpr HighPrecType ALGO_DEFECT_SCATTER_REF_HEADROOM  = 0.95;
//  Scattering share of each class's density increment at s_ref (choices,
//  ordered by the physics: a polished base absorbs nothing, a lint ball is
//  mostly opaque fibre mass, translucent dust and fibres lie between).
constexpr HighPrecType ALGO_DUST_SCATTER_SHARE           = 0.3;
constexpr HighPrecType ALGO_DEBRIS_SCATTER_SHARE         = 0.2;
constexpr HighPrecType ALGO_FIBRE_SCATTER_SHARE          = 0.4;
constexpr HighPrecType ALGO_BURNISH_SCATTER_SHARE        = 0.9;
//  A cut's wall: a scattering edge whose increment at s_ref is this fraction of
//  the cut's OFF increment (the collage bounds it at 2 %), and its scattering
//  share. Under a condenser it can turn a fine cut dark.
constexpr HighPrecType ALGO_CUT_WALL_SHARE               = 0.02;
constexpr HighPrecType ALGO_CUT_WALL_SCATTER_SHARE       = 0.8;

// ---- dust and debris -------------------------------------------------------
//  Share of dust-class specks that are REMOVALS (pinholes, emulsion pick-off)
//  rather than material on the film: the collage shows 83.6 % of specks light,
//  so 16.4 % of the opposite polarity.
constexpr HighPrecType ALGO_DUST_REMOVAL_SHARE           = 0.164;
//  Interior texture: Gaussian white noise multiplying the opacity, sigma as a
//  fraction of the peak (collage: dust 0.15, debris 0.27), on a lattice of
//  ALGO_DEFECT_TEXTURE_UM in the particle's own frame.
constexpr HighPrecType ALGO_DUST_TEXTURE_SIGMA           = 0.15;
constexpr HighPrecType ALGO_DEBRIS_TEXTURE_SIGMA         = 0.27;
//  Lattice step of every white texture, micrometres of film: the collage's
//  pixel, the finest scale at which its statistics were measured.
constexpr HighPrecType ALGO_DEFECT_TEXTURE_UM            = 8.15;
//  Shape variation: each harmonic's lobe amplitude is scaled by a factor drawn
//  uniformly on [1 - x, 1 + x] (mean 1, so the class's mean lobe depth is kept).
constexpr HighPrecType ALGO_DEFECT_LOBE_VARIATION        = 0.5;

// ---- fibres ----------------------------------------------------------------
//  Along-fibre variation (collage: width CV 0.17 - 0.30, contrast CV 0.38 -
//  0.72): log-normal multipliers, smooth over ALGO_FIBRE_MOD_CORR_MM.
constexpr HighPrecType ALGO_FIBRE_WIDTH_CV               = 0.20;
constexpr HighPrecType ALGO_FIBRE_CONTRAST_CV            = 0.50;
constexpr HighPrecType ALGO_FIBRE_MOD_CORR_MM            = 0.5;

// ---- transport tramlines ---------------------------------------------------
//  Lateral wander of the abrader track: Gaussian, rms 16 um, correlated over
//  1.7 mm of film (collage).
constexpr HighPrecType ALGO_SCRATCH_WANDER_RMS_MM        = 0.016;
constexpr HighPrecType ALGO_SCRATCH_WANDER_CORR_MM       = 1.7;
//  Sample spacing of a tramline's centreline, mm of film, on ABSOLUTE multiples
//  of the along-film coordinate so the samples are a property of the film. The
//  point store bounds the frame's along extent plus the clip margin; on a gauge
//  that would exceed it the step is widened by a whole factor.
constexpr HighPrecType ALGO_SCRATCH_SAMPLE_MM            = 0.2;
constexpr HighPrecType ALGO_SCRATCH_CLIP_MARGIN_MM       = 1.0;
constexpr int32_t      ALGO_SCRATCH_TRANSPORT_MAX_POINTS = 640;
//  Companion tracks (collage): half the tramlines carry a fainter parallel
//  track at a log-normal offset (median 0.25 mm, sigma_ln 0.3, either side) of
//  0.20 of the main contrast; half of those cover only part of the run.
constexpr HighPrecType ALGO_SCRATCH_COMPANION_PROB       = 0.5;
constexpr HighPrecType ALGO_SCRATCH_COMPANION_OFFSET_MM  = 0.25;
constexpr HighPrecType ALGO_SCRATCH_COMPANION_SIGMA_LN   = 0.3;
constexpr HighPrecType ALGO_SCRATCH_COMPANION_RATIO      = 0.20;
constexpr HighPrecType ALGO_SCRATCH_COMPANION_PARTIAL    = 0.5;

// ---- handling bursts -------------------------------------------------------
//  A single wipe leaves several near-parallel marks. Collage: 24 % of marks lie
//  in groups of 2 - 4 (mean 2.5), spacing median 0.30 mm (IQR 0.2 - 0.54, so
//  sigma_ln 0.74), angle spread about 1 degree. A seed becomes a burst with
//  probability 0.112, which puts 0.112 x 2.5 / (1 + 1.5 x 0.112) = 24 % of
//  marks in bursts; the seed rate is divided by (1 + 1.5 x 0.112) so the
//  control's areal density of MARKS is unchanged.
constexpr HighPrecType ALGO_SCRATCH_BURST_SEED_SHARE     = 0.112;
constexpr HighPrecType ALGO_SCRATCH_BURST_P2             = 0.6;
constexpr HighPrecType ALGO_SCRATCH_BURST_P3             = 0.3;
constexpr HighPrecType ALGO_SCRATCH_BURST_MEAN_EXTRA     = 1.5;   // E[size] - 1
constexpr HighPrecType ALGO_SCRATCH_BURST_SPACING_MM     = 0.30;
constexpr HighPrecType ALGO_SCRATCH_BURST_SPACING_SIGMA_LN = 0.74;
constexpr HighPrecType ALGO_SCRATCH_BURST_ANGLE_SIGMA_RAD  = 0.017;
constexpr HighPrecType ALGO_SCRATCH_BURST_LEN_SIGMA_LN   = 0.2;
constexpr int32_t      ALGO_SCRATCH_BURST_MAX            = 4;

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
// 2026-10-20: lattices of the along-mark processes and the white textures.
constexpr uint32_t ALGO_DEFECT_TAG_TEXTURE       = 0x00DE7E01u;
constexpr uint32_t ALGO_DEFECT_TAG_MODULATION    = 0x00DE7E02u;
constexpr uint32_t ALGO_DEFECT_TAG_WANDER        = 0x00DE7E03u;
constexpr uint32_t ALGO_DEFECT_TAG_BURST         = 0x00DE7E04u;
constexpr uint32_t ALGO_DEFECT_TAG_SCRATCH_GAP   = 0x00DE7E05u;
constexpr uint32_t ALGO_DEFECT_TAG_OUTLINE       = 0x00DE7E06u;   // 2026-10-20c

// ===========================================================================
//  2026-10-20c (owner-approved): RAGGED OUTLINE, SOFT EDGE, FLAT OPAQUE CORE
//  for dust and debris.
//
//  WHY. The outline used to be a radius modulated by three low harmonics
//  (orders 2-4), which draws smooth "beans"; the owner's collage particles
//  (rows 4-5, e.g. R5C1, an angular flake 0.77 mm across) have facets, corners
//  and fine boundary roughness. Their edges are also SOFTER than the old
//  8 um transition (R5C1: 10-90 % width 68 um), and an opaque core is FLAT
//  (R5C1 core median 249/255), where the interior white texture applied to
//  opacity speckled it.
//
//  OUTLINE  = polygon base x (1 + fine roughness), normalised to mean radius 1,
//             multiplying the old harmonic shape:
//    polygon   K vertices, K uniform on [VERTEX_MIN, VERTEX_MAX]; vertex k at
//              angle 2 pi (k + J (U - 1/2)) / K, radius exp(SIGMA_LN N);
//              straight edges between vertices (corners).
//    roughness sum over m = MMIN .. min(MMAX, floor(pi R_px)) of
//              a_m cos(m theta + phi_m), a_m = ROUGH_C m^-ROUGH_BETA x Rayleigh
//              (median 1), phi_m uniform: a Gaussian random outline with a
//              power-law spectrum. Harmonics finer than ~2 px of perimeter
//              are not drawn (resolution-aware, no aliasing).
//    Calibrated through the full chain at the collage's 8.15 um/px against
//    the collage's compact particles in flat-background tiles (radial
//    harmonic spectrum of 78 particles: median relative amplitude 0.0178 at
//    m 8-15 and 0.0054 at m 16-31, slope -1.67; solidity of 50 particles
//    R >= 8 px: median 0.94). Model after calibration (debris): 0.0164 /
//    0.0060, solidity 0.92. The raw collage fit (C 1.12) made spiky stars
//    (solidity 0.85): its contours carry grain and pixel noise, so C was
//    solved down to 0.60 on solidity and spectrum together, the vertex jitter
//    to 0.06, and the debris lobes 0.30 -> 0.12 (the collage particles are
//    nearly convex). RESULT_2026-10-20c. PROVISIONAL: one collage, one
//    scanner; the polygon base (K, J) is a definition.
//  EDGE     = per particle max(edgePx, <class>_EDGE_FRAC x R_px x exp(EDGE_SIGMA_LN N)),
//             the density ramping with coverage (a thinning rim, a = 1 - (1 - alpha)^cov):
//             the collage's 10-90 % edge width grows with particle size
//             (~0.24 R at 8-16 px and at R5C1); a surface particle sits off the
//             emulsion plane and its rim thins out, so the edge is graded. The
//             fraction is solved per class through the full chain so the printed
//             10-90 % width is ~0.24 R as in the collage: an opaque particle
//             needs a wider density ramp than a translucent one because the print
//             compresses the dense end.
//  TEXTURE  = the interior white texture fades as the local opacity rises
//             above TEXTURE_FADE_ALPHA, reaching zero at a = 1: translucent
//             dust keeps its measured texture, an opaque core is flat.
// ===========================================================================
constexpr int32_t      ALGO_OUTLINE_SAMPLES          = 256;
constexpr int32_t      ALGO_OUTLINE_MMIN             = 5;
constexpr int32_t      ALGO_OUTLINE_MMAX             = 48;
constexpr HighPrecType ALGO_OUTLINE_ROUGH_C          = 0.60;
constexpr HighPrecType ALGO_OUTLINE_ROUGH_BETA       = 1.67;
constexpr int32_t      ALGO_OUTLINE_VERTEX_MIN       = 5;
constexpr int32_t      ALGO_OUTLINE_VERTEX_MAX       = 12;
constexpr HighPrecType ALGO_OUTLINE_VERTEX_JITTER    = 0.6;
constexpr HighPrecType ALGO_OUTLINE_VERTEX_SIGMA_LN  = 0.06;
constexpr HighPrecType ALGO_OUTLINE_MIN_RADIUS_PX    = 1.5;
constexpr HighPrecType ALGO_OUTLINE_FLOOR            = 0.05;
constexpr HighPrecType ALGO_DUST_EDGE_FRAC           = 0.22;   // calibrated, translucent class
constexpr HighPrecType ALGO_DEBRIS_EDGE_FRAC         = 0.65;   // calibrated, opaque class
constexpr HighPrecType ALGO_DEFECT_EDGE_FRAC_SIGMA_LN = 0.35;
constexpr HighPrecType ALGO_DEFECT_TEXTURE_FADE_ALPHA = 0.5;


// ===========================================================================
//  from AlgoProcessingDefects.hpp
// ===========================================================================

// ===========================================================================
//  Definitions (NOT measurements - see the header block)
// ===========================================================================

// ---- processingQuality: development mottle --------------------------------
// Peak modulation of the net density at processingQuality x strength = 1: the
// lattice values of the field are uniform in [-1, 1], so the net density is
// multiplied by a factor inside [1 - 0.04, 1 + 0.04].
constexpr HighPrecType ALGO_MOTTLE_AMPLITUDE   = 0.04;
// Two lattice octaves, film millimetres per cell, and their weights (sum 1).
constexpr HighPrecType ALGO_MOTTLE_CELL_MM_LO  = 3.0;
constexpr HighPrecType ALGO_MOTTLE_CELL_MM_HI  = 0.75;
constexpr HighPrecType ALGO_MOTTLE_WEIGHT_LO   = 0.7;
constexpr HighPrecType ALGO_MOTTLE_WEIGHT_HI   = 0.3;

// ---- processingQuality: air bells -----------------------------------------
// Mean count per square centimetre of film at level 1, the radius range, and
// the range of the undeveloped fraction at the bell's centre.
constexpr HighPrecType ALGO_AIRBELL_PER_CM2    = 0.5;
constexpr HighPrecType ALGO_AIRBELL_R_MIN_MM   = 0.08;
constexpr HighPrecType ALGO_AIRBELL_R_MAX_MM   = 0.40;
constexpr HighPrecType ALGO_AIRBELL_DEPTH_MIN  = 0.2;
constexpr HighPrecType ALGO_AIRBELL_DEPTH_MAX  = 0.6;
// Soft edge as a fraction of the radius.
constexpr HighPrecType ALGO_AIRBELL_EDGE_FRAC  = 0.25;

// ---- dryingMarks: water spots ----------------------------------------------
// Mean count per square centimetre at level 1. Sized in FILM millimetres, so
// the gauge dependence AlgoControl.hpp describes - one 5 mm mark is a far
// larger share of a 16 mm frame than of a 35 mm still - is derived, not
// authored. 2.5 mm is the largest radius, i.e. the 5 mm mark of that example.
constexpr HighPrecType ALGO_DRYING_PER_CM2     = 0.15;
constexpr HighPrecType ALGO_DRYING_R_MIN_MM    = 0.5;
constexpr HighPrecType ALGO_DRYING_R_MAX_MM    = 2.5;
// Tide-line half-width, and the additive density of the line at its crest.
constexpr HighPrecType ALGO_DRYING_RIM_MM      = 0.06;
constexpr HighPrecType ALGO_DRYING_RIM_D_MIN   = 0.02;
constexpr HighPrecType ALGO_DRYING_RIM_D_MAX   = 0.08;
// Residue inside the ring, as a fraction of the crest density.
constexpr HighPrecType ALGO_DRYING_INSIDE_FRAC = 0.2;

// ---- shared ----------------------------------------------------------------
// Placement cell for the two discrete classes: 10 mm x 10 mm = 1 cm^2, so a
// per-cm^2 rate is the Poisson mean of one cell directly.
constexpr HighPrecType ALGO_PROCDEF_CELL_MM    = 10.0;
// Upper bound on the Poisson draw per cell; far above any mean used here.
constexpr int32_t      ALGO_PROCDEF_MAX_PER_CELL = 64;
// Smallest soft edge in pixels, so a sub-pixel mark is antialiased.
constexpr HighPrecType ALGO_PROCDEF_EDGE_MIN_PX  = 0.5;


// ===========================================================================
//  from AlgoGateDefects.hpp
// ===========================================================================

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
// ⚠ SUPERSEDED 2026-10-20b by the population-sized window below; kept only so
// older notes that cite it resolve. Not read by the engine.
constexpr int32_t ALGO_GATE_WINDOW = 96;
// 2026-10-20b: the window is now sized from the population (about six mean
// residences of arrivals plus a margin), bounded by this.
constexpr int32_t ALGO_GATE_WINDOW_MAX = 1024;

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
// ⚠ 2026-10-20b: THE TWO PER-FRAME RATES ABOVE ARE REPLACED, because a rate per
// FRAME makes the gate renew 2.5 times faster in seconds at 60 fps than at 24.
// The population is now set by its STEADY-STATE COUNT (the old pair's
// 1e-2 x 1000 = 10 particles at level 1, era reference) and its MEAN RESIDENCE
// in SECONDS (FilmDamage::dustLifetime, default 42 s = the old 1000 frames at
// 24 fps). Accretion per second = count / residence (Little's law), so the
// lifetime control changes how often the pattern renews and NOT how many
// particles are present.
constexpr HighPrecType ALGO_GATE_STEADY_COUNT = 10.0;

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
// ⚠ 2026-10-20b (owner's temporal spec): THE JITTER IS NOW SMOOTH IN TIME AND
// THE BLINK IS GONE. The jitter above was redrawn independently on every frame
// (uniform, keyed on the frame index) and one frame in ten removed the particle
// outright - exactly the uncorrelated flicker and random disappear / reappear
// the spec forbids. Each quantity is now a smooth unit Gaussian process of
// screen TIME (AlgoDefectSmoothNoise, correlation ALGO_GATE_DRIFT_CORR_S), so a
// particle drifts, rocks and dims gradually and keeps its identity. Amplitudes
// at defectVariability = 0.5: position rms = 60 / sqrt(3) um and angle rms =
// 0.35 / sqrt(3) rad (the old uniform ranges' rms), opacity factor between
// 0.55 and 1.0 via the normal CDF of its process. Scaled by variability / 0.5.
constexpr HighPrecType ALGO_GATE_JITTER_UM        = 60.0;
constexpr HighPrecType ALGO_GATE_JITTER_ANGLE_RAD = 0.35;
constexpr HighPrecType ALGO_GATE_JITTER_ALPHA_MIN = 0.55;
// Correlation time of the drift, seconds of screen time. 0.5 s: slower than
// the eye's flicker fusion, fast enough that a lodged particle visibly lives
// during a shot. A CHOICE - nothing measures dirt motion in a gate.
constexpr HighPrecType ALGO_GATE_DRIFT_CORR_S     = 0.5;

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


// ===========================================================================
//  from AlgoGateWeave.hpp
// ===========================================================================

// ---------------------------------------------------------------------------
//  Octaves in the weave's temporal noise.
//
//  5. The lowest octave has the period set by the profile's corner frequency and
//  each one above halves it, so five octaves span a factor of sixteen in
//  frequency - from the corner down to the frame rate, which is where the
//  representable band ends. A sixth would sit above Nyquist and only alias.
// ---------------------------------------------------------------------------
constexpr int32_t ALGO_WEAVE_OCTAVES = 5;

// ---------------------------------------------------------------------------
//  Variance lost to interpolating the temporal lattice.
//
//  26/35, exactly.
//
//  The noise is built by interpolating independent values on a lattice of frame
//  indices, and a smoothly interpolated point is a weighted MEAN of its two
//  neighbours - so its variance is the sum of the squared weights, which is one
//  only exactly at a node. With W = 3u^2 - 2u^3:
//
//      E[W]   = 1/2
//      E[W^2] = 9/5 - 12/6 + 4/7 = 13/35
//      E[(1-W)^2 + W^2] = 1 - 2E[W] + 2E[W^2] = 26/35
//
//  Dividing it out is what makes the requested RMS amplitude the amplitude that
//  actually appears. The same mistake in the two-dimensional dirt field produced a
//  weave that measured three quarters of what was asked for, which is small enough
//  to look plausible and wrong enough to fail a measurement.
//
//  ANYONE CHANGING THE INTERPOLATION KERNEL MUST RECOMPUTE THIS.
// ---------------------------------------------------------------------------
constexpr HighPrecType ALGO_WEAVE_INTERP_VARIANCE = 26.0 / 35.0;

// ---------------------------------------------------------------------------
//  Micrometres per millimetre. Weave amplitudes are quoted in micrometres on the
//  negative; the geometry is in millimetres.
// ---------------------------------------------------------------------------
constexpr HighPrecType ALGO_WEAVE_UM_PER_MM = 1000.0;

// ---------------------------------------------------------------------------
//  Smallest displacement worth resampling the frame for, pixels.
//
//  1/512 of a pixel. Below this the bilinear weights round to a pass-through and
//  the whole stage is an expensive copy, so it takes the copy path instead. This
//  is not a quality compromise: a 1/512 pixel shift is far below the precision of
//  the storage type at any bit depth a host will ever hand over.
// ---------------------------------------------------------------------------
constexpr HighPrecType ALGO_WEAVE_MIN_SHIFT_PX = 1.0 / 512.0;

// ---------------------------------------------------------------------------
//  Generator stream tags for the two axes.
//
//  Separate, so that horizontal and vertical weave are independent. Sharing a
//  stream would make the image travel along a diagonal, which is a distinctive and
//  quite wrong-looking motion.
// ---------------------------------------------------------------------------
constexpr uint32_t ALGO_WEAVE_TAG_X = 0x0057EA01u;
constexpr uint32_t ALGO_WEAVE_TAG_Y = 0x0057EA02u;


// ===========================================================================
//  from AlgoScannerArtifacts.hpp
// ===========================================================================

// ---- definitions -----------------------------------------------------------
// Transmittance loss in the corners at scannerArtifacts x strength = 1.
constexpr HighPrecType ALGO_SCAN_SHADING_CORNER = 0.06;
// Peak per-column gain error at level 1 (gains uniform in [1 - a, 1 + a]).
constexpr HighPrecType ALGO_SCAN_BANDING_AMP    = 0.004;
// Newton's rings: modulation depth at level 1, first-ring radius at the green
// wavelength, and the Gaussian radius of the contact patch, film millimetres.
constexpr HighPrecType ALGO_SCAN_RINGS_AMP      = 0.03;
constexpr HighPrecType ALGO_SCAN_RINGS_R1_MM    = 1.0;
constexpr HighPrecType ALGO_SCAN_RINGS_PATCH_MM = 6.0;
// Nominal channel wavelengths, nm. Only their ratios are used.
constexpr HighPrecType ALGO_SCAN_LAMBDA_R       = 610.0;
constexpr HighPrecType ALGO_SCAN_LAMBDA_G       = 540.0;
constexpr HighPrecType ALGO_SCAN_LAMBDA_B       = 465.0;


// ===========================================================================
//  from AlgoStorageAge.hpp
// ===========================================================================

// ---------------------------------------------------------------------------
//  storageSeverity (2026-10-10)
//
//  damage.storageSeverity had no reader. AlgoControl.hpp item 9 defines the
//  axis: 0 a cold vault, 1 a warm attic for forty years. This turns that
//  definition into elapsed years for the resolver above:
//
//      40 x severity x damageStrength years at 30 degC,
//
//  restated to the caller's storageCelsius through AlgoStorageTimeFactor, so
//  that adding the result to storageYears and passing storageCelsius ages the
//  film by both stretches exactly - the resolver restates the sum to the
//  record's reference temperature and the two terms come out as
//  storageYears at storageCelsius plus 40 s k years at 30 degC.
//
//  ⚠ 40 YEARS AND 30 degC ARE THE CONTROL'S OWN DEFINITION ("a warm attic for
//  forty years"), not a measurement; 30 degC is the warmest node of Wilhelm's
//  Table 5.3 and is taken because the table is never extrapolated.
//
//  ⚠ WHAT IT DOES NOT DO. Item 9 also lists colour crossover, age fog, fungus
//  and base deterioration. None has a rate in the corpus and AgingSpec is all
//  zero, so only dye fade - through DyeStabilitySpec, on the stocks that
//  publish a rate - responds. On every other stock the control is consumed and
//  has no visible effect, which is the same honest state storageYears is in.
//
//  Returns exactly 0 when film damage is off, the strength is zero or the
//  severity is zero, so every earlier render is reproduced bit for bit.
// ---------------------------------------------------------------------------
constexpr double ALGO_SEVERITY_FULL_YEARS = 40.0;
constexpr double ALGO_SEVERITY_CELSIUS    = 30.0;


// ===========================================================================
//  from Algo_09_Sim.cpp and AVX2/Algo_09_Sim.cpp (were file-local, identical)
// ===========================================================================

// ----------------------------------------------------------------------
//  Two pi, for the angular harmonics and the orientation draws.
// ----------------------------------------------------------------------
constexpr HighPrecType ALGO_DEFECT_TWO_PI = 6.283185307179586476925286766559;

// ----------------------------------------------------------------------
//  Micrometres per millimetre. Sizes are measured in micrometres and the
//  film coordinate system is in millimetres, so the conversion appears often
//  enough to be worth naming.
// ----------------------------------------------------------------------
constexpr HighPrecType ALGO_DEFECT_UM_PER_MM = 1000.0;

// ===========================================================================
//  from Algo_16_Sim.cpp and AVX2/Algo_16_Sim.cpp (were file-local, identical)
// ===========================================================================

// ----------------------------------------------------------------------
//  Micrometres per millimetre.
// ----------------------------------------------------------------------
constexpr HighPrecType ALGO_GATE_UM_PER_MM = 1000.0;

// ----------------------------------------------------------------------
//  Smallest opacity worth rasterising.
//
//  One part in a thousand of transmittance is far below one code value at any
//  bit depth a host will hand over, so a particle fainter than this cannot
//  change the output and only costs a bounding box.
// ----------------------------------------------------------------------
constexpr HighPrecType ALGO_GATE_ALPHA_EPSILON = 1.0e-3;
