#pragma once

// ---------------------------------------------------------------------------
//  AlgoCoatingField.hpp
//
//  Stage 4 and its sub-stage 4b of the film simulation pipeline: lens vignette
//  and web-coherent coating field.
//
//  TWO MECHANISMS, ONE MULTIPLY
//
//  Both are pure per-pixel multipliers on exposure, so they are built into a
//  single field and applied in one pass. The marginal cost over the previous
//  stage is one extra stream read, not two passes.
//
//  They remain conceptually separate because the physics and, more importantly,
//  the GEOMETRY differ - and the geometry is the part that is easy to get wrong.
//
//  THE VIGNETTE IS THE LENS
//
//  Off-axis illuminance on a flat focal plane falls as cos^4(theta): one cosine
//  from the tilted exit pupil, one from the tilted image plane, and two from the
//  inverse-square increase in distance. It is locked to the frame, constant for
//  the whole clip, and present in every era - even modern coated glass loses
//  several tenths of a stop in the corners. Real lenses lose more than geometry
//  alone predicts, because hoods, filter stacks and undersized rear elements add
//  mechanical vignetting on top; that surplus is what the per-era figure in the
//  profile carries.
//
//  THE COATING FIELD IS THE FILM, AND IT LIVES IN WEB COORDINATES
//
//  Film is coated as a wide web and slit into strips afterwards. The coating
//  pattern therefore knows nothing whatever about frame boundaries:
//
//    - ACROSS the web, which is the frame's horizontal axis on 35 mm, the
//      structure is FIXED for the entire roll. A left-to-right variation that
//      does not change from frame to frame.
//    - ALONG the web, vertically, the film advances by exactly one frame pitch
//      per frame, so each frame samples a different stretch of web. That, and
//      only that, is the genuine emulsion-driven frame-to-frame variation.
//
//  Two wrong alternatives, both of which look like bugs:
//
//    - Treating the whole thing as two-dimensional noise keyed on the frame
//      index makes the cross-web structure decorrelate every frame, which reads
//      as large-scale flicker.
//    - Keying it on the seed alone freezes it completely, which reads as dirt on
//      the scanner rather than a property of the film.
//
//  SYNTHESISED AS A SUM OF SINUSOIDS
//
//  The field is built as a sum of cosines in ABSOLUTE web coordinates rather than
//  as filtered noise. Three practical reasons: it is an exact function of web
//  position and seed, so any frame renders independently, out of order, with no
//  state and no seams; it slides continuously as the web advances instead of
//  being redrawn; and it costs one small low-resolution evaluation plus an
//  interpolation instead of a full-resolution spectral synthesis every frame.
// ---------------------------------------------------------------------------

// Project-wide primitives, included unconditionally as required by the project
// coding standard.
#include "Common.hpp"
#include "CompileTimeUtils.hpp"

// ImgType, AlgoType, HighPrecType. The single place numeric types are chosen.
#include "AlgoTypes.hpp"

// Arena views and the AlgoType arithmetic type.
#include "AlgoMemHandler.hpp"

// Bilinear interpolation of the low-resolution field.
#include "AlgoSeparableBlur.hpp"

// Counter-based randomness for the field coefficients.
#include "AlgoCounterRng.hpp"

// The control structure carrying the vignette and coating overrides.
#include "AlgoControl.hpp"

// FilmProfile, for default_vignette and the coating specification.
#include "film_profiles.hpp"

#include <cstdint>   // int32_t


// ---------------------------------------------------------------------------
//  NO GEOMETRY STRUCT
//
//  An earlier revision passed an AlgoFrameGeometry aggregate here. It has been
//  removed, because it was not carrying independent information:
//
//    pxPerMm       is sizeX / negWidthMm -- derived, not input
//    negWidthMm    is FilmFormat.width_mm  -- resolved from the control structure
//    negHeightMm   is FilmFormat.height_mm -- likewise
//    framePitchMm  is FilmFormat::FramePitchMm() -- now emitted by the generator
//    frameIndex    duplicated AlgoControls::frameIndex
//    seed          duplicated AlgoControls::seed
//
//  Everything is therefore either derivable inside the engine or already a control
//  field, so the physical quantities arrive here as plain scalars in the same style
//  as sizeX, sizeY and pitch.
// ---------------------------------------------------------------------------


// ---------------------------------------------------------------------------
//  VIGNETTE EXPONENT
//
//  4. The cos^4 law, from the four independent cosine factors described above.
//  It is not a fitted curve and must not be treated as a tunable: changing it
//  would decouple the model from the geometry it represents. The strength of the
//  effect is set by the corner loss in stops, not by this exponent.
// ---------------------------------------------------------------------------
constexpr int32_t ALGO_VIGNETTE_EXPONENT = 4;


// ---------------------------------------------------------------------------
//  NUMBER OF SINUSOIDAL COMPONENTS
//
//  64 per field component. This is the number of random cosines summed to build
//  one field.
//
//  The sum of N cosines of random phase approaches a Gaussian process as N grows,
//  by the central limit theorem, and the approach is quick: at 64 components the
//  amplitude distribution is indistinguishable from Gaussian for the purpose of a
//  low-amplitude multiplicative field. Fewer components leave visible periodic
//  structure - the individual cosines become discernible as a regular ripple.
//  More components cost proportionally more for no visible change.
// ---------------------------------------------------------------------------
constexpr int32_t ALGO_COATING_COMPONENTS = 64;


// ---------------------------------------------------------------------------
//  LOW-RESOLUTION GRID BOUNDS
//
//  The field is evaluated on a small grid and interpolated up, because it has no
//  structure finer than its correlation length and evaluating 64 cosines per
//  full-resolution pixel would be pointless.
//
//  Samples per correlation length: 4. This is Nyquist with headroom - two samples
//  per period is the theoretical minimum and gives a visibly faceted result after
//  interpolation, whereas four is smooth.
//
//  Floor of 24 samples per axis. A field whose correlation length is comparable
//  to the whole frame would otherwise be represented by around eight samples and
//  would interpolate into a visibly straight ramp instead of a smooth variation.
//
//  Ceiling of 192 samples per axis. Above this the evaluation cost starts to
//  matter while the interpolated result stops changing, because the field is
//  being sampled far more finely than its own correlation length.
// ---------------------------------------------------------------------------
constexpr AlgoType ALGO_COATING_SAMPLES_PER_CORR = static_cast<AlgoType>(4.0);
constexpr int32_t  ALGO_COATING_LORES_MIN        = 24;
constexpr int32_t  ALGO_COATING_LORES_MAX        = 192;


// ---------------------------------------------------------------------------
//  VARIANCE SPLIT BETWEEN THE TWO FIELD COMPONENTS
//
//  The coating hopper has two distinct signatures, and collapsing them into one
//  field gets the temporal behaviour wrong:
//
//    - The STATIC cross-web profile comes from slot and nozzle imperfections.
//      Those are fixed hardware, so they lay down streaks at fixed positions
//      across the web for the entire roll. A function of horizontal position
//      alone: identical on every frame.
//    - The DRIFTING two-dimensional field comes from coating flow wandering over
//      machine time. This is the part that slides with the web and produces the
//      frame-to-frame variation.
//
//  The two are given equal variance. Since variances add and the total must come
//  to the specified sigma, each component gets sigma / sqrt(2).
//
//  1/sqrt(2) = 0.70710678118654752440.
// ---------------------------------------------------------------------------
constexpr AlgoType ALGO_COATING_HALF_VARIANCE_SCALE =
    static_cast<AlgoType>(0.70710678118654752440);


// ---------------------------------------------------------------------------
//  MINIMUM CORRELATION LENGTH, millimetres
//
//  1.0e-6. Guards the reciprocal used to turn a correlation length into a spatial
//  frequency. A correlation length of zero is meaningless - it would describe a
//  field with no spatial structure at all - but it is representable in the data,
//  and dividing by it would produce an infinite frequency and a field of
//  non-finite values.
// ---------------------------------------------------------------------------
constexpr AlgoType ALGO_COATING_MIN_CORR_MM = static_cast<AlgoType>(1.0e-6);


// ---------------------------------------------------------------------------
//  AlgoVignetteValueAtRadius
//
//  The cos^4 falloff at a normalised radius, where r = 0 is the frame centre and
//  r = 1 is the corner.
//
//  Parametrised by the corner loss in stops so that the number in the profile is
//  directly meaningful:
//
//      cos(theta_corner) = 2^(-stops/4)
//      tan(theta_corner) = sqrt(1/cos^2 - 1)
//      tan(theta)        = r * tan(theta_corner)
//      cos(theta)        = 1 / sqrt(1 + tan^2(theta))
//      value             = cos^4(theta)
//
//  The centre is exactly 1.0 by construction and the corner is exactly the
//  requested loss, with every pixel between interpolated by its true angle
//  rather than by a fitted curve.
//
//  Exposed in the header because it is a pure function worth unit-testing
//  directly, independently of any image.
// ---------------------------------------------------------------------------
AlgoType AlgoVignetteValueAtRadius (const AlgoType rNorm, const AlgoType stops) noexcept;


// ---------------------------------------------------------------------------
//  AlgoStage04_CoatingAndVignette
//
//  Builds the combined multiplier field and applies it to all three channels:
//
//      dst = src * vignette(x, y) * coating(webX, webY)
//
//  Either component may be absent - a zero corner loss, or a stock with no
//  coating variation - in which case that factor is simply unity. If both are
//  absent the planes are copied, so the stage buffer still holds a valid image as
//  the retained-buffer policy requires.
//
//  src        stage-3b output: exposure units, mid grey at 1.0.
//  dst        the stage-4 buffer, same units.
//  fieldFull  one plane: the assembled full-resolution multiplier field.
//  fieldLo    one plane, used only in its top-left corner: the low-resolution
//             coating field before interpolation. Sharing an arena-sized plane
//             for this avoids needing a separate small allocation.
//  geom       frame geometry, supplying pxPerMm, the physical extents, the frame
//             pitch, the frame index and the seed.
//
//  Parameters and buffers are NOT validated; the caller has already done so.
// ---------------------------------------------------------------------------
void AlgoStage04_CoatingAndVignette
(
    const AlgoType* RESTRICT pSrcR,
    const AlgoType* RESTRICT pSrcG,
    const AlgoType* RESTRICT pSrcB,
    AlgoType* RESTRICT       pDstR,
    AlgoType* RESTRICT       pDstG,
    AlgoType* RESTRICT       pDstB,
    AlgoType* RESTRICT       pFieldFull,
    AlgoType* RESTRICT       pFieldLo,
    const int32_t            sizeX,
    const int32_t            sizeY,
    const int32_t            pitch,
    const film::FilmProfile& profile,
    const AlgoControls&      params,
    const AlgoType           negWidthMm,     // exposed frame width on film, mm
    const AlgoType           negHeightMm,    // exposed frame height, mm
    const AlgoType           framePitchMm,   // web advance per frame, mm; 0 = none
    const int32_t            frameIndex,     // signed, clip-relative
    const uint32_t           seed            // field seed
) noexcept;
