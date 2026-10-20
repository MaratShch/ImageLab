#pragma once

// ---------------------------------------------------------------------------
//  AlgoDefectField.hpp
//
//  The clumped spatial process every particulate defect class is placed by.
//
//  WHY UNIFORM RANDOM PLACEMENT IS WRONG
//
//  Scatter a thousand dust specks with a uniform random number generator and the
//  result looks wrong. Viewers notice without being able to say why. Real
//  particulate contamination arrives in patches: some regions of a frame carry
//  five times the average and others almost none.
//
//  This is measured, not asserted. Point-pattern analysis of four blank frames:
//
//      index of dispersion at  1.1 mm2 cells   5.3 - 9.2     Poisson = 1.0
//      index of dispersion at  4.5 mm2 cells  15.5 - 31.7    Poisson = 1.0
//      index of dispersion at 18.0 mm2 cells  46.9 - 111.0   Poisson = 1.0
//      Clark-Evans nearest-neighbour ratio     0.82 - 0.97   Poisson = 1.0
//
//  Read those two rows together, because the combination is diagnostic. A
//  Clark-Evans ratio near 0.9 says the particles are only SLIGHTLY clustered at
//  the scale of their own separation - they do not stick together in tight clumps.
//  But a dispersion index rising from ~5 at one square millimetre to ~110 at
//  eighteen says the LOCAL RATE ITSELF varies strongly across the frame.
//
//  That is the exact signature of a COX PROCESS - a Poisson process whose
//  intensity is itself a random field. Particles land independently; the
//  probability of landing varies smoothly from place to place.
//
//  THE MODEL
//
//      lambda(x,y) = lambda0 * exp( sigma_g * G(x,y) - sigma_g^2 / 2 )
//
//      G        zero-mean unit-variance Gaussian field, power spectrum ~ 1/f^beta
//      sigma_g  sqrt(ln(1 + CV^2)),  CV = 0.88 measured  ->  sigma_g = 0.757
//      beta     1.0, which makes CV very nearly scale-free over 1 - 20 mm
//
//  The exp(-sigma_g^2/2) term is what keeps the MEAN of lambda equal to lambda0.
//  Without it, raising the clumpiness would also raise the total amount of dirt,
//  so the clumping control would double as a level control. Inverting the measured
//  dispersion at three scales gave CV = 0.83 - 0.92 - near-constant across a
//  sixteen-fold range of cell areas, which is why beta is 1 rather than a
//  single-scale blob field.
//
//  THE FIELD IS SEEDED BY FILM POSITION, NOT BY FRAME
//
//  The clumping field is a property of the FILM. It is sampled at the film
//  coordinate, so a patch of dirty film stays dirty as it travels through the
//  gate, and a defect straddling a frame line matches up on both sides. Seeding
//  it per frame instead makes the field boil rather than drift - a mistake that
//  looks like animated noise rather than like film.
//
//  PLACEMENT IS STRATIFIED, NOT REJECTION-SAMPLED
//
//  The obvious way to place points against a varying intensity is rejection
//  sampling, which needs an unbounded loop and a sequential generator. Neither is
//  acceptable here: the engine must be a pure function of (seed, frame, ordinal)
//  with no accumulating state, because the host renders frames out of order and in
//  parallel.
//
//  So the film is divided into fixed cells in FILM coordinates. Each cell gets its
//  own intensity from the field, its own Poisson count from the counter-based
//  generator keyed on the cell's integer index, and its particles placed uniformly
//  inside it. This reproduces the measured dispersion at every scale, needs no
//  loop of unknown length, and gives identical results whichever frame is rendered
//  first.
// ---------------------------------------------------------------------------

// Project-wide primitives, included unconditionally as required by the project
// coding standard.
#include "Common.hpp"
#include "CompileTimeUtils.hpp"

// The single source of the engine's numeric types and alignment policy.
#include "AlgoTypes.hpp"
// 2026-10-20c: this header's compile-time constants live in film_damage_constants.hpp.
#include "film_damage_constants.hpp"

// The film-fixed coordinate system every defect is generated in.
#include "AlgoFilmCoord.hpp"

// Counter-based generator: stateless, order-independent, frame-addressable.
#include "AlgoCounterRng.hpp"

#include <cmath>    // std::sqrt, std::cos, std::sin, std::atan2
#include <cstdint>   // int32_t, uint32_t, uint64_t

// ---------------------------------------------------------------------------
//  AlgoDefectHash
//
//  Mix three integers into a generator counter.
//
//  Used to key the field lattice and the placement cells on their INTEGER FILM
//  COORDINATES, which is what makes the field a property of the film rather than
//  of the frame. Handles negative indices, because film coordinates run backwards
//  off the head of the roll and frame indices may be negative.
// ---------------------------------------------------------------------------
FORCE_INLINE uint64_t AlgoDefectHash
(
    const uint32_t seed,
    const int32_t  a,
    const int32_t  b,
    const uint32_t tag
) noexcept
{
    // Reinterpret the signed indices as unsigned so negative values hash as
    // ordinary bit patterns rather than sign-extending into each other.
    const uint64_t ua = static_cast<uint64_t>(static_cast<uint32_t>(a));
    const uint64_t ub = static_cast<uint64_t>(static_cast<uint32_t>(b));

    // Odd multipliers from the 64-bit golden-ratio family, so each field occupies
    // a distinct region of the argument space and the mixer sees well-spread bits.
    uint64_t h = static_cast<uint64_t>(seed);

    h ^= ua * 0x9E3779B97F4A7C15ull;
    h ^= ub * 0xC2B2AE3D27D4EB4Full;
    h ^= static_cast<uint64_t>(tag) * 0x165667B19E3779F9ull;

    return AlgoRngMix64(h);
}

// ---------------------------------------------------------------------------
//  AlgoDefectFieldValue
//
//  Zero-mean, approximately unit-variance 1/f^beta field, sampled at a film
//  coordinate in millimetres.
//
//  Built as a sum of value-noise octaves. Each octave is a lattice of independent
//  Gaussian values at its own spacing, smoothly interpolated; the amplitudes are
//  weighted so the summed power spectrum follows 1/f^beta and the total variance
//  is one.
//
//  Deterministic in film position: two frames that overlap the same film see the
//  same field values there.
// ---------------------------------------------------------------------------
HighPrecType AlgoDefectFieldValue
(
    const HighPrecType alongMm,
    const HighPrecType acrossMm,
    const uint32_t     seed,
    const uint32_t     tag
) noexcept;

// ---------------------------------------------------------------------------
//  AlgoDefectCoxIntensity
//
//  Turn a field sample into a local intensity, in particles per square
//  millimetre.
//
//  lambda0    the mean intensity the caller wants, per mm2
//  fieldValue a sample from AlgoDefectFieldValue
//  clumping   scale on the measured CV; 0 gives a uniform Poisson process
//
//  The exp(-sigma^2/2) correction keeps the mean at lambda0 whatever the
//  clumpiness, so the two controls stay independent.
// ---------------------------------------------------------------------------
HighPrecType AlgoDefectCoxIntensity
(
    const HighPrecType lambda0,
    const HighPrecType fieldValue,
    const HighPrecType clumping
) noexcept;

// ---------------------------------------------------------------------------
//  AlgoDefectPoisson
//
//  Poisson deviate from a counter, by Knuth's product method.
//
//  Bounded by ALGO_DEFECT_MAX_PER_CELL, so the loop cannot run away on a
//  pathological intensity. At the per-cell means this engine uses - single
//  figures - the method needs a handful of iterations and is exact.
// ---------------------------------------------------------------------------
int32_t AlgoDefectPoisson
(
    const uint64_t     counter,
    const HighPrecType mean
) noexcept;

// ---------------------------------------------------------------------------
//  AlgoDefectPowerLawSize
//
//  Sample a truncated power-law diameter, p(d) proportional to d^-gamma over
//  [dMin, dMax], by inverting the cumulative distribution exactly.
//
//  The measured dust size histogram is heavy-tailed with gamma = 2.6 over the
//  resolved 18 - 107 micrometre range. Exact inversion rather than rejection
//  because it is one expression and needs no loop.
// ---------------------------------------------------------------------------
HighPrecType AlgoDefectPowerLawSize
(
    const HighPrecType u,
    const HighPrecType dMin,
    const HighPrecType dMax,
    const HighPrecType gamma
) noexcept;

// ---------------------------------------------------------------------------
//  AlgoDefectBeta23
//
//  A Beta(2,3) deviate, for particle opacity.
//
//  Sampled exactly as the SECOND SMALLEST of four uniforms, which is the second
//  order statistic of four draws and is distributed exactly Beta(2,3). Four
//  generator calls, no transcendentals, no rejection loop.
//
//  Beta(2,3) has mean 0.4 and is skewed towards the low end, which matches the
//  observation that most particles are partially transmissive and only the large
//  ones are opaque.
// ---------------------------------------------------------------------------
HighPrecType AlgoDefectBeta23 (const uint64_t counter) noexcept;

// ---------------------------------------------------------------------------
//  AlgoDefectBlobCoverage
//
//  Fraction of a pixel covered by one irregular blob, 0 to 1.
//
//  Shared by the film-borne particulate at stage 9b and the machine-side dirt at
//  stage 16. The two work in DIFFERENT domains - one adds density on the negative,
//  the other multiplies transmittance on the positive - so they cannot share a
//  rasteriser, but the SHAPE is the same physics and must not be written twice.
//  A second copy would drift, and the first symptom would be dirt that looks
//  subtly unlike itself depending on which side of the print it came from.
//
//  dx, dy      pixel centre relative to the blob centre, pixels
//  radiusPx    mean radius
//  aspect      long axis over short axis, at least 1
//  angleRad    orientation of the long axis in the image frame
//  lobeDepth   harmonic modulation depth, as a fraction of the radius
//  phase       one phase per harmonic, so no two blobs are alike
//  edgePx      width of the edge transition; the system point-spread function
//
//  The edge is a smooth step rather than a hard test, and its width is the PSF,
//  because nothing in a real imaging chain has an edge sharper than its own
//  optics. A hard-edged speck is one of the two reliable ways to make dirt look
//  composited; the other is placing it uniformly.
// ---------------------------------------------------------------------------
FORCE_INLINE HighPrecType AlgoDefectBlobCoverage
(
    const HighPrecType dx,
    const HighPrecType dy,
    const HighPrecType radiusPx,
    const HighPrecType aspect,
    const HighPrecType angleRad,
    const HighPrecType lobeDepth,
    const HighPrecType phase[ALGO_DEFECT_BLOB_HARMONICS],
    const HighPrecType edgePx
) noexcept
{
    // Into the blob's own frame: rotate, then compress the long axis so an
    // ellipse can be tested as a circle.
    const HighPrecType ca = std::cos(angleRad);
    const HighPrecType sa = std::sin(angleRad);

    const HighPrecType u =  dx * ca + dy * sa;
    const HighPrecType v = -dx * sa + dy * ca;

    const HighPrecType uu = u / aspect;

    const HighPrecType r = std::sqrt(uu * uu + v * v);

    // Angular harmonics modulate the radius, so the outline is irregular. Circles
    // read as digital immediately, and a polygon costs more while looking no
    // better once the point-spread function has softened the boundary.
    HighPrecType shape = 1.0;

    if (r > 0.0)
    {
        const HighPrecType theta = std::atan2(v, uu);

        for (int32_t k = 0; k < ALGO_DEFECT_BLOB_HARMONICS; k++)
        {
            const HighPrecType order = static_cast<HighPrecType>(k + 2);

            shape += (lobeDepth / order) * std::cos(order * theta + phase[k]);
        }
    }

    const HighPrecType boundary = radiusPx * shape;

    // One half-edge inside the boundary the pixel is fully covered, one half-edge
    // outside it is clear.
    const HighPrecType t = CLAMP_VALUE(((boundary - r) / edgePx) + 0.5, 0.0, 1.0);

    // The cubic smooth step. Its derivative vanishes at both ends, so there is no
    // visible corner where the transition meets either the blob interior or the
    // clean film - which a linear ramp does have, at exactly the scale dirt
    // occupies.
    return t * t * (3.0 - 2.0 * t);
}
