// ---------------------------------------------------------------------------
//  AlgoDefectField.cpp
//
//  The clumped spatial process behind every particulate defect class, plus the
//  small sampling primitives the generators draw from.
//
//  Plain scalar code. No intrinsics, no manual vectorisation - correctness and
//  legibility first, as with every other stage in the engine.
// ---------------------------------------------------------------------------

#include "AlgoDefectField.hpp"

#include <cmath>   // std::sqrt, std::exp, std::log, std::pow, std::floor


namespace
{
    // ----------------------------------------------------------------------
    //  Smooth interpolation weight, the classic 3t^2 - 2t^3 Hermite.
    //
    //  Used instead of a straight linear blend between lattice values because
    //  linear interpolation leaves a visible crease along every lattice line: its
    //  first derivative is discontinuous there. The Hermite form has a zero
    //  derivative at both ends, so octaves add without the lattice showing
    //  through as a grid - which on a dust field would read as a printed pattern.
    // ----------------------------------------------------------------------
    inline HighPrecType smoothStep (const HighPrecType t) noexcept
    {
        return t * t * (3.0 - 2.0 * t);
    }


    // ----------------------------------------------------------------------
    //  One Gaussian lattice value, from the film-coordinate lattice index.
    //
    //  Keyed on the INTEGER lattice coordinates, so the value at a given place on
    //  the film is the same whichever frame happens to be looking at it. That is
    //  the property that makes the field drift with the film rather than boil.
    // ----------------------------------------------------------------------
    inline HighPrecType latticeValue
    (
        const uint32_t seed,
        const int32_t  i,
        const int32_t  j,
        const uint32_t octaveTag
    ) noexcept
    {
        return AlgoRngNormal(AlgoDefectHash(seed, i, j, octaveTag));
    }


    // ----------------------------------------------------------------------
    //  One octave of value noise: bilinear-with-Hermite interpolation of a
    //  Gaussian lattice at the given spacing.
    // ----------------------------------------------------------------------
    HighPrecType octaveValue
    (
        const HighPrecType alongMm,
        const HighPrecType acrossMm,
        const HighPrecType spacingMm,
        const uint32_t     seed,
        const uint32_t     octaveTag
    ) noexcept
    {
        // Position in lattice units.
        const HighPrecType a = alongMm  / spacingMm;
        const HighPrecType b = acrossMm / spacingMm;

        // Cell index. std::floor rather than a cast, because a cast truncates
        // towards zero and would fold the two sides of the origin onto the same
        // lattice cell - a visible seam at the head of the roll.
        const HighPrecType fa = std::floor(a);
        const HighPrecType fb = std::floor(b);

        const int32_t i = static_cast<int32_t>(fa);
        const int32_t j = static_cast<int32_t>(fb);

        // Fractional position within the cell, smoothed.
        const HighPrecType wa = smoothStep(a - fa);
        const HighPrecType wb = smoothStep(b - fb);

        // The four corners.
        const HighPrecType v00 = latticeValue(seed, i,     j,     octaveTag);
        const HighPrecType v10 = latticeValue(seed, i + 1, j,     octaveTag);
        const HighPrecType v01 = latticeValue(seed, i,     j + 1, octaveTag);
        const HighPrecType v11 = latticeValue(seed, i + 1, j + 1, octaveTag);

        // Two interpolations along, then one across.
        const HighPrecType lo = v00 + (v10 - v00) * wa;
        const HighPrecType hi = v01 + (v11 - v01) * wa;

        return lo + (hi - lo) * wb;
    }
}


// ---------------------------------------------------------------------------
//  AlgoDefectFieldValue
// ---------------------------------------------------------------------------
HighPrecType AlgoDefectFieldValue
(
    const HighPrecType alongMm,
    const HighPrecType acrossMm,
    const uint32_t     seed,
    const uint32_t     tag
) noexcept
{
    // ----------------------------------------------------------------------
    //  Octave amplitudes for a 1/f^beta power spectrum.
    //
    //  An octave whose lattice spacing is s carries spatial frequencies near 1/s.
    //  For power proportional to f^-beta the amplitude must be proportional to
    //  f^(-beta/2), which is s^(beta/2). With beta = 1 that is sqrt(s) - so each
    //  successive octave, at twice the spacing, gets sqrt(2) times the amplitude,
    //  and the coarse structure dominates. Which is what 1/f means, and what the
    //  measurement showed: more energy at large scales, in a way that keeps the
    //  coefficient of variation almost constant with measurement scale.
    // ----------------------------------------------------------------------
    HighPrecType sum      = 0.0;
    HighPrecType variance = 0.0;

    HighPrecType spacing = ALGO_DEFECT_FIELD_BASE_MM;

    for (int32_t k = 0; k < ALGO_DEFECT_FIELD_OCTAVES; k++)
    {
        // s^(beta/2), computed from the spacing ratio so the finest octave has
        // amplitude one and the constant cancels in the normalisation below.
        const HighPrecType amplitude = std::pow(spacing / ALGO_DEFECT_FIELD_BASE_MM,
                                                ALGO_DEFECT_FIELD_BETA * 0.5);

        // The octave tag separates the lattices, so two octaves never draw the
        // same value at the same integer coordinate. Combined with the caller's
        // tag so that, say, the dust field and the mottle field are independent
        // even though both are 1/f over the same film.
        const uint32_t octaveTag = tag * 64u + static_cast<uint32_t>(k);

        sum += amplitude * octaveValue(alongMm, acrossMm, spacing, seed, octaveTag);

        // Octaves are independent, so their variances add.
        variance += amplitude * amplitude;

        spacing *= 2.0;
    }

    // ----------------------------------------------------------------------
    //  Normalise to unit variance.
    //
    //  Two corrections, and both are needed.
    //
    //  The first is the octave sum: independent octaves' variances add, so the
    //  total must be divided by the root of their summed squared amplitudes.
    //  Without it the field's spread would depend on the octave count.
    //
    //  The second is the INTERPOLATION LOSS. Each octave is a lattice of
    //  unit-variance values smoothly interpolated, and interpolation is averaging:
    //  a point inside a cell has variance sum(w^2) < 1. Over the cell that factor
    //  is exactly (26/35)^2 = 0.5518, so the octave's real variance is that times
    //  the amplitude squared, not the amplitude squared.
    //
    //  Omitting the second correction leaves sigma at 0.746 rather than 1.0, which
    //  delivers a coefficient of variation of 0.63 against the measured 0.88 and
    //  an index of dispersion roughly three times too low at every scale.
    // ----------------------------------------------------------------------
    const HighPrecType norm = variance * ALGO_DEFECT_INTERP_VARIANCE;

    return (norm > 0.0) ? (sum / std::sqrt(norm)) : 0.0;
}


// ---------------------------------------------------------------------------
//  AlgoDefectCoxIntensity
// ---------------------------------------------------------------------------
HighPrecType AlgoDefectCoxIntensity
(
    const HighPrecType lambda0,
    const HighPrecType fieldValue,
    const HighPrecType clumping
) noexcept
{
    // Zero clumping means a homogeneous Poisson process: the intensity is the same
    // everywhere. Returned directly rather than falling through the exponential,
    // both to avoid the work and to guarantee the degenerate case is exact.
    if (clumping <= 0.0)
        return lambda0;

    // Coefficient of variation the caller is asking for.
    const HighPrecType cv = ALGO_DEFECT_FIELD_CV * clumping;

    // sigma of the underlying Gaussian that produces that CV after exponentiation.
    // For a log-normal, CV^2 = exp(sigma^2) - 1, so sigma = sqrt(ln(1 + CV^2)).
    // At the measured CV of 0.88 this is 0.757.
    const HighPrecType sigma = std::sqrt(std::log(1.0 + cv * cv));

    // The -sigma^2/2 term is what keeps the MEAN of the result equal to lambda0.
    // A log-normal with parameters (mu, sigma) has mean exp(mu + sigma^2/2), so
    // setting mu = ln(lambda0) - sigma^2/2 gives a mean of exactly lambda0.
    //
    // Without it, turning up the clumpiness would also turn up the total amount of
    // dirt, and the two controls would not be separable.
    return lambda0 * std::exp(sigma * fieldValue - 0.5 * sigma * sigma);
}


// ---------------------------------------------------------------------------
//  AlgoDefectPoisson
// ---------------------------------------------------------------------------
int32_t AlgoDefectPoisson
(
    const uint64_t     counter,
    const HighPrecType mean
) noexcept
{
    if (mean <= 0.0)
        return 0;

    // Knuth's product method: multiply uniforms until their product falls below
    // exp(-mean), and count how many it took. Exact for any positive mean, and at
    // the single-figure means this engine uses it needs only a few iterations.
    //
    // exp(-mean) is formed once. For a large mean it would underflow to zero and
    // the loop would run to the cap; that is why the cap exists, and why the
    // per-cell area is chosen to keep the mean small.
    const HighPrecType limit = std::exp(-mean);

    HighPrecType product = 1.0;
    int32_t      count   = 0;

    while (count < ALGO_DEFECT_MAX_PER_CELL)
    {
        // A fresh uniform each iteration, from the counter advanced by the
        // iteration number. Advancing the COUNTER rather than carrying generator
        // state is what keeps this a pure function of its arguments.
        product *= AlgoRngUniform01(counter + static_cast<uint64_t>(count) * 0x9E3779B9ull);

        if (product <= limit)
            break;

        count++;
    }

    return count;
}


// ---------------------------------------------------------------------------
//  AlgoDefectPowerLawSize
// ---------------------------------------------------------------------------
HighPrecType AlgoDefectPowerLawSize
(
    const HighPrecType u,
    const HighPrecType dMin,
    const HighPrecType dMax,
    const HighPrecType gamma
) noexcept
{
    // A power law with exponent exactly 1 has a logarithmic integral rather than a
    // power one, so the general inversion below divides by zero there. The
    // measured exponent is 2.6 and the documented range is 2.2 - 3.0, so this
    // branch is a guard rather than a case the model uses.
    const HighPrecType e = 1.0 - gamma;

    if (std::fabs(e) < 1.0e-9)
    {
        // Log-uniform: the correct limit of the power law as gamma tends to 1.
        return dMin * std::exp(u * std::log(dMax / dMin));
    }

    // Exact inversion of the truncated cumulative distribution:
    //
    //     F(d) = (d^e - dMin^e) / (dMax^e - dMin^e)
    //     d    = (dMin^e + u * (dMax^e - dMin^e))^(1/e)
    //
    // One pow to invert, two to set up. Exact, and no rejection loop.
    const HighPrecType lo = std::pow(dMin, e);
    const HighPrecType hi = std::pow(dMax, e);

    return std::pow(lo + u * (hi - lo), 1.0 / e);
}


// ---------------------------------------------------------------------------
//  AlgoDefectBeta23
// ---------------------------------------------------------------------------
HighPrecType AlgoDefectBeta23 (const uint64_t counter) noexcept
{
    // Beta(2,3) is exactly the distribution of the SECOND SMALLEST of four
    // independent uniforms. So draw four and take the second order statistic --
    // no transcendentals, no rejection, and exact rather than approximate.
    HighPrecType u[4];

    for (int32_t k = 0; k < 4; k++)
        u[k] = AlgoRngUniform01(counter + static_cast<uint64_t>(k) * 0xC2B2AE3Dull);

    // Partial selection sort: two passes are enough to put the two smallest in
    // place, and the second of those is the answer.
    for (int32_t pass = 0; pass < 2; pass++)
    {
        for (int32_t k = pass + 1; k < 4; k++)
        {
            if (u[k] < u[pass])
            {
                const HighPrecType t = u[pass];
                u[pass] = u[k];
                u[k]    = t;
            }
        }
    }

    return u[1];
}
