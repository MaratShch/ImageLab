// ---------------------------------------------------------------------------
//  AlgoSpectralSensitivity.cpp
//
//  Implementation of the measured-spectral-sensitivity consumers. See the header
//  for why this file exists, what it is and is not, and why the taking matrix is
//  computed but not wired into the pipeline.
//
//  Every routine here runs once per frame at most. There is no per-pixel code in
//  this file, which is why all arithmetic is HighPrecType.
// ---------------------------------------------------------------------------

#include "AlgoSpectralSensitivity.hpp"

#include <cmath>
#include <vector>


namespace
{
    // -----------------------------------------------------------------------
    //  Blackbody spectral radiance, Planck's law.
    //
    //  Same constants as AlgoStockColourBalance so the two agree exactly where
    //  they overlap; duplicated as file-local rather than shared because a
    //  cross-header dependency between two stage helpers buys nothing and the
    //  two values are physical constants, not tuning parameters.
    //
    //  c1 / (lambda^5 * (exp(c2 / (lambda * T)) - 1)), lambda in METRES.
    //
    //  expm1 rather than exp minus one: at low temperature and long wavelength
    //  the argument becomes small and the naive form loses every significant
    //  digit to cancellation.
    // -----------------------------------------------------------------------
    constexpr HighPrecType PLANCK_C1 = 3.741771e-16;
    constexpr HighPrecType PLANCK_C2 = 1.438777e-2;
    constexpr HighPrecType NM_TO_M   = 1.0e-9;

    HighPrecType planck (const HighPrecType lambdaNm,
                         const HighPrecType kelvin) noexcept
    {
        if (kelvin <= 0.0 || lambdaNm <= 0.0)
            return 0.0;

        const HighPrecType lam = lambdaNm * NM_TO_M;
        const HighPrecType l2  = lam * lam;
        const HighPrecType l5  = l2 * l2 * lam;
        const HighPrecType arg = PLANCK_C2 / (lam * kelvin);

        // Guard the exponential: at very short wavelength and low temperature the
        // argument overflows, and the physical answer there is zero radiance.
        if (arg > 700.0)
            return 0.0;

        const HighPrecType denom = l5 * std::expm1(arg);
        return (denom > 0.0) ? (PLANCK_C1 / denom) : 0.0;
    }


    // -----------------------------------------------------------------------
    //  Wavelength of grid sample i.
    // -----------------------------------------------------------------------
    inline HighPrecType gridLambda (const int32_t i) noexcept
    {
        return ALGO_SPECTRAL_LAMBDA_MIN
             + ALGO_SPECTRAL_LAMBDA_STEP * static_cast<HighPrecType>(i);
    }


    // -----------------------------------------------------------------------
    //  Resample one stored LOG-sensitivity curve onto the common grid and
    //  exponentiate it to LINEAR sensitivity.
    //
    //  Interpolation is done in LOG space, deliberately. A sensitisation curve
    //  is smooth in log sensitivity and emphatically not in linear sensitivity,
    //  where a four-decade span would make linear interpolation between two
    //  samples grossly wrong - the interpolated value would sit near the larger
    //  endpoint instead of near the geometric middle.
    //
    //  Outside the curve's own measured range the output is ZERO, not an
    //  extrapolated tail: see the header note on why extrapolation is refused.
    // -----------------------------------------------------------------------
    void resampleLinear (const std::vector<double>& logS,
                         const HighPrecType         startNm,
                         const HighPrecType         stepNm,
                         HighPrecType               out[ALGO_SPECTRAL_N]) noexcept
    {
        const int32_t n = static_cast<int32_t>(logS.size());

        for (int32_t i = 0; i < ALGO_SPECTRAL_N; i++)
            out[i] = 0.0;

        if (n < 2 || stepNm <= 0.0)
            return;

        const HighPrecType endNm = startNm
                                 + stepNm * static_cast<HighPrecType>(n - 1);

        for (int32_t i = 0; i < ALGO_SPECTRAL_N; i++)
        {
            const HighPrecType lam = gridLambda(i);

            // Outside the measured range: no sensitivity, no invention.
            if (lam < startNm || lam > endNm)
                continue;

            const HighPrecType pos = (lam - startNm) / stepNm;

            int32_t k = static_cast<int32_t>(pos);
            if (k < 0)          k = 0;
            if (k > (n - 2))    k = n - 2;

            const HighPrecType frac = pos - static_cast<HighPrecType>(k);

            // Linear in log sensitivity, then exponentiate.
            const HighPrecType lg = logS[static_cast<size_t>(k)] * (1.0 - frac)
                                  + logS[static_cast<size_t>(k + 1)] * frac;

            out[i] = std::pow(10.0, lg);
        }
    }


    // -----------------------------------------------------------------------
    //  Trapezoidal integral over the common grid. The grid is uniform, so the
    //  rule reduces to step * (sum of interior + half of each endpoint).
    // -----------------------------------------------------------------------
    HighPrecType integrate (const HighPrecType f[ALGO_SPECTRAL_N]) noexcept
    {
        HighPrecType acc = 0.5 * (f[0] + f[ALGO_SPECTRAL_N - 1]);

        for (int32_t i = 1; i < (ALGO_SPECTRAL_N - 1); i++)
            acc += f[i];

        return acc * ALGO_SPECTRAL_LAMBDA_STEP;
    }


    // -----------------------------------------------------------------------
    //  One smooth primary SPD, normalised to unit area so that an equal RGB
    //  triple corresponds to a smooth broadband spectrum rather than to three
    //  arbitrarily weighted lobes.
    // -----------------------------------------------------------------------
    void primarySpd (const HighPrecType centreNm,
                     HighPrecType       out[ALGO_SPECTRAL_N]) noexcept
    {
        for (int32_t i = 0; i < ALGO_SPECTRAL_N; i++)
        {
            const HighPrecType d = (gridLambda(i) - centreNm)
                                 / ALGO_SPECTRAL_PRIMARY_WIDTH_NM;
            out[i] = std::exp(-0.5 * d * d);
        }

        const HighPrecType area = integrate(out);

        if (area > 0.0)
        {
            for (int32_t i = 0; i < ALGO_SPECTRAL_N; i++)
                out[i] /= area;
        }
    }


    // -----------------------------------------------------------------------
    //  Blackbody SPD on the grid, normalised to unity at 560 nm.
    //
    //  Every use forms a RATIO, so the normalisation cancels and cannot affect
    //  a result. It exists only to keep the intermediates near 1 instead of
    //  near 1e13, which makes the values readable in a debugger.
    // -----------------------------------------------------------------------
    void blackbodySpd (const HighPrecType kelvin,
                       HighPrecType       out[ALGO_SPECTRAL_N]) noexcept
    {
        for (int32_t i = 0; i < ALGO_SPECTRAL_N; i++)
            out[i] = planck(gridLambda(i), kelvin);

        const HighPrecType ref = planck(560.0, kelvin);

        if (ref > 0.0)
        {
            for (int32_t i = 0; i < ALGO_SPECTRAL_N; i++)
                out[i] /= ref;
        }
    }


    // -----------------------------------------------------------------------
    //  Fetch the three colour layers, or the single pan layer, as linear
    //  sensitivity on the common grid.
    //
    //  Returns the layer count: 3 for a colour stock, 1 for a pan-only stock,
    //  0 when there is nothing usable.
    // -----------------------------------------------------------------------
    int32_t fetchLayers (const film::SpectralSensitivity& sp,
                         HighPrecType sens[3][ALGO_SPECTRAL_N]) noexcept
    {
        if (!sp.log_s_r.empty() && !sp.log_s_g.empty() && !sp.log_s_b.empty())
        {
            resampleLinear(sp.log_s_r, sp.lambda_start_nm, sp.lambda_step_nm, sens[0]);
            resampleLinear(sp.log_s_g, sp.lambda_start_nm, sp.lambda_step_nm, sens[1]);
            resampleLinear(sp.log_s_b, sp.lambda_start_nm, sp.lambda_step_nm, sens[2]);
            return 3;
        }

        if (!sp.log_s_pan.empty())
        {
            resampleLinear(sp.log_s_pan, sp.lambda_start_nm, sp.lambda_step_nm, sens[0]);
            return 1;
        }

        return 0;
    }
}   // anonymous namespace


// ---------------------------------------------------------------------------
bool AlgoSpectralHasCurves (const film::FilmProfile& profile) noexcept
{
    return profile.spectral.hasData();
}


// ---------------------------------------------------------------------------
bool AlgoSpectralBalanceGains
(
    const film::FilmProfile& profile,
    const HighPrecType       sceneKelvin,
    AlgoType                 gains[3]
) noexcept
{
    HighPrecType sens[3][ALGO_SPECTRAL_N];

    if (fetchLayers(profile.spectral, sens) != 3)
        return false;

    const HighPrecType stockKelvin =
        static_cast<HighPrecType>(profile.balance_kelvin);

    if (sceneKelvin <= 0.0 || stockKelvin <= 0.0)
        return false;

    HighPrecType sceneSpd[ALGO_SPECTRAL_N];
    HighPrecType stockSpd[ALGO_SPECTRAL_N];
    blackbodySpd(sceneKelvin, sceneSpd);
    blackbodySpd(stockKelvin, stockSpd);

    HighPrecType ratio[3];

    for (int32_t c = 0; c < 3; c++)
    {
        HighPrecType prodScene[ALGO_SPECTRAL_N];
        HighPrecType prodStock[ALGO_SPECTRAL_N];

        for (int32_t i = 0; i < ALGO_SPECTRAL_N; i++)
        {
            prodScene[i] = sens[c][i] * sceneSpd[i];
            prodStock[i] = sens[c][i] * stockSpd[i];
        }

        const HighPrecType eScene = integrate(prodScene);
        const HighPrecType eStock = integrate(prodStock);

        // A layer with no response under the stock's own balance illuminant
        // cannot yield a ratio. Refuse rather than divide.
        if (!(eStock > 0.0))
            return false;

        ratio[c] = eScene / eStock;
    }

    if (!(ratio[1] > 0.0))
        return false;

    // Green normalised to exactly 1.0: this stage changes the balance between
    // records, never the overall exposure.
    for (int32_t c = 0; c < 3; c++)
        gains[c] = static_cast<AlgoType>(ratio[c] / ratio[1]);

    gains[1] = static_cast<AlgoType>(1.0);
    return true;
}


// ---------------------------------------------------------------------------
bool AlgoSpectralMonoWeights
(
    const film::FilmProfile& profile,
    AlgoType                 weights[3]
) noexcept
{
    HighPrecType sens[3][ALGO_SPECTRAL_N];

    if (fetchLayers(profile.spectral, sens) != 1)
        return false;

    const HighPrecType centres[3] =
    {
        ALGO_SPECTRAL_PRIMARY_R_NM,
        ALGO_SPECTRAL_PRIMARY_G_NM,
        ALGO_SPECTRAL_PRIMARY_B_NM
    };

    HighPrecType w[3];
    HighPrecType total = 0.0;

    for (int32_t c = 0; c < 3; c++)
    {
        HighPrecType prim[ALGO_SPECTRAL_N];
        primarySpd(centres[c], prim);

        HighPrecType prod[ALGO_SPECTRAL_N];

        for (int32_t i = 0; i < ALGO_SPECTRAL_N; i++)
            prod[i] = sens[0][i] * prim[i];

        w[c]   = integrate(prod);
        total += w[c];
    }

    if (!(total > 0.0))
        return false;

    for (int32_t c = 0; c < 3; c++)
        weights[c] = static_cast<AlgoType>(w[c] / total);

    return true;
}


// ---------------------------------------------------------------------------
//  Computed, reported, and deliberately not wired into the pipeline. The header
//  states why at length: the pipeline already carries cross-channel mixing in
//  dye_matrix and in InterimageSpec, and stacking a third mixing stage on top
//  without a measured reference to validate against would double-count the same
//  physics.
// ---------------------------------------------------------------------------
bool AlgoSpectralTakingMatrix
(
    const film::FilmProfile& profile,
    const HighPrecType       sceneKelvin,
    film::Matrix3&           matrixOut
) noexcept
{
    HighPrecType sens[3][ALGO_SPECTRAL_N];

    if (fetchLayers(profile.spectral, sens) != 3)
        return false;

    if (sceneKelvin <= 0.0)
        return false;

    HighPrecType illum[ALGO_SPECTRAL_N];
    blackbodySpd(sceneKelvin, illum);

    const HighPrecType centres[3] =
    {
        ALGO_SPECTRAL_PRIMARY_R_NM,
        ALGO_SPECTRAL_PRIMARY_G_NM,
        ALGO_SPECTRAL_PRIMARY_B_NM
    };

    HighPrecType prim[3][ALGO_SPECTRAL_N];

    for (int32_t p = 0; p < 3; p++)
        primarySpd(centres[p], prim[p]);

    for (int32_t l = 0; l < 3; l++)
    {
        HighPrecType row[3];
        HighPrecType rowSum = 0.0;

        for (int32_t p = 0; p < 3; p++)
        {
            HighPrecType prod[ALGO_SPECTRAL_N];

            for (int32_t i = 0; i < ALGO_SPECTRAL_N; i++)
                prod[i] = sens[l][i] * prim[p][i] * illum[i];

            row[p]  = integrate(prod);
            rowSum += row[p];
        }

        if (!(rowSum > 0.0))
            return false;

        // Row-normalised: a neutral input stays neutral, and the matrix carries
        // only the cross-channel mixing that is genuinely the film's character.
        for (int32_t p = 0; p < 3; p++)
            matrixOut[static_cast<size_t>(l)][static_cast<size_t>(p)] =
                static_cast<float>(row[p] / rowSum);
    }

    return true;
}
