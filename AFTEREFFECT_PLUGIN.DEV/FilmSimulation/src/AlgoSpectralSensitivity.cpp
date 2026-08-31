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


    // -----------------------------------------------------------------------
    //  Trapezoidal integral over a contiguous run of uniformly spaced samples.
    //
    //  Returns zero for fewer than two samples, which is what makes the
    //  out-of-reach test below silently correct when a curve stops at or before
    //  the basis limit: there is no tail, so there is no tail integral.
    // -----------------------------------------------------------------------
    HighPrecType trapzUniform (const HighPrecType* v,
                               const int32_t       n,
                               const HighPrecType  step) noexcept
    {
        if (n < 2)
            return 0.0;

        HighPrecType sum = 0.0;

        for (int32_t i = 0; i < n; i++)
            sum += v[i];

        return step * (sum - 0.5 * (v[0] + v[n - 1]));
    }


    // -----------------------------------------------------------------------
    //  THE GAMUT-REACH GUARD. See the long note in the header for why it
    //  exists, why it reads the STORED samples rather than the render grid, and
    //  why it deliberately does not catch ROLLEI_INFRARED_400.
    //
    //  ⚠ Mirrors film_sim.spectral_peak_lambda() and
    //  film_sim.spectral_out_of_reach() exactly, including the "at least two
    //  samples beyond the limit" condition. spectral_mono_parity.py is what
    //  keeps the two in step; it drives THIS function, not a restatement of it.
    // -----------------------------------------------------------------------
    bool panWithinBasisReach (const film::SpectralSensitivity& sp,
                              const HighPrecType cutOnNm) noexcept
    {
        //  cutOnNm is the profile's TAKING FILTER (schema v20, queue C39): the
        //  longpass the stock's look assumes in front of the lens, 0 = none.
        //
        //  \warning THE FILTER GOES IN HERE, BEFORE BOTH TESTS, BECAUSE THE
        //  GUARD AND THE COLLAPSE MUST JUDGE THE SAME EMULSION. Applied later
        //  the guard would pass a bare curve while the weights came off a
        //  filtered one, which is the split this guard was written to close.
        //  ROLLEI_INFRARED_400 is the case: bare it peaks at 410 nm and looks
        //  panchromatic, so this returned true and the engine derived a
        //  near-flat triple for a film nobody shoots unfiltered. Behind the
        //  sheet's own 715 nm filter its peak moves to 720 nm and every
        //  remaining photon is past the basis ceiling, so this now returns
        //  false and the authored red-dominant triple is used.
        const std::size_t n = sp.log_s_pan.size();

        if (n < 2)
            return false;

        const HighPrecType start = static_cast<HighPrecType>(sp.lambda_start_nm);
        const HighPrecType step  = static_cast<HighPrecType>(sp.lambda_step_nm);

        if (!(step > 0.0))
            return false;

        // Linear sensitivity on the curve's OWN sampling, unclipped. The stored
        // values are LOG sensitivity, so they are exponentiated here; nothing is
        // resampled, because resampling is what discarded the far red.
        std::vector<HighPrecType> lin(n);
        std::size_t               peakIdx = 0;

        for (std::size_t i = 0; i < n; i++)
        {
            const HighPrecType nm =
                start + step * static_cast<HighPrecType>(i);

            lin[i] = std::pow(10.0,
                              static_cast<HighPrecType>(sp.log_s_pan[i]));

            //  Ideal step. A real filter has a finite edge and no source
            //  states one; the question this answers is whether the usable
            //  energy lies inside the basis at all, for which the printed
            //  wavelength is enough. Mirrors film_sim.taking_filter_transmission.
            if ((cutOnNm > 0.0) && (nm < cutOnNm))
                lin[i] = 0.0;

            if (lin[i] > lin[peakIdx])
                peakIdx = i;
        }

        const HighPrecType peakNm =
            start + step * static_cast<HighPrecType>(peakIdx);

        if (peakNm > ALGO_SPECTRAL_BASIS_LAMBDA_MAX)
            return false;

        const HighPrecType total = trapzUniform(lin.data(),
                                                static_cast<int32_t>(n), step);

        if (!(total > 0.0))
            return false;

        // First sample strictly beyond the limit. The tail is contiguous
        // because the sampling is uniform and ascending.
        std::size_t first = n;

        for (std::size_t i = 0; i < n; i++)
        {
            if (start + step * static_cast<HighPrecType>(i)
                > ALGO_SPECTRAL_BASIS_LAMBDA_MAX)
            {
                first = i;
                break;
            }
        }

        if (first >= n)
            return true;                       // nothing beyond the limit at all

        const HighPrecType beyond =
            trapzUniform(lin.data() + first,
                         static_cast<int32_t>(n - first), step);

        return (beyond / total) <= ALGO_SPECTRAL_OUT_OF_REACH_MAX;
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

    // ⚠ QUEUE C40. Refuse a stock the visible basis cannot reach, and refuse it
    // BEFORE integrating: the integral is perfectly well defined for an
    // infrared emulsion and perfectly meaningless. Writing nothing on refusal
    // is the contract -- Algo_07_Sim.cpp falls back to profile.spectral_weights,
    // which for these stocks is the authored triple and the right answer.
    if (!panWithinBasisReach(
            profile.spectral,
            static_cast<HighPrecType>(profile.taking_filter_cut_on_nm)))
        return false;

    //  \warning THE FILTER IS APPLIED ON THE INTEGRATION PATH TOO, AND LEAVING
    //  IT OFF WOULD HAVE BEEN INVISIBLE. The guard above judges the FILTERED
    //  emulsion; if the integral below used the bare one, the guard and the
    //  collapse would disagree about which film they were looking at -- and on
    //  every stock the guard refuses, that disagreement never shows.
    //  Mirrors film_sim.layer_sensitivities.
    {
        const HighPrecType cutOn =
            static_cast<HighPrecType>(profile.taking_filter_cut_on_nm);

        if (cutOn > 0.0)
        {
            for (int32_t i = 0; i < ALGO_SPECTRAL_N; i++)
            {
                if (gridLambda(i) < cutOn)
                    sens[0][i] = 0.0;
            }
        }
    }

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
