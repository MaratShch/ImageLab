#pragma once

// ---------------------------------------------------------------------------
//  AlgoDefectOptics.hpp
//
//  Shared helpers for sub-stage 9b, added 2026-10-20:
//
//    1. THE DEFECT WRITE. How one defect's opacity at one pixel becomes a
//       change of negative density, with the reader-geometry law and the
//       Callier exclusion behind FilmDamage::defectOptics (default OFF).
//    2. STOCHASTIC PRIMITIVES the defect generators use for along-stroke
//       variation and interior texture: a smooth Gaussian process on a line and
//       interpolated Gaussian white noise on a plane, both keyed on film
//       coordinates so that the same patch of film always draws the same values.
//
//  Header only, SHARED by the scalar and the AVX2 trees. Every computation is
//  HighPrecType in both builds and only the store into the AlgoType plane
//  rounds, exactly like AlgoProcessingDefects.hpp and the 9b stroke rasteriser.
//
//  ---- 1. THE READER LAW ----------------------------------------------------
//
//  Film attenuates light by ABSORPTION and by SCATTERING OUT OF THE READER'S
//  ACCEPTANCE CONE (Trumpy & Gschwind, «Optical Detection of Dust and Scratches
//  on Photographic Film», ACM JOCCH 8(2), Art. 7, 2015, Eq. 1, §4.1, Figs. 3-4).
//  A reader with specular fraction s (AlgoControls::scannerSpecular, 0 diffuse
//  ... 1 condenser) sees a defect whose absorbing density is Da and whose
//  scattering density is Ds as, in the transmittance-mixing form of
//  AlgoCallierNet (Silberstein & Tuttle):
//
//      D_read = Da - log10( (1 - s) + s * 10^-Ds )
//
//  The defect constants were calibrated on scans whose reader geometry is NOT
//  documented (doc/RESULT_2026-10-20_defect_optics.md §1). The law is therefore
//  ANCHORED at s_ref = ALGO_DEFECT_OPTICS_S_REF: at s = s_ref every defect keeps
//  exactly the density increment the OFF path gives it, and only the departure
//  of s from s_ref changes anything. A class's scattering share f splits that
//  increment: f * dD is the scattering part as read AT s_ref, and Ds is solved
//  from it:
//
//      10^-Ds = ( 10^-(f dD) - (1 - s_ref) ) / s_ref
//
//  The read scattering part cannot exceed -log10(1 - s_ref) (an infinitely
//  dense scatterer under that reader), so f dD is capped at
//  ALGO_DEFECT_SCATTER_REF_HEADROOM of that ceiling and the excess is carried
//  as absorption - which keeps the total at s_ref unchanged.
//
//  ---- THE CALLIER EXCLUSION ------------------------------------------------
//
//  Stage 12b applies the film's Callier law dmin + C(D - dmin) to the WHOLE
//  density of a stock with callier_q > 1, defects included - but Q is a property
//  of developed silver grains, not of dust. With the switch ON, an ADDED defect
//  is composed in the reader's density domain:
//
//      x_new = C^-1( C(x_old) + dRead )          x = D - dmin
//
//  so after stage 12b the pixel reads C(x_old) + dRead: the silver is steepened
//  and the defect is not. Overlapping defects telescope exactly. A REMOVAL (a
//  cut, a pinhole) takes silver away, x_new = x_old (1 - frac), and is left in
//  the silver domain, where stage 12b correctly steepens what remains.
//
//  ⚠ APPROXIMATION, STATED: stages 10 (scan MTF), 10b and 11 (grain) run between
//  9b and 12b. The blur is linear in density while C is not, so the exclusion is
//  exact on a defect wider than the scan PSF and approximate at its edge. That is
//  the price of needing no extra plane between 9b and 12b.
//
//  ---- 2. STOCHASTIC PRIMITIVES ---------------------------------------------
//
//  Both are built from INDEPENDENT standard-normal lattice samples
//  (AlgoRngNormal of a hash of the lattice index), interpolated, and divided by
//  the root of the sum of squared weights, so that the value at EVERY point is
//  exactly N(0, 1) - not merely on average - and no position in the lattice is
//  quieter than another. Keys are integer lattice indices in film millimetres,
//  so the values are a property of the film: no frame index enters, nothing
//  flickers, and nothing repeats (the hash has no period on any usable range).
// ---------------------------------------------------------------------------

// Project-wide primitives, included unconditionally as required by the project
// coding standard.
#include "Common.hpp"
#include "CompileTimeUtils.hpp"

// ImgType, AlgoType, HighPrecType.
#include "AlgoTypes.hpp"

// Counter-based generator and the lattice hash.
#include "AlgoCounterRng.hpp"
#include "AlgoDefectField.hpp"

// AlgoCallierNet, AlgoCallierInert.
#include "AlgoCallier.hpp"

// film::FilmProfile (callier_q, curve dmin).
#include "film_profiles.hpp"

#include <cmath>     // std::floor, std::sqrt, std::pow, std::log10
#include <cstdint>   // int32_t, uint32_t, uint64_t


// ===========================================================================
//  Context: everything the write needs, formed once per frame in stage 9b.
// ===========================================================================
struct AlgoDefectOpticsCtx
{
    bool         on;          // FilmDamage::defectOptics
    bool         callier;     // on, and stage 12b will act on this stock
    HighPrecType s;           // reader specular fraction, clamped to [0, 1]
    HighPrecType sRef;        // anchor of the law
    HighPrecType q[3];        // beta per reader channel (AlgoCallierQChannel, v67)
    HighPrecType dmin[3];     // curve D_min per record
    HighPrecType visibility;  // FilmDamage::defectVisibility (2026-10-20b)
    HighPrecType baseD[3];    // clear-base density per record (2026-10-20b)
};


// ---------------------------------------------------------------------------
//  AlgoDefectOpticsMake
//
//  sRef is passed in rather than read from AlgoNegativeDefects.hpp so this
//  header does not depend on the defect constants.
// ---------------------------------------------------------------------------
inline AlgoDefectOpticsCtx AlgoDefectOpticsMake
(
    const bool               on,
    const film::FilmProfile& profile,
    const HighPrecType       scannerSpecular,
    const HighPrecType       sRef,
    const HighPrecType       visibility,
    const HighPrecType       clearBaseD
) noexcept
{
    AlgoDefectOpticsCtx ctx{};

    ctx.visibility = MAX_VALUE(visibility, 0.0);

    ctx.on      = on;
    ctx.s       = CLAMP_VALUE(scannerSpecular, 0.0, 1.0);
    ctx.sRef    = CLAMP_VALUE(sRef, 1.0e-3, 1.0);
    ctx.q[0]    = AlgoCallierQChannel(profile, 0);
    ctx.q[1]    = AlgoCallierQChannel(profile, 1);
    ctx.q[2]    = AlgoCallierQChannel(profile, 2);
    ctx.callier = on && !AlgoCallierInert(profile, scannerSpecular);
    ctx.dmin[0] = static_cast<HighPrecType>(profile.curves.r.dmin);
    ctx.dmin[1] = static_cast<HighPrecType>(profile.curves.g.dmin);
    ctx.dmin[2] = static_cast<HighPrecType>(profile.curves.b.dmin);

    // The support alone: never denser than the record's own D_min.
    for (int32_t c = 0; c < 3; c++)
        ctx.baseD[c] = MIN_VALUE(MAX_VALUE(clearBaseD, 0.0), ctx.dmin[c]);

    return ctx;
}


// ---------------------------------------------------------------------------
//  AlgoDefectScatterRead
//
//  The density a reader of specular fraction s sees from a defect whose total
//  increment at s_ref is dInc and whose scattering share is `share`.
//  At s == sRef this returns dInc (to rounding); at s = 0 only the absorbing
//  part remains; at s = 1 the full scattering loss is seen.
// ---------------------------------------------------------------------------
inline HighPrecType AlgoDefectScatterRead
(
    const HighPrecType dInc,
    const HighPrecType share,
    const HighPrecType s,
    const HighPrecType sRef,
    const HighPrecType headroom
) noexcept
{
    if (dInc <= 0.0)
        return 0.0;

    const HighPrecType f = CLAMP_VALUE(share, 0.0, 1.0);

    // Ceiling of the read scattering part at s_ref, and the share of dInc that
    // goes to scattering, capped below it.
    const HighPrecType ceilRef = -std::log10(1.0 - MIN_VALUE(sRef, 0.999999));
    const HighPrecType dRefS   = MIN_VALUE(f * dInc, headroom * ceilRef);
    const HighPrecType dA      = dInc - dRefS;

    if (dRefS <= 0.0)
        return dA;

    // Invert the anchor for the scattering density Ds (kept as 10^-Ds).
    const HighPrecType tS = (std::pow(10.0, -dRefS) - (1.0 - sRef)) / sRef;

    // tS is in (0, 1] by the cap above; guarded anyway.
    const HighPrecType tSafe = CLAMP_VALUE(tS, 1.0e-12, 1.0);

    const HighPrecType tRead = (1.0 - s) + (s * tSafe);

    return dA - std::log10(tRead);
}


// ---------------------------------------------------------------------------
//  AlgoDefectCallierInverse
//
//  x such that AlgoCallierNet(x, q, s) == y. Newton on a strictly increasing,
//  smooth function whose slope lies between 1 and q, started from the caller's
//  hint. Six iterations take any start inside the stage's density range to
//  double rounding.
// ---------------------------------------------------------------------------
inline HighPrecType AlgoDefectCallierInverse
(
    const HighPrecType y,
    const HighPrecType xHint,
    const HighPrecType q,
    const HighPrecType s
) noexcept
{
    const HighPrecType e = 1.0 - s;

    HighPrecType x = xHint;

    for (int32_t it = 0; it < 6; it++)
    {
        const HighPrecType a = std::pow(10.0, -x);
        const HighPrecType b = std::pow(10.0, -(q * x));
        const HighPrecType t = (e * a) + ((1.0 - e) * b);

        if (t <= 1.0e-300)
            break;

        const HighPrecType f     = -std::log10(t);
        const HighPrecType slope = ((e * a) + ((1.0 - e) * q * b)) / t;

        if (slope <= 0.0)
            break;

        const HighPrecType dx = (f - y) / slope;

        x -= dx;

        if (std::fabs(dx) < 1.0e-13)
            break;
    }

    return x;
}


// ---------------------------------------------------------------------------
//  AlgoDefectComposeRead
//
//  Adds a READ density dRead to the absolute density dOld of one record.
//  OFF, or no Callier on this stock: plain addition. ON with Callier: composed
//  in the reader's domain, so stage 12b leaves the defect unsteepened.
// ---------------------------------------------------------------------------
inline HighPrecType AlgoDefectComposeRead
(
    const HighPrecType          dOld,
    const HighPrecType          dRead,
    const int32_t               channel,
    const AlgoDefectOpticsCtx&  ctx
) noexcept
{
    if (!ctx.callier || dRead == 0.0)
        return dOld + dRead;

    const HighPrecType dm   = ctx.dmin[channel];
    const HighPrecType q    = ctx.q[channel];   // 2026-10-26b: per channel
    const HighPrecType xOld = dOld - dm;
    const HighPrecType yOld = AlgoCallierNet(xOld, q, ctx.s);

    // Start from the first-order step: dRead over the local slope.
    const HighPrecType e  = 1.0 - ctx.s;
    const HighPrecType a  = std::pow(10.0, -xOld);
    const HighPrecType b  = std::pow(10.0, -(q * xOld));
    const HighPrecType t  = MAX_VALUE((e * a) + ((1.0 - e) * b), 1.0e-300);
    const HighPrecType sl = MAX_VALUE(((e * a) + ((1.0 - e) * q * b)) / t,
                                      1.0e-6);

    const HighPrecType xNew = AlgoDefectCallierInverse(
        yOld + dRead, xOld + (dRead / sl), q, ctx.s);

    return dm + xNew;
}


// ---------------------------------------------------------------------------
//  AlgoDefectRemove
//
//  Takes the fraction `frac` of the EMULSION away: emulsion cut out by a
//  scratch, or missing at a pinhole. `dmin` is the floor the removal works
//  towards; since 2026-10-20b the caller passes the CLEAR-BASE density, so fog
//  and - on a colour negative - the orange mask, which live in the emulsion,
//  go with it (a deep cut on a masked negative prints dark and coloured, and on
//  a slide it shows the bare support). Never adds.
// ---------------------------------------------------------------------------
inline HighPrecType AlgoDefectRemove
(
    const HighPrecType dOld,
    const HighPrecType frac,
    const HighPrecType dmin
) noexcept
{
    const HighPrecType net = dOld - dmin;

    if (net <= 0.0)
        return dOld;

    return dmin + (net * (1.0 - CLAMP_VALUE(frac, 0.0, 1.0)));
}


// ---------------------------------------------------------------------------
//  AlgoDefectWrite
//
//  One record of one pixel, from the defect's local opacity `a` (coverage x
//  peak opacity x chroma, already capped below 1).
//
//    polarity > 0   material ON or IN the film that blocks light (dust, debris,
//                   fibres, a burnished base): density ADDED.
//    polarity < 0   material REMOVED (a cut, a pinhole).
//
//  OFF (ctx.on false) is the pre-2026-10-20 law (exactly so at
//  defectVisibility 1; v scales the density change):
//      D <- max(D + polarity * v * (-log10(1 - a)), 0)
//
//  ON:
//      added     D <- Compose(D, ScatterRead(-log10(1 - a), share))
//      removed   D <- Remove(D, a), then the cut WALL, a scattering edge of
//                wallShare x the OFF increment, is added through the law with
//                scattering share wallScatter.
// ---------------------------------------------------------------------------
inline HighPrecType AlgoDefectWrite
(
    const HighPrecType          dOld,
    const HighPrecType          a,
    const HighPrecType          polarity,
    const HighPrecType          share,
    const HighPrecType          wallShare,
    const HighPrecType          wallScatter,
    const HighPrecType          headroom,
    const int32_t               channel,
    const AlgoDefectOpticsCtx&  ctx
) noexcept
{
    // defectVisibility scales the density change: opacity a -> 1 - (1 - a)^v.
    const HighPrecType dInc = -ctx.visibility * std::log10(1.0 - a);

    if (!ctx.on)
        return MAX_VALUE(dOld + (polarity * dInc), 0.0);

    if (polarity >= 0.0)
    {
        const HighPrecType dRead = AlgoDefectScatterRead(
            dInc, share, ctx.s, ctx.sRef, headroom);

        return MAX_VALUE(AlgoDefectComposeRead(dOld, dRead, channel, ctx), 0.0);
    }

    // Exact at visibility 1 (no pow round trip).
    const HighPrecType aVis = (ctx.visibility == 1.0)
                            ? a : (1.0 - std::pow(1.0 - a, ctx.visibility));

    const HighPrecType dCut = AlgoDefectRemove(dOld, aVis, ctx.baseD[channel]);

    const HighPrecType dWall = AlgoDefectScatterRead(
        wallShare * dInc, wallScatter, ctx.s, ctx.sRef, headroom);

    return MAX_VALUE(AlgoDefectComposeRead(dCut, dWall, channel, ctx), 0.0);
}


// ===========================================================================
//  Stochastic primitives
// ===========================================================================

// ---------------------------------------------------------------------------
//  AlgoDefectSubSeed
//
//  Folds a 64-bit draw counter into a 32-bit hash seed, so a particle's or a
//  scratch's own stream can key a lattice of its own.
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
//  AlgoDefectLatticeKey
//
//  A lattice counter for a 64-bit index. Film coordinates divided by an 8 um
//  step exceed int32 after about ten hours of 35 mm film (frame numbers from
//  a timecode reach that), so the index is carried as int64 and its high word
//  is folded into the seed before the 32-bit hash. Identical to the plain
//  hash for every index that fits in int32 with a zero high word.
// ---------------------------------------------------------------------------
FORCE_INLINE uint64_t AlgoDefectLatticeKey
(
    const uint32_t seed,
    const int64_t  index,
    const int32_t  stream,
    const uint32_t tag
) noexcept
{
    const uint64_t u  = static_cast<uint64_t>(index);
    const uint32_t hi = static_cast<uint32_t>(u >> 32);
    const uint32_t lo = static_cast<uint32_t>(u & 0xFFFFFFFFull);

    // hi is 0 for small non-negative indices and 0xFFFFFFFF for small negative
    // ones; both fold to a fixed seed change, so short ranges stay continuous.
    return AlgoDefectHash(seed ^ (hi * 0x9E3779B1u), static_cast<int32_t>(lo),
                          stream, tag);
}


FORCE_INLINE uint32_t AlgoDefectSubSeed (const uint64_t counter) noexcept
{
    const uint64_t m = AlgoRngMix64(counter ^ 0xA5A5A5A5DEFEC701ull);

    return static_cast<uint32_t>(m ^ (m >> 32));
}


// ---------------------------------------------------------------------------
//  AlgoDefectSmoothNoise
//
//  A smooth, unit-variance Gaussian process along one coordinate, in film
//  millimetres. Independent N(0,1) samples on a lattice of spacing corrMm,
//  joined by a smoothstep blend and renormalised point by point, so its
//  correlation falls to zero at one lattice step and its derivative is
//  continuous. Keyed by (seed, stream, lattice index): the same coordinate of
//  the same film always gives the same value.
// ---------------------------------------------------------------------------
inline HighPrecType AlgoDefectSmoothNoise
(
    const uint32_t     seed,
    const int32_t      stream,
    const uint32_t     tag,
    const HighPrecType coordMm,
    const HighPrecType corrMm
) noexcept
{
    const HighPrecType c  = coordMm / MAX_VALUE(corrMm, 1.0e-6);
    const HighPrecType fl = std::floor(c);
    const int64_t      i  = static_cast<int64_t>(fl);

    const HighPrecType t  = c - fl;
    const HighPrecType w1 = t * t * (3.0 - (2.0 * t));
    const HighPrecType w0 = 1.0 - w1;

    const HighPrecType g0 = AlgoRngNormal(AlgoDefectLatticeKey(seed, i,     stream, tag));
    const HighPrecType g1 = AlgoRngNormal(AlgoDefectLatticeKey(seed, i + 1, stream, tag));

    return ((w0 * g0) + (w1 * g1)) / std::sqrt((w0 * w0) + (w1 * w1));
}


// ---------------------------------------------------------------------------
//  AlgoDefectWhiteNoise1D
//
//  Gaussian WHITE noise along one coordinate: independent N(0,1) samples every
//  stepMm, linearly interpolated and renormalised to unit variance at every
//  point. Used for the short-scale texture along a scratch, keyed on the
//  absolute along-film coordinate for a transport scratch.
// ---------------------------------------------------------------------------
inline HighPrecType AlgoDefectWhiteNoise1D
(
    const uint32_t     seed,
    const int32_t      stream,
    const uint32_t     tag,
    const HighPrecType coordMm,
    const HighPrecType stepMm
) noexcept
{
    const HighPrecType c  = coordMm / MAX_VALUE(stepMm, 1.0e-6);
    const HighPrecType fl = std::floor(c);
    const int64_t      i  = static_cast<int64_t>(fl);

    const HighPrecType w1 = c - fl;
    const HighPrecType w0 = 1.0 - w1;

    const HighPrecType g0 = AlgoRngNormal(AlgoDefectLatticeKey(seed, i,     stream, tag));
    const HighPrecType g1 = AlgoRngNormal(AlgoDefectLatticeKey(seed, i + 1, stream, tag));

    return ((w0 * g0) + (w1 * g1)) / std::sqrt((w0 * w0) + (w1 * w1));
}


// ---------------------------------------------------------------------------
//  AlgoDefectWhiteNoise2D
//
//  Gaussian WHITE noise on a plane: independent N(0,1) samples on a square
//  lattice of unit step (the caller scales its coordinates), bilinearly
//  interpolated and renormalised to unit variance at every point. Used for the
//  interior texture of dust and debris, in each particle's own frame so the
//  texture travels with the particle.
// ---------------------------------------------------------------------------
inline HighPrecType AlgoDefectWhiteNoise2D
(
    const uint32_t     seed,
    const uint32_t     tag,
    const HighPrecType u,
    const HighPrecType v
) noexcept
{
    const HighPrecType fu = std::floor(u);
    const HighPrecType fv = std::floor(v);
    const int32_t      iu = static_cast<int32_t>(fu);
    const int32_t      iv = static_cast<int32_t>(fv);

    const HighPrecType tu = u - fu;
    const HighPrecType tv = v - fv;

    const HighPrecType w00 = (1.0 - tu) * (1.0 - tv);
    const HighPrecType w10 = tu * (1.0 - tv);
    const HighPrecType w01 = (1.0 - tu) * tv;
    const HighPrecType w11 = tu * tv;

    // Separate rows by mixing the second index into the tag-side argument.
    const HighPrecType g00 = AlgoRngNormal(AlgoDefectHash(seed, iu,     iv,     tag));
    const HighPrecType g10 = AlgoRngNormal(AlgoDefectHash(seed, iu + 1, iv,     tag));
    const HighPrecType g01 = AlgoRngNormal(AlgoDefectHash(seed, iu,     iv + 1, tag));
    const HighPrecType g11 = AlgoRngNormal(AlgoDefectHash(seed, iu + 1, iv + 1, tag));

    const HighPrecType norm = std::sqrt((w00 * w00) + (w10 * w10)
                                      + (w01 * w01) + (w11 * w11));

    return ((w00 * g00) + (w10 * g10) + (w01 * g01) + (w11 * g11)) / norm;
}


// ---------------------------------------------------------------------------
//  AlgoDefectLogNormalTrunc
//
//  Log-normal with an explicit median, truncated to [lo, hi] by inverse-CDF
//  REMAPPING rather than clamping, so no probability piles up at the bounds.
//  The normal quantile is computed with Acklam's rational approximation
//  (relative error below 1.2e-9), which is a pure function of the uniform.
// ---------------------------------------------------------------------------
inline HighPrecType AlgoDefectNormalQuantile (const HighPrecType pIn) noexcept
{
    const HighPrecType p = CLAMP_VALUE(pIn, 1.0e-15, 1.0 - 1.0e-15);

    static const HighPrecType a[6] = { -3.969683028665376e+01,  2.209460984245205e+02,
                                       -2.759285104469687e+02,  1.383577518672690e+02,
                                       -3.066479806614716e+01,  2.506628277459239e+00 };
    static const HighPrecType b[5] = { -5.447609879822406e+01,  1.615858368580409e+02,
                                       -1.556989798598866e+02,  6.680131188771972e+01,
                                       -1.328068155288572e+01 };
    static const HighPrecType c[6] = { -7.784894002430293e-03, -3.223964580411365e-01,
                                       -2.400758277161838e+00, -2.549732539343734e+00,
                                        4.374664141464968e+00,  2.938163982698783e+00 };
    static const HighPrecType d[4] = {  7.784695709041462e-03,  3.224671290700398e-01,
                                        2.445134137142996e+00,  3.754408661907416e+00 };

    const HighPrecType pLow = 0.02425;

    if (p < pLow)
    {
        const HighPrecType qq = std::sqrt(-2.0 * std::log(p));
        return (((((c[0] * qq + c[1]) * qq + c[2]) * qq + c[3]) * qq + c[4]) * qq + c[5])
             / ((((d[0] * qq + d[1]) * qq + d[2]) * qq + d[3]) * qq + 1.0);
    }

    if (p > (1.0 - pLow))
    {
        const HighPrecType qq = std::sqrt(-2.0 * std::log(1.0 - p));
        return -(((((c[0] * qq + c[1]) * qq + c[2]) * qq + c[3]) * qq + c[4]) * qq + c[5])
              / ((((d[0] * qq + d[1]) * qq + d[2]) * qq + d[3]) * qq + 1.0);
    }

    const HighPrecType qq = p - 0.5;
    const HighPrecType r  = qq * qq;

    return (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * qq
         / (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1.0);
}

inline HighPrecType AlgoDefectNormalCdf (const HighPrecType z) noexcept
{
    return 0.5 * std::erfc(-z / 1.4142135623730950488);
}

inline HighPrecType AlgoDefectLogNormalTrunc
(
    const uint64_t     counter,
    const HighPrecType median,
    const HighPrecType sigmaLn,
    const HighPrecType lo,
    const HighPrecType hi
) noexcept
{
    const HighPrecType zLo = std::log(lo / median) / sigmaLn;
    const HighPrecType zHi = std::log(hi / median) / sigmaLn;

    const HighPrecType pLo = AlgoDefectNormalCdf(zLo);
    const HighPrecType pHi = AlgoDefectNormalCdf(zHi);

    const HighPrecType p = pLo + ((pHi - pLo) * AlgoRngUniform01(counter));

    const HighPrecType v = median * std::exp(sigmaLn * AlgoDefectNormalQuantile(p));

    return CLAMP_VALUE(v, lo, hi);
}
