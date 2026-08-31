#pragma once

#include <cmath>

// ---------------------------------------------------------------------------
//  AlgoCallier.hpp
//
//  Callier's coefficient: the density a DIRECTIONAL reader sees, as opposed to
//  the diffuse density every curve in the database is expressed in.
//
//  WHAT IT IS
//  ----------
//  Developed silver grains scatter the measuring beam. An integrating sphere
//  collects the scattered light and reads the diffuse density; a condenser or a
//  point source loses it outside its acceptance angle and reads a HIGHER
//  density. The ratio of the two is Callier's Q, and it steepens the whole tone
//  scale by that factor - which is exactly why a silver negative printed on a
//  condenser enlarger is contrastier than the same negative on a diffusion
//  enlarger, at the same paper grade.
//
//  A chromogenic DYE image scatters almost nothing, so Q is 1.0 for every colour
//  stock in the file and this code is inert on them at any setting. It moves the
//  66 monochrome profiles and nothing else.
//
//  WHY THE FIELD ALONE WAS THE WRONG SHAPE (queue C22)
//  --------------------------------------------------
//  `FilmProfile::callier_q` modelled Q as a property of the FILM. It is a
//  property of film x MEASURING GEOMETRY: the same negative reads differently on
//  a diffuse LED integrating-sphere scanner and on a directed halogen condenser.
//  So the film contributes its scattering (Q, stored per stock) and the reader
//  contributes how directional it is (`AlgoControls::scannerSpecular`), and
//  neither number answers the question by itself.
//
//      D_read = dmin + (D - dmin) * (1 + specular * (Q - 1))
//
//  ⚠ REFERENCED TO dmin, NOT TO ZERO, and that is physics rather than
//  convenience: the scattering scales with the amount of developed silver, so
//  clear base carries none of it. Scaling absolute density would make a
//  condenser darken the film base, which no densitometer measures.
//
//  ⚠ AND IT MUST BE VISIBLE TO THE ANCHOR SOLVE, WHICH IS THE PART THAT IS EASY
//  TO GET WRONG. A lab that switches to a condenser head RE-TIMES the print;
//  that is what printer lights are for. If only the pixel pass sees Callier and
//  the solve does not, the render both steepens the tone scale AND shifts mid
//  grey - and the shift is the bigger of the two: measured on EASTMAN DOUBLE-X
//  at specular = 1, mid grey moved by roughly a fifth of the output range before
//  the solve was taught about it, against a contrast change of a few per cent.
//  One of those is the physics; the other is the laboratory failing to do its
//  job.
//
//  ⚠ TWO RECORDS OF THAT ONE MEASUREMENT DISAGREE, WHICH IS WHY NO EXACT FIGURE
//  IS QUOTED HERE ANY MORE. This header said "+54/255 ... against a contrast
//  change of 22 %"; film_sim.py's own call site says "+48/255 ... against a
//  contrast change of a few per cent". One experiment, two write-ups, and
//  nothing in the repository settles which transcription is right. What the
//  argument actually rests on is the ORDER of the two effects, and both records
//  agree on that, so that is what is stated. Re-measure before quoting a number.
//
//  ⚠ THE CONSUMER LIST HERE WAS WRONG ON TWO OF ITS THREE ENTRIES UNTIL
//  2026-08-30, AND WRONG IN THE DIRECTION THAT COSTS WORK: it named call sites
//  that do not exist, so anyone wiring this up from the header would have
//  CREATED divergences instead of closing them.
//
//      AlgoSolveAnchors         ✅ correct. film_sim applies the same factor at
//                               two points inside the solve -- once on the
//                               reversal branch's `mixed`, once on the negative
//                               branch's `d_mid`.
//      AlgoNeutralMidDensity    ⚠ WRONG, and wiring it would CREATE a
//                               divergence. film_sim.neutral_mid_density()
//                               takes no `scanner_specular` argument and applies
//                               nothing at all. Deliberately left alone.
//      AlgoStage12_DyeImpurity  ⚠ WRONG about the location. Callier is its own
//                               STAGE 12b, run after stage 12 and before stage
//                               13, in place -- not folded into the dye matrix.
//                               That boundary is chosen because BOTH readers in
//                               the chain are affected: an optical printer with
//                               a condenser and a scanner with a directed source
//                               see the same steepened density, and the print
//                               stage's own curve must act on what its optics
//                               actually see.
//
//  So the factor is consumed in TWO places, and they have to agree:
//
//      AlgoSolveAnchors         the print offset that lands mid grey
//      AlgoStage12b_Callier     the per-pixel pass
//
//  INERT BY DEFAULT. `scannerSpecular` is 0.0, the factor is exactly 1.0, both
//  consumers return early, and every render made before this was wired is
//  reproduced bit for bit.
//
//  ⚠ THE FILM HALF OF THE PRODUCT IS A CLASS ESTIMATE. The two monochrome values
//  (1.3 negative, 1.25 reversal) come from a class rule in the generator, not
//  from any document in the corpus; the geometry half is exact. That asymmetry is
//  why the control ships at zero rather than at some "typical scanner" value.
//  What would fix it is one densitometer specification stating a
//  diffuse-versus-specular ratio for a named emulsion.
//
//  ONE LAW, TWO LANGUAGES. film_sim._callier_factor() / callier_density() is the
//  reference.
// ---------------------------------------------------------------------------

// Project-wide primitives, included unconditionally as required by the project
// coding standard.
#include "Common.hpp"
#include "CompileTimeUtils.hpp"

// ImgType, AlgoType, HighPrecType. The single place numeric types are chosen.
#include "AlgoTypes.hpp"

// film::FilmProfile.
#include "film_profiles.hpp"


// ---------------------------------------------------------------------------
//  AlgoCallierFactor
//
//  The multiplier a reader of the stated directionality applies to density
//  above dmin. Exactly 1.0 when the reader is diffuse or the image does not
//  scatter, which is the case that must stay bit-identical.
// ---------------------------------------------------------------------------
inline HighPrecType AlgoCallierNet
(
    const HighPrecType dNet,
    const HighPrecType q,
    const HighPrecType scannerSpecular
) noexcept
{
    //  Silberstein & Tuttle, Mees printed page 644:
    //
    //      10^-D_sp = E * 10^-D_diff + (1 - E) * 10^-(beta * D_diff)
    //
    //  with E the fraction of scattered light the reader ACCEPTS, so
    //  E = 1 - scannerSpecular, and beta the film's scattering-to-absorption
    //  ratio, which is callier_q. The book states both limits outright: E = 0
    //  gives Q exactly, E = 1 gives the diffuse density unchanged.
    //
    //  ⚠ EXACT AT BOTH ENDS BY CONSTRUCTION, and that is the whole inertness
    //  contract. At scannerSpecular = 0 this returns dNet bit-for-bit, so every
    //  render made before the stage existed is reproduced.
    //
    //  ⚠ THE ARGUMENT IS NET DENSITY AND MAY BE NEGATIVE. Grain pushes pixels
    //  below dmin; the law is smooth there and needs no branch, which is what
    //  keeps the base from acquiring a second code path.
    const HighPrecType e = 1.0 - scannerSpecular;
    const HighPrecType t = (e * std::pow(10.0, -dNet))
                         + ((1.0 - e) * std::pow(10.0, -(q * dNet)));

    return -std::log10((t > 1e-300) ? t : 1e-300);
}


// ---------------------------------------------------------------------------
//  AlgoCallierInert
//
//  The identity test, in one place. specular 0 is the shipped default and
//  Q = 1.0 is every colour stock: both must leave planes untouched.
// ---------------------------------------------------------------------------
inline bool AlgoCallierInert
(
    const film::FilmProfile& profile,
    const HighPrecType       scannerSpecular
) noexcept
{
    return (scannerSpecular <= 0.0)
        || (static_cast<HighPrecType>(profile.callier_q) == 1.0);
}


// ---------------------------------------------------------------------------
//  AlgoCallierLut
//
//  ⚠ THE TABLE IS PART OF THE LAW, NOT AN OPTIMISATION OF IT. AlgoCallierNet
//  costs two pow() and a log10() per channel per pixel, and neither has an AVX2
//  intrinsic. Evaluating it directly would either drop the AVX2 twin to scalar
//  for this stage or compute the law DIFFERENTLY in the two flavours, and a law
//  that differs between twins is precisely the defect cpp_parity's twin check
//  exists to catch. So both flavours -- and film_sim.py -- build the SAME table,
//  with the same bounds, the same count and the same interpolation. Parity holds
//  by construction rather than by tolerance; measured interpolation error
//  against the exact law is 2.2e-07 over the whole span.
//
//  Outside the span the end slopes are used: below the floor the curve is
//  smooth and near-linear, above the ceiling it is asymptotically
//  dNet - log10(E), slope exactly 1.
// ---------------------------------------------------------------------------
#define ALGO_CALLIER_LUT_MIN  (-1.0)
#define ALGO_CALLIER_LUT_MAX  ( 5.0)
#define ALGO_CALLIER_LUT_N    (1025)

struct AlgoCallierLut
{
    bool         active;
    HighPrecType lo;
    HighPrecType step;
    HighPrecType invStep;
    HighPrecType hi;
    AlgoType     slopeLo;
    AlgoType     slopeHi;
    AlgoType     v[ALGO_CALLIER_LUT_N];
};


inline void AlgoCallierLutBuild
(
    AlgoCallierLut&          lut,
    const film::FilmProfile& profile,
    const HighPrecType       scannerSpecular
) noexcept
{
    lut.active = !AlgoCallierInert(profile, scannerSpecular);

    if (!lut.active)
        return;

    const HighPrecType q = static_cast<HighPrecType>(profile.callier_q);

    lut.lo      = ALGO_CALLIER_LUT_MIN;
    lut.step    = (ALGO_CALLIER_LUT_MAX - ALGO_CALLIER_LUT_MIN)
                / static_cast<HighPrecType>(ALGO_CALLIER_LUT_N - 1);
    lut.invStep = 1.0 / lut.step;
    lut.hi      = ALGO_CALLIER_LUT_MAX;

    for (int32_t i = 0; i < ALGO_CALLIER_LUT_N; i++)
    {
        const HighPrecType x = lut.lo + (lut.step * static_cast<HighPrecType>(i));
        lut.v[i] = static_cast<AlgoType>(AlgoCallierNet(x, q, scannerSpecular));
    }

    lut.slopeLo = static_cast<AlgoType>(
        (lut.v[1] - lut.v[0]) * static_cast<AlgoType>(lut.invStep));
    lut.slopeHi = static_cast<AlgoType>(
        (lut.v[ALGO_CALLIER_LUT_N - 1] - lut.v[ALGO_CALLIER_LUT_N - 2])
        * static_cast<AlgoType>(lut.invStep));

    return;
}


inline AlgoType AlgoCallierLutAt
(
    const AlgoCallierLut& lut,
    const AlgoType        dNet
) noexcept
{
    const AlgoType t = (dNet - static_cast<AlgoType>(lut.lo))
                     * static_cast<AlgoType>(lut.invStep);

    if (t <= static_cast<AlgoType>(0))
        return lut.v[0] + ((dNet - static_cast<AlgoType>(lut.lo)) * lut.slopeLo);

    if (t >= static_cast<AlgoType>(ALGO_CALLIER_LUT_N - 1))
        return lut.v[ALGO_CALLIER_LUT_N - 1]
             + ((dNet - static_cast<AlgoType>(lut.hi)) * lut.slopeHi);

    const int32_t  i    = static_cast<int32_t>(t);
    const AlgoType frac = t - static_cast<AlgoType>(i);

    return lut.v[i] + ((lut.v[i + 1] - lut.v[i]) * frac);
}


//  AlgoCallierApplyScalar
//
//  One density, one channel's dmin. Used by the two solvers, which work in
//  HighPrecType on three scalars rather than on planes.
// ---------------------------------------------------------------------------
inline HighPrecType AlgoCallierApplyScalar
(
    const HighPrecType d,
    const HighPrecType dmin,
    const HighPrecType q,
    const HighPrecType scannerSpecular
) noexcept
{
    //  ⚠ THE SOLVE EVALUATES THE LAW EXACTLY WHILE THE PIXEL PASS USES THE
    //  TABLE, AND THAT ASYMMETRY IS DELIBERATE. This is called on a handful of
    //  scalars per solve iteration, where two pow() cost nothing and an exact
    //  answer is worth having. The table exists for the millions of pixels in
    //  the stage, and is built from this same function, so the two cannot
    //  disagree about the LAW -- only by the table's interpolation error, which
    //  cpp_parity measures rather than assumes.
    if ((scannerSpecular <= 0.0) || (q == 1.0))
        return d;

    return dmin + AlgoCallierNet(d - dmin, q, scannerSpecular);
}


// ---------------------------------------------------------------------------
//  AlgoStage12b_Callier
//
//  The per-pixel pass. In place on three density planes, between stage 12 and
//  stage 13.
//
//  ⚠ IN PLACE, AND WITH NO SCRATCH PLANE, WHICH IS WHY THIS SLOTS IN WITHOUT
//  TOUCHING THE ARENA. The operation is pointwise and idempotent in shape --
//  d <- dmin + (d - dmin) * factor -- so the stage-12 output planes can be
//  rewritten where they lie and stage 13 reads the corrected values from the
//  same pointers. Adding a stage that needed its own buffer would have meant
//  re-costing AlgoMemHandler for a control that ships at zero.
//
//  ⚠ ONE IMPLEMENTATION FOR BOTH FLAVOURS, DELIBERATELY, AND NOT AN OVERSIGHT
//  OF THE TWIN RULE. AlgorithmMain.cpp is itself shared between the two builds,
//  so this inline is compiled once per flavour with that flavour's AlgoType and
//  its own -mavx2 flags. The body is a branchless multiply-add over contiguous
//  RESTRICT-qualified planes -- the one shape a compiler vectorises reliably --
//  so hand-written intrinsics would duplicate a two-operation law to buy
//  nothing, and duplicating a law is exactly what cpp_parity's twin check
//  exists to prevent.
//
//  ⚠ THE CONDITION THAT NOTE SET HAS NOW BEEN MET, AND THE ANSWER IS STILL
//  ONE IMPLEMENTATION -- for a different reason, so read this rather than
//  the sentence it replaces. It said: \"if this ever grows a branch or a
//  table it MUST be split into twins like Algo_11\". Queue M3 gave it BOTH.
//  What makes one body still correct is that the branch and the table are
//  now the LAW ITSELF (AlgoCallierLutAt), shared by the solve, both
//  flavours and film_sim.py. Splitting into twins here would create two
//  spellings of one law, which is the thing the original note was protecting
//  against. The cost is real and is accepted: the per-pixel lookup does not
//  auto-vectorise, so at a non-zero setting this stage is scalar in both
//  builds. It ships at zero, where it returns before touching a pixel.
//
//  ⚠ RETURNS IMMEDIATELY WHEN THE FACTOR IS EXACTLY 1.0, which is the case for
//  every colour stock at any setting and for every stock at the default. That
//  early exit is not an optimisation, it is the bit-identity contract: an
//  untouched plane cannot differ from what the previous pipeline produced.
//
//  pDstR/G/B     density planes, modified in place
//  dmin          per-channel base plus fog of the curves that produced them
//  profile       supplies callier_q; a dye image scatters nothing and carries
//                Q = 1.0, so all 93 colour stocks are inert here
//  scannerSpecular  0 fully diffuse, 1 fully specular
// ---------------------------------------------------------------------------
inline void AlgoStage12b_Callier
(
    AlgoType* RESTRICT       pDstR,
    AlgoType* RESTRICT       pDstG,
    AlgoType* RESTRICT       pDstB,
    const int32_t            sizeX,
    const int32_t            sizeY,
    const int32_t            pitch,
    const AlgoType           dmin[3],
    const film::FilmProfile& profile,
    const HighPrecType       scannerSpecular
) noexcept
{
    AlgoCallierLut lut;

    AlgoCallierLutBuild(lut, profile, scannerSpecular);

    if (!lut.active)
        return;

    AlgoType* RESTRICT plane[3] = { pDstR, pDstG, pDstB };

    for (int32_t c = 0; c < 3; c++)
    {
        AlgoType* RESTRICT pD = plane[c];

        // The base carries no developed silver and therefore no scattering, so
        // the law is referenced to dmin and clear base is left where it is.
        const AlgoType dm = dmin[c];

        for (int32_t y = 0; y < sizeY; y++)
        {
            AlgoType* RESTRICT rD = pD + static_cast<std::ptrdiff_t>(y) * pitch;

            for (int32_t x = 0; x < sizeX; x++)
                rD[x] = dm + AlgoCallierLutAt(lut, rD[x] - dm);
        }
    }

    return;
}
