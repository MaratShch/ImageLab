#pragma once

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
//  at specular = 1, mid grey moved +54/255 before the solve was taught about it,
//  against a contrast change of 22 %. One of those is the physics; the other is
//  the laboratory failing to do its job. So the factor below is consumed in
//  THREE places, and they have to agree:
//
//      AlgoSolveAnchors        the print offset that lands mid grey
//      AlgoNeutralMidDensity   the print chain's own mid-grey reference
//      AlgoStage12_DyeImpurity the per-pixel pass
//
//  INERT BY DEFAULT. `scannerSpecular` is 0.0, the factor is exactly 1.0, and
//  every render made before this file existed is reproduced bit for bit.
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
inline HighPrecType AlgoCallierFactor
(
    const film::FilmProfile& profile,
    const HighPrecType       scannerSpecular
) noexcept
{
    const HighPrecType q = static_cast<HighPrecType>(profile.callier_q);

    if ((scannerSpecular <= 0.0) || (q == 1.0))
        return 1.0;

    return 1.0 + (scannerSpecular * (q - 1.0));
}


// ---------------------------------------------------------------------------
//  AlgoCallierApplyScalar
//
//  One density, one channel's dmin. Used by the two solvers, which work in
//  HighPrecType on three scalars rather than on planes.
// ---------------------------------------------------------------------------
inline HighPrecType AlgoCallierApplyScalar
(
    const HighPrecType d,
    const HighPrecType dmin,
    const HighPrecType factor
) noexcept
{
    if (1.0 == factor)
        return d;

    return dmin + ((d - dmin) * factor);
}
