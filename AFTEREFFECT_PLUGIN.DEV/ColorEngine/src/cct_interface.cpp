// cct_interface.cpp
//
// Implementation of the CctHandle<T> public API over the compile-time
// Planckian-locus tables. See cct_interface.hpp for the design notes and
// cct_refine.cpp for the solver math.
//
// There is deliberately NO initialization code in this file: the locus
// tables are generated constexpr arrays living in .rodata. The former
// lazy-init machinery (static ready flags + mutex + per-instance vectors)
// is gone, and with it the multi-instance lifetime bug and the
// "getPlanckianUV before first ComputeCct" ordering hazard.

#include "cct_interface.hpp"
#include <algorithm>
#include <cmath>

using namespace AlgoCCT;

// -----------------------------------------------------------------------------
// cct_compute - thin adapter between the public entry point and the solver.
// Zeroes the outputs and runs the Ohno triangular+parabolic solver on the
// given chromaticity against the given locus table.
// The previous binary pre-search was removed long ago (its result was
// discarded; distance-to-locus is only piecewise monotonic and plateaus at
// high CCT could stall a bisection; its fallback conflated "not found" with
// "index 0"). refine() performs its own robust argmin.
// -----------------------------------------------------------------------------
template <typename T, typename std::enable_if<std::is_floating_point<T>::value>::type* E>
bool CctHandle<T, E>::cct_compute(const double& u, const double& v, double& cct, double& duv,
                                  const LutRow* lut, std::size_t n)
{
    cct = 0.0;
    duv = 0.0;
    return refine(u, v, cct, duv, lut, n);
}

// -----------------------------------------------------------------------------
// ComputeCct - PUBLIC ENTRY POINT.
// Input chain (see the super-pixel bridge): linear working-space RGB
//   --(3x3 matrix)--> XYZ --> u = 4X/(X+15Y+3Z), v = 6Y/(X+15Y+3Z).
// The coefficients 4/15/3 and 6 are the CIE 1960 UCS projection constants
// (MacAdam); they are what makes distances here match the CIE definition
// of CCT. Observer selects the table (the locus differs per observer
// because the color matching functions differ).
// -----------------------------------------------------------------------------
template <typename T, typename std::enable_if<std::is_floating_point<T>::value>::type* E>
std::pair<T, T> CctHandle<T, E>::ComputeCct (const std::pair<T, T>& uv, eCOLOR_OBSERVER observer)
{
    double Cct = 0.0;
    double Duv = 0.0;

    // Promote API scalars to the double core exactly once, at the boundary.
    const double u = static_cast<double>(uv.first);
    const double v = static_cast<double>(uv.second);

    if (observer_CIE_1931 == observer)
        cct_compute(u, v, Cct, Duv, m_lut1, m_size1);
    else
        cct_compute(u, v, Cct, Duv, m_lut2, m_size2);

    // Demote to the API scalar exactly once, at the boundary.
    return std::make_pair (static_cast<T>(Cct), static_cast<T>(Duv));
}

// -----------------------------------------------------------------------------
// getPlanckianUV - inverse mapping (CCT, Duv) -> (u, v).
//
// Step-agnostic: entries are only assumed SORTED by ascending cct (any
// spacing, uniform or not): binary search locates the bracketing segment,
// the interpolation factor is normalized by the ACTUAL segment length, and
// the tangent is a central difference over the neighbors - no table-step
// constant appears anywhere.
//
// The perpendicular is oriented so POSITIVE Duv is ABOVE the locus (toward
// green), per the CIE convention - matching the sign refine() produces
// (verified against the colour-science reference: D65 -> +0.0032).
// -----------------------------------------------------------------------------
template <typename T, typename std::enable_if<std::is_floating_point<T>::value>::type* E>
std::pair<T, T> CctHandle<T, E>::getPlanckianUV (T cct, T Duv, eCOLOR_OBSERVER observer)
{
    const LutRow*     lut = (observer_CIE_1931 == observer) ? m_lut1  : m_lut2;
    const std::size_t n   = (observer_CIE_1931 == observer) ? m_size1 : m_size2;
    if (n < 2u)
        return std::make_pair(static_cast<T>(0), static_cast<T>(0));

    const double Tk = static_cast<double>(cct);   // requested temperature [K]

    // Out-of-range: clamp to the locus endpoints.
    if (Tk <= lut[0].cct)
        return std::make_pair(static_cast<T>(lut[0].u), static_cast<T>(lut[0].v));
    if (Tk >= lut[n - 1u].cct)
        return std::make_pair(static_cast<T>(lut[n - 1u].u), static_cast<T>(lut[n - 1u].v));

    // Binary search: first entry with entry.cct > Tk.
    const LutRow* it = std::upper_bound(lut, lut + n, Tk,
        [](double value, const LutRow& e) { return value < e.cct; });
    const std::size_t hi = static_cast<std::size_t>(it - lut);
    const std::size_t lo = hi - 1u;

    const LutRow& L = lut[lo];
    const LutRow& H = lut[hi];

    // Normalized interpolation factor over the ACTUAL segment length.
    const double span = H.cct - L.cct;
    const double t = (span > 0.0) ? (Tk - L.cct) / span : 0.0;

    // Planckian point at Tk.
    const double u0 = L.u + t * (H.u - L.u);
    const double v0 = L.v + t * (H.v - L.v);

    // Tangent from the surrounding entries (central difference where
    // possible), independent of table spacing.
    const std::size_t im = (lo > 0u) ? (lo - 1u) : lo;
    const std::size_t ip = (hi + 1u < n) ? (hi + 1u) : hi;
    double du = lut[ip].u - lut[im].u;
    double dv = lut[ip].v - lut[im].v;
    const double len = std::sqrt(du * du + dv * dv);
    if (len > 0.0) { du /= len; dv /= len; }

    // Perpendicular: POSITIVE Duv above the locus (green), CIE convention.
    const double perp_u =  dv;
    const double perp_v = -du;

    return std::make_pair(static_cast<T>(u0 + static_cast<double>(Duv) * perp_u),
                          static_cast<T>(v0 + static_cast<double>(Duv) * perp_v));
}

template <typename T, typename std::enable_if<std::is_floating_point<T>::value>::type* E>
std::pair<T, T> CctHandle<T, E>::getPlanckianUV (const std::pair<T, T>& cct_Duv, eCOLOR_OBSERVER observer)
{
    return getPlanckianUV(cct_Duv.first, cct_Duv.second, observer);
}


// =============================================================================
// Explicit instantiation - PER MEMBER, only the members DEFINED in this TU.
//
// A whole-class instantiation (template class CctHandle<float>;) is avoided
// on purpose: it would also force refine(), whose definition lives in
// cct_refine.cpp, and MSVC then emits warning C4661 ("no suitable definition
// provided for explicit template instantiation request") because it cannot
// see that body from this translation unit. (GCC/Clang do not warn, which is
// why a Linux build is clean while the MSVC build is not.) Instead each
// member is explicitly instantiated in the TU that DEFINES it - the members
// below here, refine() in cct_refine.cpp - so every symbol is instantiated
// exactly once, in-view, warning-free. The ONLY two supported API scalar
// types are float and double.
// =============================================================================

// constructor
template AlgoCCT::CctHandle<float >::CctHandle();
template AlgoCCT::CctHandle<double>::CctHandle();

// ComputeCct
template std::pair<float , float > AlgoCCT::CctHandle<float >::ComputeCct (const std::pair<float , float >&, eCOLOR_OBSERVER);
template std::pair<double, double> AlgoCCT::CctHandle<double>::ComputeCct (const std::pair<double, double>&, eCOLOR_OBSERVER);

// getPlanckianUV (scalar overload)
template std::pair<float , float > AlgoCCT::CctHandle<float >::getPlanckianUV (float , float , eCOLOR_OBSERVER);
template std::pair<double, double> AlgoCCT::CctHandle<double>::getPlanckianUV (double, double, eCOLOR_OBSERVER);

// getPlanckianUV (pair overload)
template std::pair<float , float > AlgoCCT::CctHandle<float >::getPlanckianUV (const std::pair<float , float >&, eCOLOR_OBSERVER);
template std::pair<double, double> AlgoCCT::CctHandle<double>::getPlanckianUV (const std::pair<double, double>&, eCOLOR_OBSERVER);

// cct_compute (adapter defined in this TU)
template bool AlgoCCT::CctHandle<float >::cct_compute (const double&, const double&, double&, double&, const LutRow*, std::size_t);
template bool AlgoCCT::CctHandle<double>::cct_compute (const double&, const double&, double&, double&, const LutRow*, std::size_t);
