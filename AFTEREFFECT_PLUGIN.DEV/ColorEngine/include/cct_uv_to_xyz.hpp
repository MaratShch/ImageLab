#ifndef __IMAGELAB2_CCT_UV_TO_XYZ_HELPERS__
#define __IMAGELAB2_CCT_UV_TO_XYZ_HELPERS__

// =============================================================================
// cct_uv_to_xyz.hpp - inverse chromaticity projections for the CCT engine
// =============================================================================
//
// PURPOSE
//   Completes the synthesis direction of the CCT engine:
//
//       CCT/Duv --getPlanckianUV()--> (u,v) --THIS FILE--> XYZ --matrix--> RGB
//
//   ComputeCct() consumes (u,v); getPlanckianUV() produces (u,v). This header
//   provides the missing step from CIE 1960 UCS (u,v) back to tristimulus XYZ
//   (and to CIE 1931 (x,y) as an intermediate, exposed because it is useful
//   on its own for UI/scope display).
//
//   NOTE ON SEMANTICS: a chromaticity carries NO luminance. The caller must
//   supply the desired Y (luminance) for the reconstructed point; Y = 1.0 is
//   the conventional choice for a white point. This function reconstructs the
//   WHITE POINT (or any single chromaticity) - it cannot and does not
//   "restore" per-pixel image data, which is information CCT/Duv never held.
//
// FORMULAS AND COEFFICIENTS
//   Forward (defined in cct_interface.cpp / the analysis bridge):
//       u = 4X / (X + 15Y + 3Z)
//       v = 6Y / (X + 15Y + 3Z)
//   The constants 4, 15, 3, 6 are the MacAdam CIE 1960 UCS projection
//   constants. Inverting that projective map gives, via CIE 1931 (x,y):
//       denom = 2u - 8v + 4
//       x = 3u / denom
//       y = 2v / denom
//   and from (x, y, Y) to XYZ (the standard chromaticity lift):
//       X = (x / y) * Y
//       Z = ((1 - x - y) / y) * Y
//   Derivation sketch: u = 4x / (-2x + 12y + 3) and v = 6y / (-2x + 12y + 3)
//   (the classic xy->uv form); solving the two equations for x and y yields
//   the 3/2/(-8)/4 coefficients above.
//
// PRECISION
//   Follows the project precision rule: computation in double; thin float
//   convenience overloads at the API boundary. This code runs per white
//   point (a handful of times per parameter change), never per pixel.
//
// VERIFICATION
//   Round-trip uv->XYZ->uv is identity to ~1 ulp; XYZ values validated
//   against the colour-science reference for D65 and the Planckian points
//   used in the engine test suite.
//
// PORTABILITY
//   Standard C++14 only; OS- and compiler-independent. Header-only.
// =============================================================================

#include <utility>

namespace AlgoCCT
{

    // Simple aggregate for a tristimulus value (kept local to avoid coupling
    // this header to any project-wide vector type).
    struct XYZ_d
    {
        double X;
        double Y;
        double Z;
    };

    // -------------------------------------------------------------------------
    // uv_to_xy - CIE 1960 UCS (u, v) -> CIE 1931 chromaticity (x, y).
    //
    // denom = 2u - 8v + 4;  x = 3u/denom;  y = 2v/denom.
    // The denominator is strictly positive for every physically meaningful
    // chromaticity (all Planckian/near-locus points; e.g. D65 gives
    // denom = 2*0.19784 - 8*0.31222 + 4 = 1.89793). A non-finite or
    // degenerate input is the caller's contract violation; we guard the
    // division only against exact zero to keep the function total.
    // -------------------------------------------------------------------------
    inline std::pair<double, double> uv_to_xy (double u, double v) noexcept
    {
        const double denom = 2.0 * u - 8.0 * v + 4.0;
        if (0.0 == denom)
            return std::make_pair(0.0, 0.0);        // degenerate input

        return std::make_pair(3.0 * u / denom,      // x
                              2.0 * v / denom);     // y
    }

    // -------------------------------------------------------------------------
    // xy_to_XYZ - CIE 1931 chromaticity (x, y) + luminance Y -> XYZ.
    //
    // X = (x/y)*Y;  Z = ((1-x-y)/y)*Y.  y = 0 (the alychne) carries no
    // luminance and cannot be lifted; returns all-zero in that degenerate
    // case.
    // -------------------------------------------------------------------------
    inline XYZ_d xy_to_XYZ (double x, double y, double Y = 1.0) noexcept
    {
        if (0.0 == y)
            return XYZ_d{ 0.0, 0.0, 0.0 };          // degenerate input

        const double X = (x / y) * Y;
        const double Z = ((1.0 - x - y) / y) * Y;
        return XYZ_d{ X, Y, Z };
    }

    // -------------------------------------------------------------------------
    // uv_to_XYZ - CIE 1960 UCS (u, v) + luminance Y -> XYZ. Composition of
    // the two functions above; the primary entry point of this header.
    //
    // Typical use (build the white point of a CCT/Duv pair):
    //     const auto uv  = handle.getPlanckianUV(cct, duv, observer_CIE_1931);
    //     const XYZ_d wp = uv_to_XYZ(uv.first, uv.second);       // Y = 1
    //     // wp feeds the CAT (source or target white), or, via the inverse
    //     // working-space matrix, becomes a display RGB swatch.
    // -------------------------------------------------------------------------
    inline XYZ_d uv_to_XYZ (double u, double v, double Y = 1.0) noexcept
    {
        const std::pair<double, double> xy = uv_to_xy(u, v);
        return xy_to_XYZ(xy.first, xy.second, Y);
    }

    // -------------------------------------------------------------------------
    // Float convenience overloads (thin boundary casts; math stays double).
    // -------------------------------------------------------------------------
    inline std::pair<float, float> uv_to_xy (float u, float v) noexcept
    {
        const std::pair<double, double> xy =
            uv_to_xy(static_cast<double>(u), static_cast<double>(v));
        return std::make_pair(static_cast<float>(xy.first),
                              static_cast<float>(xy.second));
    }

    inline XYZ_d uv_to_XYZ (float u, float v, float Y) noexcept
    {
        return uv_to_XYZ(static_cast<double>(u),
                         static_cast<double>(v),
                         static_cast<double>(Y));
    }

}; // namespace AlgoCCT

#endif // __IMAGELAB2_CCT_UV_TO_XYZ_HELPERS__
