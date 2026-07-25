#ifndef __IMAGELAB2_MIRED_CONVERT__
#define __IMAGELAB2_MIRED_CONVERT__

// =============================================================================
// mired_convert.hpp - reciprocal conversions between color temperature (K)
// and mired (MIcro REciprocal Degree, a.k.a. mirek / MK^-1).
//
//   mired = 1e6 / CCT[K]        CCT[K] = 1e6 / mired
//
// WHY MIRED
//   Equal mired steps are ~equal PERCEIVED color-temperature steps, while
//   equal Kelvin steps are not (1 mired ~ 1 JND). A UI temperature slider
//   should therefore be linear in mired, not in Kelvin. Example: 3000->3100 K
//   is ~10.7 mired (large, obvious) but 9000->9100 K is ~1.2 mired (barely
//   visible) - both "100 K", very different perceptually.
//
// TEMPLATE / PRECISION
//   Templated on a floating-point scalar T (float or double), SFINAE-guarded.
//   The reciprocal is evaluated in double internally for accuracy, then
//   returned as T; the division cost is trivial and this is not a per-pixel
//   path. constexpr + noexcept so it can be used in constant expressions.
//
// DOMAIN
//   Both quantities are strictly positive in the working range
//   (1000..26000 K  <->  ~38.46..1000 mired). Input <= 0 is out of domain;
//   to keep the functions total and constexpr-usable they return 0 for a
//   non-positive input rather than dividing by zero.
//
// C++14, header-only, OS- and compiler-independent.
// =============================================================================

#include <type_traits>

namespace AlgoCCT
{
    // Numerator of the reciprocal relation (1e6). Kept as a named constant so
    // both directions provably use the same value.
    constexpr double MIRED_SCALE = 1.0e6;

    // -------------------------------------------------------------------------
    // cct_to_mired - color temperature [K] -> mired.
    //   mired = 1e6 / cct
    // Returns 0 for cct <= 0 (out of domain) to avoid division by zero.
    // -------------------------------------------------------------------------
    template
    <
        typename T,
        typename std::enable_if<std::is_floating_point<T>::value>::type* = nullptr
    >
    constexpr T cct_to_mired (const T cct) noexcept
    {
        return (cct > static_cast<T>(0))
             ? static_cast<T>(MIRED_SCALE / static_cast<double>(cct))
             : static_cast<T>(0);
    }

    // -------------------------------------------------------------------------
    // mired_to_cct - mired -> color temperature [K].
    //   cct = 1e6 / mired
    // Returns 0 for mired <= 0 (out of domain) to avoid division by zero.
    // -------------------------------------------------------------------------
    template
    <
        typename T,
        typename std::enable_if<std::is_floating_point<T>::value>::type* = nullptr
    >
    constexpr T mired_to_cct (const T mired) noexcept
    {
        return (mired > static_cast<T>(0))
             ? static_cast<T>(MIRED_SCALE / static_cast<double>(mired))
             : static_cast<T>(0);
    }

} // namespace AlgoCCT

#endif // __IMAGELAB2_MIRED_CONVERT__
