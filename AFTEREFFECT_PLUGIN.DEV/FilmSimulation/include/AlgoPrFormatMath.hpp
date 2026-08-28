#ifndef __IMAGELAB2_PR_FORMAT_MATH_HPP__
#define __IMAGELAB2_PR_FORMAT_MATH_HPP__

// =============================================================================
// AlgoPrFormatMath.hpp - DIRECT computation of the transfer functions. No LUT,
// no table, no gather, no libm call on the pixel path.
//
// Owner instruction: "LUT usage less preferred - please use direct computations.
// I believe you may use fast approximate computations for Pow and for
// Log/Exp." This header is that, and it is the SINGLE definition of the
// transfer curve for the whole conversion layer: the scalar readers/writers and
// the AVX2 kernels both go through it, so the two paths agree to within one
// float ULP instead of being two independent approximations.
//
// -----------------------------------------------------------------------------
// WHICH FAST APPROXIMATION, AND WHY NOT THE OTHERS - MEASURED, NOT ASSUMED
// -----------------------------------------------------------------------------
// The metric that matters for a pixel path is not relative error, it is error
// expressed in DESTINATION CODES: 0.5 code is the rounding boundary, so any
// approximation whose error approaches 0.5 changes the output value. Max error
// over all 32768 16-bit codes, both directions of the sRGB curve:
//
//   implementation                        enc 8u   enc 16u   dec 8u   dec 16u
//   ------------------------------------------------------------------------
//   fast_log2 / fast_exp2 (this header)   0.0001     0.010   0.0002     0.021
//   FastCompute::AVX2::Log + fast_exp2    0.0016     0.201   0.0034     0.436
//   FastCompute::AVX2::Pow                5.8753   754.961   2.8679   368.518
//   FastCompute::Pow (bit hack)           6.8864   884.887   8.8927  1142.698
//   FastCompute::Log + Exp                 167.6  21533.2     243.1  31240.8
//
// So:
//   * ⚠ FastCompute::AVX2::Exp is the Schraudolph integer-reinterpretation
//     trick. Measured max relative error 2.98e-2 - THREE PERCENT. That is
//     ~750 codes at 16 bit and ~6 codes at 8 bit: visible banding and a colour
//     shift, not a rounding difference. It cannot carry a pixel transfer, and
//     since AVX2::Pow is built on it, neither can that.
//   * ⚠ FastCompute::Pow (the `u.x = b*(u.x - 1064866805) + 1064866805` bit
//     hack) is the same class of approximation - 885 codes at 16 bit.
//   * ⚠ FastCompute::Log is currently BROKEN, independently of accuracy:
//     __int_as_float() in FastAriphmetics.hpp is declared to return `int`, so
//     the mantissa gets truncated to an integer. Log(0.75) returns -2.3407817
//     instead of -0.2876821. Exact powers of two survive (m == 1.0), which is
//     why it can pass a casual test. One-line fix: return type `float`.
//   * FastCompute::AVX2::Log is sound (njuffa's log1p form) and would be usable
//     at 8 bit, but at 16 bit it lands at 0.44 code - close enough to the 0.5
//     boundary that some codes flip. It also uses mm256_fmaf(), which is
//     mul+add rather than a real FMA, and that is where most of its accuracy
//     goes; with -mfma available there is no reason to pay for that.
//
// The pair below is therefore what ships: a minimax polynomial log2 and exp2
// over the exponent/mantissa decomposition, FMA throughout, branch-free, no
// table and no libm. Cost per channel is ~15 FMA - cheaper than the LUT path it
// replaces, which paid a dependent load on ingest and a ~15-step binary search
// on egress.
//
// -----------------------------------------------------------------------------
// TRANSFER SELECTION
// -----------------------------------------------------------------------------
// Dropping the LUT means the curve is no longer supplied by the caller, so it
// has to be named here. kTransfer picks it at COMPILE TIME (one constant, both
// paths, no per-pixel branch):
//
//   kTransfer_sRGB   - IEC 61966-2-1. Default; what After Effects uses for 8/16
//                      bit project working space.
//   kTransfer_Rec709 - ITU-R BT.709 OETF (the 0.45 / 4.5 / 1.099 form).
//   kTransfer_Gamma  - pure power law, exponent kGammaExp.
//
// The _Linear formats never reach this header: Adobe already linearized them.
//
// C++14, no allocation, no STL, no OS- or compiler-specific API. Scalar only -
// the vector twins live in AlgoPrFormatAVX2.hpp and use the same constants.
// =============================================================================

#include <cstdint>
#include <cstring>       // std::memcpy - the only legal float<->bits punning
#include <cmath>         // std::floor only

namespace AlgoPrIngest
{
    // =========================================================================
    // BIT PUNNING. std::memcpy, not a union and not a pointer cast.
    //
    // ⚠ This is not pedantry. `*(int*)&f` - the idiom in FastAriphmetics.hpp -
    // is a strict-aliasing violation; gcc-13 at -O2 with -fstrict-aliasing (on
    // by default) is entitled to reorder the store and the load, and does so in
    // real code. memcpy of 4 bytes compiles to a single register move on every
    // compiler in the build matrix, so this costs nothing.
    // =========================================================================
    inline std::int32_t bits_of(float f)
    {
        std::int32_t i = 0;
        std::memcpy(&i, &f, sizeof(i));
        return i;
    }

    inline float float_of(std::int32_t i)
    {
        float f = 0.0f;
        std::memcpy(&f, &i, sizeof(f));
        return f;
    }

    // =========================================================================
    // fast_log2 - x = 2^e * m, m in [1,2); degree-7 minimax on m-1.
    // Domain: x > 0. Callers clamp first; x <= 0 is not defined here.
    // =========================================================================
    inline float fast_log2(float x)
    {
        const std::int32_t xi = bits_of(x);
        const int   e  = ((xi >> 23) & 0xFF) - 127;
        const float m  = float_of((xi & 0x007FFFFF) | 0x3F800000);
        const float t  = m - 1.0f;
        float p = 1.459860554e-02f;
        p = std::fma(p, t, -7.592089396e-02f);
        p = std::fma(p, t,  1.886527228e-01f);
        p = std::fma(p, t, -3.214835301e-01f);
        p = std::fma(p, t,  4.717218708e-01f);
        p = std::fma(p, t, -7.202026917e-01f);
        p = std::fma(p, t,  1.442633691e+00f);
        p = std::fma(p, t,  8.116678600e-07f);
        return static_cast<float>(e) + p;
    }

    // =========================================================================
    // fast_exp2 - 2^y = 2^floor(y) * 2^frac; degree-6 minimax on the fraction,
    // the integer part folded straight into the exponent field.
    // =========================================================================
    inline float fast_exp2(float y)
    {
        const float fl = std::floor(y);
        const float f  = y - fl;
        float p = 2.187125795e-04f;
        p = std::fma(p, f, 1.238241248e-03f);
        p = std::fma(p, f, 9.686187232e-03f);
        p = std::fma(p, f, 5.547891155e-02f);
        p = std::fma(p, f, 2.402310971e-01f);
        p = std::fma(p, f, 6.931468376e-01f);
        p = std::fma(p, f, 1.000000006e+00f);
        const std::int32_t ei = static_cast<std::int32_t>(fl);
        return p * float_of((ei + 127) << 23);
    }

    //! x^p for x >= 0. x <= 0 returns 0 rather than a NaN, so a negative sample
    //! - legal on a scene-linear buffer - cannot poison the result.
    inline float fast_pow(float x, float p)
    {
        return (x > 0.0f) ? fast_exp2(p * fast_log2(x)) : 0.0f;
    }

    // =========================================================================
    // TRANSFER CURVES. Constants named once; the vector twins in
    // AlgoPrFormatAVX2.hpp reference these same names.
    // =========================================================================
    enum eTransferFunc
    {
        kTransfer_sRGB   = 0,   //!< IEC 61966-2-1
        kTransfer_Rec709 = 1,   //!< ITU-R BT.709 OETF
        kTransfer_Gamma  = 2    //!< pure power law, kGammaExp
    };

    //! ⚠ THE ONE PLACE THE CURVE IS CHOSEN. Compile-time, both paths.
    constexpr eTransferFunc kTransfer = kTransfer_sRGB;

    //! Exponent for kTransfer_Gamma. Ignored by the other two.
    constexpr float kGammaExp    = 2.2f;
    constexpr float kGammaInvExp = 1.0f / kGammaExp;

    // ---- sRGB (IEC 61966-2-1) ---------------------------------------------
    constexpr float kSrgbDecThreshold = 0.04045f;
    constexpr float kSrgbEncThreshold = 0.0031308f;
    constexpr float kSrgbSlope        = 12.92f;
    constexpr float kSrgbInvSlope     = 1.0f / 12.92f;
    constexpr float kSrgbOffset       = 0.055f;
    constexpr float kSrgbScale        = 1.055f;
    constexpr float kSrgbInvScale     = 1.0f / 1.055f;
    constexpr float kSrgbGamma        = 2.4f;
    constexpr float kSrgbInvGamma     = 1.0f / 2.4f;

    // ---- Rec.709 OETF ------------------------------------------------------
    constexpr float k709DecThreshold  = 0.081f;
    constexpr float k709EncThreshold  = 0.018f;
    constexpr float k709Slope         = 4.5f;
    constexpr float k709InvSlope      = 1.0f / 4.5f;
    constexpr float k709Offset        = 0.099f;
    constexpr float k709Scale         = 1.099f;
    constexpr float k709InvScale      = 1.0f / 1.099f;
    constexpr float k709Gamma         = 1.0f / 0.45f;
    constexpr float k709InvGamma      = 0.45f;

    //! Display-encoded [0,1] -> linear. Direct computation; no table.
    inline float transfer_decode(float c)
    {
        if (kTransfer == kTransfer_Gamma)
            return fast_pow(c, kGammaExp);
        if (kTransfer == kTransfer_Rec709)
            return (c <= k709DecThreshold)
                 ? (c * k709InvSlope)
                 : fast_pow((c + k709Offset) * k709InvScale, k709Gamma);
        return (c <= kSrgbDecThreshold)
             ? (c * kSrgbInvSlope)
             : fast_pow((c + kSrgbOffset) * kSrgbInvScale, kSrgbGamma);
    }

    //! Linear -> display-encoded [0,1]. Exact inverse of transfer_decode.
    inline float transfer_encode(float v)
    {
        if (v <= 0.0f) return 0.0f;
        if (kTransfer == kTransfer_Gamma)
            return fast_pow(v, kGammaInvExp);
        if (kTransfer == kTransfer_Rec709)
            return (v <= k709EncThreshold)
                 ? (v * k709Slope)
                 : (k709Scale * fast_pow(v, k709InvGamma) - k709Offset);
        return (v <= kSrgbEncThreshold)
             ? (v * kSrgbSlope)
             : (kSrgbScale * fast_pow(v, kSrgbInvGamma) - kSrgbOffset);
    }

} // namespace AlgoPrIngest

#endif // __IMAGELAB2_PR_FORMAT_MATH_HPP__
