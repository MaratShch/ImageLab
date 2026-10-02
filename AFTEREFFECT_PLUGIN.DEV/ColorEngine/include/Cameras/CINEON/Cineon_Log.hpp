/*
 * Cineon_Log.hpp
 * !!! GENERATED FILE - DO NOT EDIT MANUALLY !!!
 * Kodak Cineon film log decode + encode (10-bit printing-density convention,
 * white 685, black 95, 300 code values/decade, default black offset
 * 10^((95-685)/300)). Legacy but still met in scans and as the ancestor of
 * many "filmic" profiles.
 *   decode: x = (10^((1023*y-685)/300) - bo) / (1 - bo)
 *   encode: y = (685 + 300*log10(x*(1-bo)+bo)) / 1023
 * Constants via colour-science (verified). Encoded domain: normalized [0,1].
 * Generated: 2026-07-07 08:20:28 | C++14
 */
#ifndef __GENERATED_CINEON_LOG_HPP__
#define __GENERATED_CINEON_LOG_HPP__
#include <cmath>
namespace Cineon_Log
{
    // Default black offset = 10^((95-685)/300); stored at full double repr.
    constexpr double kBlackOffset = 0.0107977516232771;

    template<typename T>
    inline T decode(T y, T bo = static_cast<T>(kBlackOffset)) noexcept
    {
        return (std::pow(static_cast<T>(10),
                         (static_cast<T>(1023) * y - static_cast<T>(685))
                         / static_cast<T>(300)) - bo) / (static_cast<T>(1) - bo);
    }

    template<typename T>
    inline T encode(T x, T bo = static_cast<T>(kBlackOffset)) noexcept
    {
        return (static_cast<T>(685) + static_cast<T>(300) *
                std::log10(x * (static_cast<T>(1) - bo) + bo))
               / static_cast<T>(1023);
    }
} // namespace Cineon_Log
#endif
