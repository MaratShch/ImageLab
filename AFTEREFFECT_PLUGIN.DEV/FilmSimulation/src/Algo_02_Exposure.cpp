#include "Algo_02_Exposure.hpp"
#include <cmath>

AlgoType Algo_02_GainFromStops (const double exposureStops) noexcept
{
    // std::exp2 is C++11 and exact for integral arguments, unlike pow(2,x).
    // This runs once per frame, so accuracy costs nothing here -- the fast-math
    // replacements belong at stage 8, where exp is called per pixel.
    return static_cast<AlgoType>(std::exp2 (exposureStops));
}

void Algo_02_Exposure_Plane
(
    const AlgoType* RESTRICT src,
    AlgoType*       RESTRICT dst,
    const std::size_t             count,
    const AlgoType                gain
) noexcept
{
    // Single counted loop, no branches, no early exit, invariant gain hoisted
    // by the caller. This is the shape every later stage should copy: GCC
    // vectorises it at -O3 (NOT at -O2 -- -O2 needs -ftree-vectorize), and
    // MSVC 2015 auto-vectorises it with /O2 /arch:AVX2.
    for (std::size_t i = 0u; i < count; ++i)
    {
        dst[i] = src[i] * gain;
    }
}
