#pragma once

#include <cmath>
#include <cstdint>
#include <utility>
#include "FastAriphmetics.hpp"

namespace FourierTransform
{
	
template <typename T>
inline void dft_1D
(
	const T* in,
	T* out,
	int32_t N
) noexcept
{
    constexpr T PI = static_cast<T>(3.14159265358979323846);

    for (int32_t k = 0; k < N; k++)
    {
        T sumRe{0};
        T sumIm{0};

        for (int32_t n = 0; n < N; n++)
        {
            const T xr = in[2 * n + 0];
            const T xi = in[2 * n + 1];

            const T angle = static_cast<T>(-2 * k * n) * PI / static_cast<T>(N);
			T cr, sr;
			FastCompute::SinCos(angle, sr, cr);
			
            sumRe += xr * cr - xi * sr;
            sumIm += xr * sr + xi * cr;
        }

        out[2 * k + 0] = sumRe;
        out[2 * k + 1] = sumIm;
    }
	
	return;
}

}