#ifndef __IMAGE_LAB_IMAGE_COLOR_TEMPERATURE_ALGO_PROC__
#define __IMAGE_LAB_IMAGE_COLOR_TEMPERATURE_ALGO_PROC__

#include "CommonColorTemperature.hpp"
#include "CommonAuxPixFormat.hpp"


template<typename T>
constexpr T tWhitePoint{ 6500 };


template <typename T>
inline float rgb2cct (const _tXYZPix<T>& pixel) noexcept
{
	return __rgb2cct (pixel.R, pixel.G, pixel.B);
}


template <typename T>
inline fRGB cct2rgb (const T& temperature) noexcept
{
	_tXYZPix<T> pixel;
	__cct2rgb (temperature, pixel.R, pixel.G, pixel.B);
	return pixel;
}



#endif /* __IMAGE_LAB_IMAGE_COLOR_TEMPERATURE_ALGO_PROC__ */
