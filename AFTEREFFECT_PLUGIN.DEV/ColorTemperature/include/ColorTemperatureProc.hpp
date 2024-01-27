#ifndef __IMAGE_LAB_IMAGE_COLOR_TEMPERATURE_ALGO_PROC__
#define __IMAGE_LAB_IMAGE_COLOR_TEMPERATURE_ALGO_PROC__

#include "CommonColorTemperature.hpp"
#include "CommonAuxPixFormat.hpp"
#include "ColorTemperatureConstants.hpp"

constexpr float  f32WhitePoint = tWhitePoint<float>;
constexpr double f64WhitePoint = tWhitePoint<double>;

inline float rgb2cct (const fRGB& pixel) noexcept
{
	return __srgb2cct (pixel.R, pixel.G, pixel.B);
}

inline double rgb2cct(const dRGB& pixel) noexcept
{
	return __srgb2cct (pixel.R, pixel.G, pixel.B);
}


inline fRGB cct2rgb (const float& temperature) noexcept
{
	fRGB pixel;
	__cct2srgb (temperature, pixel.R, pixel.G, pixel.B);
	return pixel;
}

inline dRGB cct2rgb (const double& temperature) noexcept
{
	dRGB pixel;
	__cct2srgb (temperature, pixel.R, pixel.G, pixel.B);
	return pixel;
}



#endif /* __IMAGE_LAB_IMAGE_COLOR_TEMPERATURE_ALGO_PROC__ */
