#ifndef __IMAGE_LAB_IMAGE_COLOR_ILLUMINANT_ALGORITHM__
#define __IMAGE_LAB_IMAGE_COLOR_ILLUMINANT_ALGORITHM__

#include "AlgoPlanckianLocus.hpp"
#include "ColorTemperatureEnums.hpp"
#include <limits>

using ColorTemperatureT = WaveLengthT;

/* White Point color temperature for every Illuminant in Kelvins */
constexpr ColorTemperatureT white_point_D65						= 6504.0;
constexpr ColorTemperatureT white_point_D65_Cloudy_Tint			= 0.030;
constexpr ColorTemperatureT white_point_Tungsten				= 3200.0;
constexpr ColorTemperatureT white_point_FluorescentDayLight		= 6500.0;
constexpr ColorTemperatureT white_point_FluorescentWarmWhite	= 3000.0;
constexpr ColorTemperatureT white_point_FluorescentSoftWhite	= 4200.0;
constexpr ColorTemperatureT white_point_Incandescent			= 2700.0;
constexpr ColorTemperatureT white_point_Moonlight				= 4100.0;


template<typename T, typename std::enable_if<std::is_floating_point<T>::value>::type* = nullptr>
inline const std::vector<T> init_illuminant (const T& minWlength, const T& maxWlength, const T& step, const T& whitePoint) noexcept
{
	const size_t vectorSize = static_cast<size_t>((maxWlength - minWlength) / step) + 1;
	std::vector<T> spectralRadiance(vectorSize);
	T maxIlluminant {std::numeric_limits<T>::min()};
	T waveLength {minWlength};

	/* compute non normalized Spectral Radiance values in correspondent to defined wave length */
	for (size_t i = 0; i < vectorSize; i++)
	{
		T value = Planck(waveLength, whitePoint);
		maxIlluminant = std::max(value, maxIlluminant);
		spectralRadiance[i] = value;
		waveLength += step;
	}

	/* Normalize computed Spectral Radiance values before return to caller. */
	/* Perform in-place normalizing											*/					
	SpectralRadianceNormalize(maxIlluminant, spectralRadiance);

	return spectralRadiance;
}


template<typename T, typename std::enable_if<std::is_floating_point<T>::value>::type* = nullptr>
inline const std::vector<T> init_illuminant_D65 (void) noexcept
{
	return init_illuminant (waveLengthStart, waveLengthStop, wavelengthStepFinest, white_point_D65);
}

template<typename T, typename std::enable_if<std::is_floating_point<T>::value>::type* = nullptr>
inline const std::vector<T> init_illuminant_D65_Cloudy(void) noexcept
{
	constexpr T tintFactor = static_cast<T>(1.0) - white_point_D65_Cloudy_Tint;
	std::vector<T> d65_illuminant_with_tint = init_illuminant (waveLengthStart, waveLengthStop, wavelengthStepFinest, white_point_D65);
	/* apply Tint value */
	const size_t vectorSize = d65_illuminant_with_tint.size();
	for (size_t i = 0; i < vectorSize; i++)
		d65_illuminant_with_tint[i] *= tintFactor;

	return d65_illuminant_with_tint;
}

template<typename T, typename std::enable_if<std::is_floating_point<T>::value>::type* = nullptr>
inline const std::vector<T> init_illuminant_Tungsten (void) noexcept
{
	return init_illuminant (waveLengthStart, waveLengthStop, wavelengthStepFinest, white_point_Tungsten);
}

template<typename T, typename std::enable_if<std::is_floating_point<T>::value>::type* = nullptr>
inline const std::vector<T> init_illuminant_FluorescentDayLight (void) noexcept
{
	return init_illuminant(waveLengthStart, waveLengthStop, wavelengthStepFinest, white_point_FluorescentDayLight);
}

template<typename T, typename std::enable_if<std::is_floating_point<T>::value>::type* = nullptr>
inline const std::vector<T> init_illuminant_FluorescentWarmWhite (void) noexcept
{
	return init_illuminant (waveLengthStart, waveLengthStop, wavelengthStepFinest, white_point_FluorescentWarmWhite);
}

template<typename T, typename std::enable_if<std::is_floating_point<T>::value>::type* = nullptr>
inline const std::vector<T> init_illuminant_FluorescentSoftWhite (void) noexcept
{
	return init_illuminant (waveLengthStart, waveLengthStop, wavelengthStepFinest, white_point_FluorescentSoftWhite);
}

template<typename T, typename std::enable_if<std::is_floating_point<T>::value>::type* = nullptr>
inline const std::vector<T> init_illuminant_Incandescent (void) noexcept
{
	return init_illuminant (waveLengthStart, waveLengthStop, wavelengthStepFinest, white_point_Incandescent);
}

template<typename T, typename std::enable_if<std::is_floating_point<T>::value>::type* = nullptr>
inline const std::vector<T> init_illuminant_Moonlight (void) noexcept
{
	return init_illuminant (waveLengthStart, waveLengthStop, wavelengthStepFinest, white_point_Moonlight);
}


#endif /* __IMAGE_LAB_IMAGE_COLOR_ILLUMINANT_ALGORITHM__ */