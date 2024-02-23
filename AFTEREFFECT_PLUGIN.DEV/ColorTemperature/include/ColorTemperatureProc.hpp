#ifndef __IMAGE_LAB_IMAGE_COLOR_TEMPERATURE_ALGO_PROC__
#define __IMAGE_LAB_IMAGE_COLOR_TEMPERATURE_ALGO_PROC__

#include "CommonColorTemperature.hpp"
#include "CommonAuxPixFormat.hpp"
#include "ColorTemperatureEnums.hpp"

template<typename T, typename std::enable_if<std::is_floating_point<T>::value>::type* = nullptr>
void getColorCoefficients (const T& cct, const T& tint, T& R, T& G, T& B) noexcept
{
	constexpr T tMin {static_cast<T>(algoColorTempMin)};
	constexpr T tMax {static_cast<T>(algoColorTempMax)};

	return;
}


#endif /* __IMAGE_LAB_IMAGE_COLOR_TEMPERATURE_ALGO_PROC__ */
