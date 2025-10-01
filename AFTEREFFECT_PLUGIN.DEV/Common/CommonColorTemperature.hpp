#ifndef __IMAGELAB_COLOR_TEMPERATURE_COMPUTATION_ALGO__
#define __IMAGELAB_COLOR_TEMPERATURE_COMPUTATION_ALGO__

#include <type_traits>

template <typename T>
inline const typename std::enable_if<std::is_floating_point<T>::value, T>::type __rgb2cct (const T R, const T G, const T B) noexcept
{
	/* 
		Convert RGB to Correlated Color Temperature under Wide-Gamut/RGB observer and D65 illuminant.
		In order to properly use this conversion, the RGB values must be linear and in following range:  0.f <= R,G,B <= 1.f 
	*/
	const T X = R * static_cast<T>(0.6499260) + G * static_cast<T>(0.1034550) + B * static_cast<T>(0.1971090);
	const T Y = R * static_cast<T>(0.2343270) + G * static_cast<T>(0.7430750) + B * static_cast<T>(0.0225980);
	const T Z = /* R * 0.000000 + */            G * static_cast<T>(0.0530770) + B * static_cast<T>(1.0357630);
	const T sum_XYZ = X + Y + Z;
	const T x = (0 != sum_XYZ ? X / (X + Y + Z) : 0);
	const T y = (0 != sum_XYZ ? Y / (X + Y + Z) : 0);
	const T n = (x - static_cast<T>(0.3320)) / (static_cast<T>(0.18580) - y);
	return static_cast<T>(437.0) * n * n * n + static_cast<T>(3601.0) * n * n + static_cast<T>(6861.0) * n + static_cast<T>(5517.0);
}


template <typename T, typename std::enable_if<std::is_floating_point<T>::value>::type* = nullptr>
inline void  __cct2rgb (const T temperature, T& R, T& G, T& B) noexcept
{
	/* TODO */
}


#endif /* __IMAGELAB_COLOR_TEMPERATURE_COMPUTATION_ALGO__ */