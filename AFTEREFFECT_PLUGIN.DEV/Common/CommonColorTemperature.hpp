#ifndef __IMAGELAB_COLOR_TEMPERATURE_COMPUTATION_ALGO__
#define __IMAGELAB_COLOR_TEMPERATURE_COMPUTATION_ALGO__

#include <type_traits>

template <typename T>
inline const typename std::enable_if<std::is_floating_point<T>::value, T>::type __srgb2cct (const T& R, const T& G, const T& B) noexcept
{
	/* convert sRGB [input range 0...0.999 to Correlated Color Temperature] */
	const T X = R * static_cast<T>(0.649926) + G * static_cast<T>(0.103455) + B * static_cast<T>(0.197109);
	const T Y = R * static_cast<T>(0.234327) + G * static_cast<T>(0.743075) + B * static_cast<T>(0.022598);
	const T Z = /* r * 0.000000 + */           G * static_cast<T>(0.053077) + B * static_cast<T>(1.035763);
	const T x = X / (X + Y + Z);
	const T y = Y / (X + Y + Z);
	const T n = (x - static_cast<T>(0.3320)) / (static_cast<T>(0.18580) - y);
	return static_cast<T>(437.0) * n * n * n + static_cast<T>(3601.0) * n * n + static_cast<T>(6861.0) * n + static_cast<T>(5517.0);
}


template <typename T, typename std::enable_if<std::is_floating_point<T>::value>::type* = nullptr>
inline void  __cct2srgb (const T& temperature, T& R, T& G, T& B) noexcept
{
	/* TODO */
}


#endif /* __IMAGELAB_COLOR_TEMPERATURE_COMPUTATION_ALGO__ */