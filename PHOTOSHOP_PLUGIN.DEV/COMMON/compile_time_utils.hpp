#ifndef __IMAGE_LAB_COMPILE_TIME_UTILS__
#define __IMAGE_LAB_COMPILE_TIME_UTILS__

#include <type_traits>

template<typename T>
T MIN(const T& a, const T& b) { return ((a < b) ? a : b); }

template<typename T>
T MAX(const T& a, const T& b) { return ((a > b) ? a : b); }

template <typename T>
T CLAMP_VALUE(const T& val, const T& min, const T& max)
{
	return ((val < min) ? min : ((val > max) ? max : val));
}

template <typename T>
constexpr typename std::enable_if<std::is_integral<T>::value, T>::type CreateAlignment(const T& x, const T& a)
{
	return (x > 0) ? ((x + a - 1) / a * a) : a;
}


#endif /* __IMAGE_LAB_COMPILE_TIME_UTILS__ */