#ifndef __IMAGE_LAB_COMPILE_TIME_UTILS__
#define __IMAGE_LAB_COMPILE_TIME_UTILS__

#include <type_traits>

template <typename T>
inline T MIN_VALUE (const T& a, const T& b) { return ((a < b) ? a : b); }

template <typename T>
inline T MAX_VALUE (const T& a, const T& b) { return ((a > b) ? a : b); }

template <typename T>
inline T MIN3_VALUE(const T& a, const T& b, const T& c) { return MIN_VALUE(c, MIN_VALUE(a, b)); }

template <typename T>
inline T MAX3_VALUE(const T& a, const T& b, const T& c) { return MAX_VALUE(c, MAX_VALUE(a, b)); }

template <typename T>
inline T CLAMP_VALUE(const T& val, const T& min, const T& max)
{
	return ((val < min) ? min : ((val > max) ? max : val));
}

template <typename T>
constexpr typename std::enable_if<std::is_integral<T>::value, T>::type CreateAlignment(const T& x, const T& a)
{
	return (x > 0) ? ((x + a - 1) / a * a) : a;
}

#endif /* __IMAGE_LAB_COMPILE_TIME_UTILS__ */