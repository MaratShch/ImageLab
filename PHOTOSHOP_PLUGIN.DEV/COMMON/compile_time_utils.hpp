#ifndef __IMAGE_LAB_COMPILE_TIME_UTILS__
#define __IMAGE_LAB_COMPILE_TIME_UTILS__

#include <type_traits>

template<typename T>
T MIN(T a, T b) { return ((a < b) ? a : b); }

template<typename T>
T MAX(T a, T b) { return ((a > b) ? a : b); }

template <typename T>
constexpr typename std::enable_if<std::is_integral<T>::value, T>::type CreateAlignment(T x, T a)
{
	return (x > 0) ? ((x + a - 1) / a * a) : a;
}


#endif /* __IMAGE_LAB_COMPILE_TIME_UTILS__ */