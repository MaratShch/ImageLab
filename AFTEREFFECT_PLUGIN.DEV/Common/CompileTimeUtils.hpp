#ifndef __IMAGE_LAB_COMPILE_TIME_UTILS__
#define __IMAGE_LAB_COMPILE_TIME_UTILS__

#include <type_traits>

template <typename T>
inline T 
#ifdef __NVCC__
__device__
#endif
MIN_VALUE (const T& a, const T& b) noexcept { return ((a < b) ? a : b); }

template <typename T>
inline T 
#ifdef __NVCC__
__device__
#endif
MAX_VALUE (const T& a, const T& b) noexcept  { return ((a > b) ? a : b); }

template <typename T>
inline T 
#ifdef __NVCC__
__device__
#endif
MIN3_VALUE(const T& a, const T& b, const T& c) noexcept  { return (a < b) ? MIN_VALUE(a, c) : MIN_VALUE(b, c); }

template <typename T>
inline T 
#ifdef __NVCC__
__device__
#endif
MAX3_VALUE(const T& a, const T& b, const T& c) noexcept  { return (a > b) ? MAX_VALUE(a, c) : MAX_VALUE(b, c); }

template <typename T>
inline T 
#ifdef __NVCC__
__device__
#endif
CLAMP_VALUE(const T& val, const T& min, const T& max) noexcept
{
	return ((val < min) ? min : ((val > max) ? max : val));
}

template <typename T>
constexpr typename std::enable_if<std::is_integral<T>::value, T>::type CreateAlignment(const T& x, const T& a)
{
	return (x > 0) ? ((x + a - 1) / a * a) : a;
}


template <typename T>
constexpr typename std::enable_if<std::is_integral<T>::value, T>::type IsPowerOf2 (const T& x)
{
	return (x && !(x & (x - static_cast<T>(1))));
}


#endif /* __IMAGE_LAB_COMPILE_TIME_UTILS__ */