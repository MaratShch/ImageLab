#ifndef __IMAGE_LAB_COMPILE_TIME_UTILS__
#define __IMAGE_LAB_COMPILE_TIME_UTILS__

#include <type_traits>

template <typename T>
inline T 
#ifdef __NVCC__
inline __device__
#endif
MIN_VALUE (const T& a, const T& b) noexcept { return ((a < b) ? a : b); }

template <typename T>
inline T 
#ifdef __NVCC__
inline __device__
#endif
MAX_VALUE (const T& a, const T& b) noexcept  { return ((a > b) ? a : b); }

template <typename T>
inline T 
#ifdef __NVCC__
inline __device__
#endif
MIN3_VALUE(const T& a, const T& b, const T& c) noexcept  { return (a < b) ? MIN_VALUE(a, c) : MIN_VALUE(b, c); }

template <typename T>
inline T 
#ifdef __NVCC__
inline __device__
#endif
MAX3_VALUE(const T& a, const T& b, const T& c) noexcept  { return (a > b) ? MAX_VALUE(a, c) : MAX_VALUE(b, c); }

template <typename T>
inline T 
#ifdef __NVCC__
inline __device__
#endif
CLAMP_VALUE(const T& val, const T& min, const T& max) noexcept
{
	return ((val < min) ? min : ((val > max) ? max : val));
}

template <typename T>
constexpr typename std::enable_if<std::is_integral<T>::value, T>::type CreateAlignment(const T& x, const T& a) noexcept
{
	/* create value X aligned on A */
	return (x > 0) ? ((x + a - 1) / a * a) : a;
}


template <typename T>
constexpr typename std::enable_if<std::is_integral<T>::value, T>::type IsPowerOf2 (const T& x) noexcept
{
	return (x && !(x & (x - static_cast<const T>(1))));
}


template <typename T>
constexpr typename std::enable_if<std::is_integral<T>::value, T>::type ODD_VALUE(const T& x) noexcept
{
	return (x | static_cast<const T>(1));
}

template <typename T>
constexpr typename std::enable_if<std::is_floating_point<T>::value, T>::type ODD_VALUE(const T& x) noexcept
{
	return (x + static_cast<const T>(1));
}

template <typename T>
constexpr typename std::enable_if<std::is_integral<T>::value, bool>::type IsOddValue(const T& x) noexcept
{
	return ((x % static_cast<const T>(2) != static_cast<const T>(0)) ? true : false);
}

template <typename T>
constexpr typename std::enable_if<std::is_integral<T>::value, T>::type EVEN_VALUE(const T& x) noexcept
{
	return (x & ~(static_cast<const T>(1)));
}

template <typename T>
constexpr typename std::enable_if<std::is_integral<T>::value, bool>::type IsEvenValue(const T& x) noexcept
{
	return ((x % static_cast<const T>(2) == static_cast<const T>(0)) ? true : false);
}

template <typename T>
constexpr typename std::enable_if<std::is_integral<T>::value, T>::type HALF(const T& x) noexcept
{
	return (x >> static_cast<const T>(1));
}

template <typename T>
constexpr typename std::enable_if<std::is_floating_point<T>::value, T>::type HALF(const T& x) noexcept
{
	return (x / static_cast<const T>(2));
}

template <typename T>
constexpr typename std::enable_if<std::is_floating_point<T>::value, T>::type Lerp (const T& v0, const T& v1, const T& t) noexcept
{
	return v0 + (v1 - v0) * t;
}


#endif /* __IMAGE_LAB_COMPILE_TIME_UTILS__ */