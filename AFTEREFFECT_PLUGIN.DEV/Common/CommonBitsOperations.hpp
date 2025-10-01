#pragma once

#ifdef USE_IMAGELAB_BITSET_MACROS
/* Not type checking provided ! */

#define IMLAB_BIT_SET(a,b)    ((a) |= (1ULL << (b)))
#define IMLAB_BIT_CLEAR(a,b)  ((a) &= ~(1ULL<< (b)))
#define IMLAB_BIT_FLIP(a,b)   ((a) ^= (1ULL << (b)))

#else

#include <type_traits>
#include <cstdint>

template <typename T>
inline constexpr typename std::enable_if<std::is_integral<T>::value, T>::type IMLAB_BIT_SET(const T x, const uint32_t pos) noexcept
{
	return x | static_cast<T>(1ull << pos);
}

template <typename T>
inline constexpr typename std::enable_if<std::is_integral<T>::value, T>::type IMLAB_BIT_CLEAR(const T x, const uint32_t pos) noexcept
{
	return x & static_cast<T>(~(1ull << pos));
}

template <typename T>
inline constexpr typename std::enable_if<std::is_integral<T>::value, T>::type IMLAB_BIT_FLIP(const T x, const uint32_t pos) noexcept
{
	return x ^ static_cast<T>(1ull << pos);
}

#endif