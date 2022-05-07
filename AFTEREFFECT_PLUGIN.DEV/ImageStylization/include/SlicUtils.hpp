/**
 Simple Linear Interactive Clustering utils
**/

#pragma once
#include "CommonPixFormat.hpp"

template <typename T>
inline constexpr T sq (const T& x) noexcept
{ 
	return (x * x);
}

template <class T, std::enable_if_t<!is_YUV_proc<T>::value>* = nullptr>
inline constexpr int color_distance (const T& c1, const T& c2) noexcept
{
	return (sq(c1.B - c2.B) + sq(c1.G - c2.G) + sq(c1.R - c2.R));
}


template <class T, std::enable_if_t<!is_YUV_proc<T>::value>* = nullptr>
inline constexpr float color_distance(const T& c1, const T& c2) noexcept
{
	return (sq(c1.B - c2.B) + sq(c1.G - c2.G) + sq(c1.R - c2.R));
}
