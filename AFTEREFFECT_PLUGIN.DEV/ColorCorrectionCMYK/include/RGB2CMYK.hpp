#pragma once

#include "CompileTimeUtils.hpp"

template<typename T>
inline const typename std::enable_if<std::is_floating_point<T>::value>::type
rgb_to_cmyk (const T& R, const T& G, const T& B, T& C, T& M, T& Y, T& K) noexcept
{
	constexpr T One = static_cast<T>(1);
	T const& k = One - MAX3_VALUE(R, G, B);
	T const& ReciprocOneMinusK = One / (One - k);

	T const& c = (One - R - k) * ReciprocOneMinusK;
	T const& m = (One - G - k) * ReciprocOneMinusK;
	T const& y = (One - B - k) * ReciprocOneMinusK;

	C = c;
	M = m;
	Y = y;
	K = k;

	return;
}


template<typename T>
inline const typename std::enable_if<std::is_floating_point<T>::value>::type
cmyk_to_rgb (const T& C, const T& M, const T& Y, const T& K, T& R, T& G, T& B) noexcept
{
	constexpr T One = static_cast<T>(1);
	T const& OneMinucC = One - C;
	T const& OneMinusM = One - M;
	T const& OneMinusY = One - Y;
	T const& OneMinusK = One - K;

	R = OneMinucC * OneMinusK;
	G = OneMinusM * OneMinusK;
	B = OneMinusY * OneMinusK;
	return;
}