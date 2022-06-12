#pragma once

#include "CommonPixFormat.hpp"
#include "ClassRestrictions.hpp"


template <typename T>
inline constexpr T sq (const T& x) noexcept
{
	return (x * x);
}

template <class T, std::enable_if_t<!is_YUV_proc<T>::value>* = nullptr>
inline constexpr int color_distance(const T& c1, const T& c2) noexcept
{
	return (sq(c1.B - c2.B) + sq(c1.G - c2.G) + sq(c1.R - c2.R));
}


class Pixel
{
public:
	A_long x, y;
	Pixel() { ; }
	explicit Pixel (const A_long& x0, const A_long& y0)
	{
		x = x0;
		y = y0;
	}
};


template <typename T>
class Color
{
public:
	T r, g, b;

	Color() { ; }
	explicit Color (const T& r0, const T& g0, const T& b0)
	{
		r = r0;
		g = g0;
		b = b0;
	}

};