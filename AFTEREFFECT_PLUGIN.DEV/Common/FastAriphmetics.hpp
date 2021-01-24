#pragma once

namespace FastCompute
{
	constexpr auto CHAR_BITS = 8;

	inline constexpr int Min(const int& x, const int& y)
	{
		return y + ((x - y) & ((x - y) >>
			(sizeof(int) * CHAR_BITS - 1)));
	}

	inline constexpr int Max(const int& x, const int& y)
	{
		return x - ((x - y) & ((x - y) >>
			(sizeof(int) * CHAR_BITS - 1)));
	}

	template <typename T>
	inline constexpr typename std::enable_if<std::is_integral<T>::value, T>::type Min(const T& x, const T& y)
	{	/* find minimal value between 2 fixed point values without branch */
		return y ^ ((x ^ y) & -(x < y));
	}

	template <typename T>
	inline constexpr typename std::enable_if<std::is_integral<T>::value, T>::type Max(const T& x, const T& y)
	{   /* find maximal value between 2 fixed point values without branch */
		return x ^ ((x ^ y) & -(x < y));
	}


	inline double Sqrt(const double x)
	{
		const double   xHalf = 0.50 * x;
		long long int  tmp = 0x5FE6EB50C7B537AAl - (*(long long int*)&x >> 1); //initial guess
		double         xRes = *(double*)&tmp;
		xRes *= (1.50 - (xHalf * xRes * xRes));
		return xRes * x;
	}

	inline float Sqrt(const float x)
	{
		const float xHalf = 0.50f * x;
		int   tmp = 0x5F3759DF - (*(int*)&x >> 1); //initial guess
		float xRes = *(float*)&tmp;
		xRes *= (1.50f - (xHalf * xRes * xRes));
		return xRes * x;
	}

}; /* namespace FastCompute */