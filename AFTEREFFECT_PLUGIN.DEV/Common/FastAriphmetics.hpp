#pragma once
#include <cmath>

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


	inline double Sqrt(const double& x)
	{
		const double   xHalf = 0.50 * x;
		long long int  tmp = 0x5FE6EB50C7B537AAl - (*(long long int*)&x >> 1); //initial guess
		double         xRes = *(double*)&tmp;
		xRes *= (1.50 - (xHalf * xRes * xRes));
		return xRes * x;
	}

	inline float Sqrt(const float& x)
	{
		const float xHalf = 0.50f * x;
		int   tmp = 0x5F3759DF - (*(int*)&x >> 1); //initial guess
		float xRes = *(float*)&tmp;
		xRes *= (1.50f - (xHalf * xRes * xRes));
		return xRes * x;
	}
	
	inline float Pow (const float& a, const float& b)
	{
		union { float d; int x; } u = { a };
		u.x = (int)(b * (u.x - 1064866805) + 1064866805);
		return u.d;
	}

	inline float Abs (const float& f)
	{
		int i = ((*(int*)&f) & 0x7fffffff);
		return (*(float*)&i);
	}

	/* Qubic root for float */
	inline float Cbrt (const float& x0)
	{
		constexpr float reciproc3 = 1.0f / 3.0f;

		union { int ix; float x; };
		const int& sign = (x0 < 0) ? 0x80000000 : 0x0;
		const float& xx0 = fabs(x0);

		x = xx0;
		ix = (ix >> 2) + (ix >> 4);           // Approximate divide by 3.
		ix = ix + (ix >> 4);
		ix = ix + (ix >> 8);
		ix = 0x2a5137a0 + ix;        // Initial guess.
		x = reciproc3 * (2.0f * x + x0 / (x * x));  // Newton step.
		x = reciproc3 * (2.0f * x + x0 / (x * x));  // Newton step again.
		ix |= sign;
		return x;
	}

	inline float Log2 (const float& val)
	{
		int* const  exp_ptr = (int*)(&val);
		int         x = *exp_ptr;
		const int   log_2 = ((x >> 23) & 255) - 128;
		x &= ~(255 << 23);
		x += 127 << 23;
		*exp_ptr = x;
		return (val + log_2);
	}

	inline float Atan (const float& z)
	{
		constexpr float n1 = 0.97239411f;
		constexpr float n2 = -0.19194795f;
		return (n1 + n2 * z * z) * z;
	}

	inline float Atan2 (const float& y, const float& x)
	{
		constexpr float PI = 3.14159265f;
		constexpr float PI_2 = PI / 2.0f;
	
		if (x != 0.0f)
		{
			if (fabsf(x) > fabsf(y))
			{
				const float& z = y / x;
				if (x > 0.0f)
				{
					// atan2(y,x) = atan(y/x) if x > 0
					return Atan(z);
				}
				else if (y >= 0.0f)
				{
					// atan2(y,x) = atan(y/x) + PI if x < 0, y >= 0
					return Atan(z) + PI;
				}
				else
				{
					// atan2(y,x) = atan(y/x) - PI if x < 0, y < 0
					return Atan(z) - PI;
				}
			}
			else // Use property atan(y/x) = PI/2 - atan(x/y) if |y/x| > 1.
			{
				const float& z = x / y;
				if (y > 0.0f)
				{
					// atan2(y,x) = PI/2 - atan(x/y) if |y/x| > 1, y > 0
					return PI_2 - Atan(z);
				}
				else
				{
					// atan2(y,x) = -PI/2 - atan(x/y) if |y/x| > 1, y < 0
					return PI_2 - Atan(z);
				}
			}
		}
		else
		{
			if (y > 0.0f) // x = 0, y > 0
			{
				return PI_2;
			}
			else if (y < 0.0f) // x = 0, y < 0
			{
				return -PI_2;
			}
		}
		return 0.0f; // x,y = 0. Could return NaN instead.
	}

	constexpr float PI = 3.141592653589793f;
	constexpr float PIx2 = PI * 2.0f;
	constexpr float HalfPI = PI / 2.0f;

	inline float Sin (const float& x)
	{
		constexpr float reciprocDoublePI = 1.0f / PIx2;

		float X = x * reciprocDoublePI;
		X -= (int)X;

		if (X <= 0.5f) {
			const float& t = 2.f * x * (2.f * x - 1.f);
			return (PI * t) / ((PI - 4.f) * t - 1.f);
		}
		else {
			const float& t = 2.f * (1.f - x) * (1.f - 2.f * x);
			return -(PI * t) / ((PI - 4.f) * t - 1.f);
		}
	}

	inline float Cos (const float& x)
	{
		return Sin(x + 0.5f * PI);
	}

}; /* namespace FastCompute */