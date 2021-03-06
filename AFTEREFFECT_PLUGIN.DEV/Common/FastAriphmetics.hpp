#pragma once
#include <cmath>

namespace FastCompute
{
	constexpr float PI = 3.14159265358979323846f;
	constexpr float PIx2 = PI * 2.0f;
	constexpr float HalfPI = PI / 2.0f;

	constexpr auto CHAR_BITS = 8;

	inline constexpr int Min(const int& x, const int& y) noexcept
	{
		return y + ((x - y) & ((x - y) >>
			(sizeof(int) * CHAR_BITS - 1)));
	}

	inline constexpr int Max(const int& x, const int& y) noexcept
	{
		return x - ((x - y) & ((x - y) >>
			(sizeof(int) * CHAR_BITS - 1)));
	}

	template <typename T>
	inline constexpr typename std::enable_if<std::is_integral<T>::value, T>::type Min(const T& x, const T& y) noexcept
	{	/* find minimal value between 2 fixed point values without branch */
		return y ^ ((x ^ y) & -(x < y));
	}

	template <typename T>
	inline constexpr typename std::enable_if<std::is_integral<T>::value, T>::type Max(const T& x, const T& y) noexcept
	{   /* find maximal value between 2 fixed point values without branch */
		return x ^ ((x ^ y) & -(x < y));
	}

	template <typename T>
	inline constexpr typename std::enable_if<std::is_integral<T>::value, T>::type Min3(const T& x, const T& y, const T& z) noexcept
	{	/* find minimal value between 3 fixed point values without branch */
		return Min(Min(x,y), z);
	}

	template <typename T>
	inline constexpr typename std::enable_if<std::is_integral<T>::value, T>::type Max3(const T& x, const T& y, const T& z) noexcept
	{   /* find maximal value between 3 fixed point values without branch */
		return Max(Max(x, y), z);
	}

	inline double Sqrt(const double& x) noexcept
	{
		const double   xHalf = 0.50 * x;
		long long int  tmp = 0x5FE6EB50C7B537AAl - (*(long long int*)&x >> 1); //initial guess
		double         xRes = *(double*)&tmp;
		xRes *= (1.50 - (xHalf * xRes * xRes));
		return xRes * x;
	}

	inline float Sqrt(const float& x) noexcept
	{
		const float xHalf = 0.50f * x;
		int   tmp = 0x5F3759DF - (*(int*)&x >> 1); //initial guess
		float xRes = *(float*)&tmp;
		xRes *= (1.50f - (xHalf * xRes * xRes));
		return xRes * x;
	}
	
	inline float Pow (const float& a, const float& b) noexcept
	{
		union { float d; int x; } u = { a };
		u.x = (int)(b * (u.x - 1064866805) + 1064866805);
		return u.d;
	}

	inline float Abs (const float& f) noexcept
	{
		int i = ((*(int*)&f) & 0x7fffffff);
		return (*(float*)&i);
	}

	/* Qubic root for float */
	inline float Cbrt (const float& x0) noexcept
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

	inline float Log2 (const float& val) noexcept
	{
		int* const  exp_ptr = (int*)(&val);
		int         x = *exp_ptr;
		const int   log_2 = ((x >> 23) & 255) - 128;
		x &= ~(255 << 23);
		x += 127 << 23;
		*exp_ptr = x;
		return (val + log_2);
	}

#if !defined __INTEL_COMPILER
    #pragma warning( push )
    #pragma warning( disable : 4244 )
#endif
	inline int __float_as_int (const float& in) noexcept
	{
		union fi { int i; float f; } conv;
		conv.f = in;
		return conv.i;
	}

	inline int __int_as_float(const int& in) noexcept
	{
		union fi { int i; float f; } conv;
		conv.i = in;
		return conv.f;
	}

	inline float Log (const float& a) noexcept
	{
		float m, r, s, t, i, f;
		int32_t e;

		e = (__float_as_int(a) - 0x3f2aaaab) & 0xff800000;
		m = __int_as_float(__float_as_int(a) - e);
		i = static_cast<float>(e) * 1.19209290e-7f; 
		f = m - 1.0f;
		s = f * f;
		r = fmaf(0.230836749f, f, -0.279208571f);
		t = fmaf(0.331826031f, f, -0.498910338f);
		r = fmaf(r, s, t);
		r = fmaf(r, s, f);
		r = fmaf(i, 0.693147182f, r);
		return r;
	}
#if !defined __INTEL_COMPILER
    #pragma warning( pop )
#endif

	inline float Acos (float x) noexcept
	{
		const float& negate = float(x < 0.f);
		x = abs(x);
		float ret = -0.0187293f;
		ret = ret * x;
		ret = ret + 0.0742610f;
		ret = ret * x;
		ret = ret - 0.2121144f;
		ret = ret * x;
		ret = ret + 1.5707288f;
		ret = ret * Sqrt(1.0f - x);
		ret = ret - 2.f * negate * ret;
		return negate * PI + ret;
	}

	inline float Asin (float x) noexcept
	{
		const float& negate = float(x < 0.f);
		x = abs(x);
		float ret = -0.0187293f;
		ret *= x;
		ret += 0.0742610f;
		ret *= x;
		ret -= 0.2121144f;
		ret *= x;
		ret += 1.5707288f;
		ret = HalfPI - Sqrt(1.0f - x) * ret;
		return ret - 2.f * negate * ret;
	}

	inline float Atan (const float& z) noexcept
	{
		constexpr float n1 = 0.97239411f;
		constexpr float n2 = -0.19194795f;
		return (n1 + n2 * z * z) * z;
	}

	inline float Atan2 (const float& y, const float& x) noexcept
	{
		constexpr float PI_2 = HalfPI;
	
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

#define EXTRA_PRECISION
	template<typename T>
	inline T Cos (T x) noexcept
	{
		constexpr T DoublePI = T(2.) * T(3.14159265358979323846264);
		constexpr T tp = T(1.) / DoublePI;
		x *= tp;
		x -= T(.25) + std::floor(x + T(.25));
		x *= T(16.) * (std::abs(x) - T(.5));
#ifdef EXTRA_PRECISION
		x += T(.225) * x * (std::abs(x) - T(1.));
#endif
		return x;
	}

	inline float Sin (float x) noexcept
	{
		constexpr float HalfPI = 3.14159265358979323846264f / 2.f;
		return Cos(x + HalfPI);
	}

	inline double Sin (double x) noexcept
	{
		constexpr double HalfPI = 3.14159265358979323846264 / 2.;
		return Cos(x + HalfPI);
	}

	inline long double Sin(long double x) noexcept
	{
		constexpr long double HalfPI = 3.14159265358979323846264l / 2.l;
		return Cos(x + HalfPI);
	}

	namespace AVX2
	{
		inline __m256 mm256_fmaf(__m256 a, __m256 b, __m256 c) noexcept
		{
			return _mm256_add_ps(_mm256_mul_ps(a, b), c);
		}

		/* https://stackoverflow.com/questions/39821367/very-fast-approximate-logarithm-natural-log-function-in-c  */
		inline __m256 Log(__m256 a) noexcept
		{
			__m256i aInt = *(__m256i*)(&a);
			__m256i e = _mm256_sub_epi32(aInt, _mm256_set1_epi32(0x3f2aaaab));
			e = _mm256_and_si256(e, _mm256_set1_epi32(0xff800000));

			__m256i subtr = _mm256_sub_epi32(aInt, e);
			__m256 m = *(__m256*)&subtr;

			__m256 i = _mm256_mul_ps(_mm256_cvtepi32_ps(e), _mm256_set1_ps(1.19209290e-7f));// 0x1.0p-23
																							/* m in [2/3, 4/3] */
			__m256 f = _mm256_sub_ps(m, _mm256_set1_ps(1.0f));
			__m256 s = _mm256_mul_ps(f, f);
			/* Compute log1p(f) for f in [-1/3, 1/3] */
			__m256 r = mm256_fmaf(_mm256_set1_ps(0.230836749f), f, _mm256_set1_ps(-0.279208571f));// 0x1.d8c0f0p-3, -0x1.1de8dap-2
			__m256 t = mm256_fmaf(_mm256_set1_ps(0.331826031f), f, _mm256_set1_ps(-0.498910338f));// 0x1.53ca34p-2, -0x1.fee25ap-2

			r = mm256_fmaf(r, s, t);
			r = mm256_fmaf(r, s, f);
			r = mm256_fmaf(i, _mm256_set1_ps(0.693147182f), r);  // 0x1.62e430p-1 // log(2)
			return r;
		}

	} /* namespace AVX2 */

}; /* namespace FastCompute */