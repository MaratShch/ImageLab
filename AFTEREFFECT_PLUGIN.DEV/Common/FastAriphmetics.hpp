#pragma once
#include <cmath>
#include <climits>
#include <algorithm>
#include <type_traits>

#ifndef __NVCC__
#include "FastAriphmeticsAVX.hpp"
#endif

namespace FastCompute
{
	constexpr float PI{ 3.14159265358979323846f };
	constexpr float PIx2{ PI * 2.0f };
	constexpr float HalfPI{ PI / 2.0f };
	constexpr float RECIPROC_PI = 1.0f / PI;

	constexpr float EXP{ 2.718281828459f };
	constexpr float RECIPROC_EXP = 1.0f / EXP;

	constexpr auto CHAR_BITS = 8;

	inline constexpr int Min(const int x, const int y) noexcept
	{
		return y + ((x - y) & ((x - y) >>
			(sizeof(int) * CHAR_BITS - 1)));
	}

	inline constexpr int Max(const int x, const int y) noexcept
	{
		return x - ((x - y) & ((x - y) >>
			(sizeof(int) * CHAR_BITS - 1)));
	}

	template <typename T>
	inline constexpr typename std::enable_if<std::is_integral<T>::value, T>::type Min(const T x, const T y) noexcept
	{	/* find minimal value between 2 fixed point values without branch */
		return y ^ ((x ^ y) & - (x < y));
	}

	template <typename T>
	inline constexpr typename std::enable_if<std::is_integral<T>::value, T>::type Max(const T x, const T y) noexcept
	{   /* find maximal value between 2 fixed point values without branch */
		return x ^ ((x ^ y) & - (x < y));
	}

	template <typename T>
	inline constexpr typename std::enable_if<std::is_floating_point<T>::value, T>::type Min(const T x, const T y) noexcept
	{	
		return std::min(x, y);
	}

	template <typename T>
	inline constexpr typename std::enable_if<std::is_floating_point<T>::value, T>::type Max(const T x, const T y) noexcept
	{   /* find maximal value between 2 fixed point values without branch */
		return std::max(x, y);
	}

	template <typename T>
	inline constexpr T Min3(const T x, const T y, const T z) noexcept
	{	/* find minimal value between 3 fixed point values without branch */
		return Min(Min(x, y), z);
	}

	template <typename T>
	inline constexpr T Max3(const T x, const T y, const T z) noexcept
	{   /* find maximal value between 3 fixed point values without branch */
		return Max(Max(x, y), z);
	}

	namespace FastBitsOperations {
		template <typename T> 
		constexpr typename std::enable_if<std::is_integral<T>::value, T>::type BitSet  { static_cast<T>(1) };

		template <typename T>
		constexpr typename std::enable_if<std::is_integral<T>::value, T>::type BitReset{ static_cast<T>(0) };

		template <typename T>
		inline constexpr typename std::enable_if<std::is_integral<T>::value, T>::type BitOperation(const T op, const T mask, const T val) noexcept
		{
			/* if (op) val |= mask; else val &= ~mask  */
			return ((val & ~mask) | (-op & mask));
		}
	};

	template<typename T>
	inline constexpr auto Abs (const T x) noexcept -> std::enable_if_t<std::is_unsigned<T>::value, T> 
	{
		return x;
	}

	inline constexpr int Abs (const int x) noexcept
	{
		return (x + (x >> sizeof(int) * CHAR_BIT - 1)) ^ (x >> sizeof(int) * CHAR_BIT - 1);
	}

	inline float Abs (const float f) noexcept
	{
		int i = ((*(int*)&f) & 0x7fffffff);
		return (*(float*)&i);
	}

	inline double Abs (const double f) noexcept
	{
		long long i = ((*(long long*)&f) & 0x7fffffffffffffff);
		return (*(double*)&i);
	}


	template<typename T>
	inline constexpr auto Abs(const T x)-> std::enable_if_t<!std::is_unsigned<T>::value, T>
	{
		return std::abs(x);
	}

#if defined(_MSC_VER)
    inline long double Sqrt (const long double x) noexcept
    {
        static_assert(8u == sizeof(long double), "Long double isn't 64 bits");
        const long double xHalf{ 0.50l * x };
        long long int  tmp = 0x5FE6EB50C7B537AAl - (*(long long int*)&x >> 1); //initial guess
        long double    xRes = *(long double*)&tmp;
        xRes *= (1.50l - (xHalf * xRes * xRes));
        return xRes * x;
    }
#endif

	inline double Sqrt(const double x) noexcept
	{
		const double   xHalf{ 0.50 * x };
		long long int  tmp = 0x5FE6EB50C7B537AAl - (*(long long int*)&x >> 1); //initial guess
		double         xRes = *(double*)&tmp;
		xRes *= (1.50 - (xHalf * xRes * xRes));
		return xRes * x;
	}

	inline float Sqrt(const float x) noexcept
	{
		const float xHalf{ 0.50f * x };
		int   tmp = 0x5F3759DF - (*(int*)&x >> 1); //initial guess
		float xRes = *(float*)&tmp;
		xRes *= (1.50f - (xHalf * xRes * xRes));
		return xRes * x;
	}

	template <typename T>
	inline constexpr T Sqrt (const T x) noexcept
	{
		return std::sqrt(x);
	}

	template <typename T>
	inline const typename std::enable_if<std::is_floating_point<T>::value, T>::type Sqrt(const T n, const T eps /* accuracy */) noexcept
    {
		//  Newton-Raphson iterative method
		T x { 1 };
		while (std::abs(x * x - n) > eps)
			x = (x + n / x) / static_cast<T>(2);
		return x;
	}

	inline float InvSqrt(const float x) noexcept
	{
		union { float f; uint32_t u; } y;
		y.f = x;
		y.u = ( 0xBE6EB50Cu - y.u ) >> 1;
		y.f = 0.5f * y.f * (3.0f - x * y.f * y.f);
		return y.f;
	}

	inline double InvSqrt (const double x) noexcept
	{
		union { double d; unsigned long long ull; } y;
		y.d   = x;
		y.ull = (0xBFCDD6A18F6A6F55u - y.ull) >> 1;
		y.d   = 0.5 * y.d * (3.0 - x * y.d * y.d);
		return y.d;
	}

	template <typename T>
	inline constexpr T InvSqrt(const T x) noexcept
	{
		return static_cast<T>(1) / std::sqrt(x);
	}

	inline float Pow (const float a, const float b) noexcept
	{
		union { float d; int x; } u = { a };
		u.x = (int)(b * (u.x - 1064866805) + 1064866805);
		return u.d;
	}

	template <typename T>
	inline constexpr T Pow(const T x) noexcept
	{
		return std::pow(x);
	}

	/* Qubic root for float */
	inline float Cbrt (const float x0) noexcept
	{
		constexpr float reciproc3 = 1.0f / 3.0f;

		union { int ix; float x; };
		const int sign = (x0 < 0) ? 0x80000000 : 0x0;
		const float xx0 = fabs(x0);

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

	template <typename T>
	inline constexpr T Cbrt (const T x) noexcept
	{
		return std::cbrt(x);
	}

	inline float InvCbrt (const float x) noexcept
	{
		constexpr float k1 = 1.7523196760f;
		constexpr float k2 = 1.2509524245f;
		constexpr float k3 = 0.5093818292f;
		constexpr float reciproc3 = 1.0f / 3.0f;

		int i = *(int*) & x;
		i = 0x548c2b4b - i * reciproc3;
		float y = *(float*) & i;
		float c = x * y * y * y;
		y = y * (k1 - c * (k2 - k3 * c));
		c = 1.0f - x * y * y * y;
		y = y * (1.0f + reciproc3 * c);
		return y;
	}

	template <typename T>
	inline constexpr T InvCbrt(const T x) noexcept
	{
		return static_cast<T>(1) / std::cbrt(x);
	}

	inline float Log2 (const float val) noexcept
	{
		int* const  exp_ptr = (int*)(&val);
		int         x = *exp_ptr;
		const int   log_2 = ((x >> 23) & 255) - 128;
		x &= ~(255 << 23);
		x += 127 << 23;
		*exp_ptr = x;
		return (val + log_2);
	}

	template <typename T>
	inline constexpr T Log2(const T x) noexcept
	{
		return std::log2(x);
	}


	// This is a fast approximation to log2()
	inline float Log2f (const float X) noexcept
	{
		int E;
		const float F = frexpf(fabsf(X), &E);
		float Y = 1.23149591368684f;
		Y *= F;
		Y += -4.11852516267426f;
		Y *= F;
		Y += 6.02197014179219f;
		Y *= F;
		Y += -3.13396450166353f;
		Y += E;
		return(Y);
   }

	inline float Log10f (const float x) noexcept
	{
		return Log2f(x) * 0.3010299956639812f;
    }


#if !defined __INTEL_COMPILER
    #pragma warning( push )
    #pragma warning( disable : 4244 )
#endif
	inline int __float_as_int (const float in) noexcept
	{
		union fi { int i; float f; } conv;
		conv.f = in;
		return conv.i;
	}

	inline int __int_as_float(const int in) noexcept
	{
		union fi { int i; float f; } conv;
		conv.i = in;
		return conv.f;
	}

	inline float Log (const float a) noexcept
	{
		float m, r, s, t, i, f;
		int e;
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

	template <typename T>
	inline constexpr T Log (const T x) noexcept
	{
		return std::log(x);
	}


#if !defined __INTEL_COMPILER
    #pragma warning( pop )
#endif

	inline float Acos (float x) noexcept
	{
		const float negate = float(x < 0.f);
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

	template <typename T>
	inline constexpr T Acos(const T x) noexcept
	{
		return std::acos(x);
	}

	inline float Asin (float x) noexcept
	{
		const float negate = float(x < 0.f);
		x = Abs(x);
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

	inline float Atan (const float z) noexcept
	{
		constexpr float n1{ 0.97239411f };
		constexpr float n2{ -0.19194795f };
		return (n1 + n2 * z * z) * z;
	}

	inline float Atan2 (const float y, const float x) noexcept
	{
		constexpr float PI_2 = HalfPI;
	
		if (x != 0.0f)
		{
			if (Abs(x) > Abs(y))
			{
				const float z = y / x;
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
				const float z = x / y;
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

	inline float Atan2d (const float y, const float x) noexcept
	{
		constexpr float coeff = 180.f / FastCompute::PI;
		return (coeff * Atan2(y, x));
	}


    inline double Sin (double x) noexcept
    {
        // Constants
        constexpr double INV_PI_2 = 0.63661977236758134308; // 2/pi
        constexpr double PI_2 = 1.57079632679489661923; // pi/2

        // Range reduction: k = round(x * 2/pi), xr = x - k*(pi/2)
        double k_real = x * INV_PI_2;
        long long k = llround(k_real);
        double xr = x - (double)k * PI_2;

        // Reduced powers
        double x2 = xr * xr;

        // Sin minimax/polynomial coefficients (Remez-like)
        constexpr double p1 = -0.16666667163372;
        constexpr double p2 = 0.00833334773742;
        constexpr double p3 = -0.00019840903752;

        // Evaluate sin polynomial: sin(r) approx
        double sin_poly = xr + x2 * xr * (p1 + x2 * (p2 + x2 * p3));

        // Cos polynomial: cos(r) approx
        constexpr double c1 = -0.49999999725103100312;
        constexpr double c2 = 0.04166662332373906319;
        constexpr double c3 = -0.00138867637746099295;
        double cos_poly = 1.0 + x2 * (c1 + x2 * (c2 + x2 * c3));

        // Quadrant 0..3
        const int q = static_cast<int>(k & 3);

        // Branchless sign/selection tables (corrected)
        // S: multiplier to apply to sin_poly when forming the sin-based candidate
        // C: multiplier to apply to cos_poly when forming the cos-based candidate
        // use_cos: 1.0 when we should pick the cos-based candidate for sin(x), else 0.0
        constexpr double S[4] = { +1.0, +1.0, -1.0, +1.0 };
        constexpr double C[4] = { +1.0, +1.0, -1.0, -1.0 };
        constexpr double use_cos[4] = { 0.0,  1.0,  0.0,  1.0 };

        double sin_candidate = sin_poly * S[q];
        double cos_candidate = cos_poly * C[q];

        // Branchless blend: pick cos_candidate when use_cos==1, otherwise sin_candidate.
        double result = sin_candidate + use_cos[q] * (cos_candidate - sin_candidate);
        return result;
    }

#if defined(_MSC_VER)
    inline long double Sin (long double x) noexcept { return Sin (static_cast<double>(x)); }
#endif

    inline float Sin (float x) noexcept
    {
        // --------------------------------------------
        // Constants for range reduction
        // --------------------------------------------
        constexpr float INV_PI_2 = 0.63661977236758134308f; // 2/pi
        constexpr float PI_2 = 1.57079632679489661923f; // pi/2

        // --------------------------------------------
        // Step 1: Range reduction
        // k = nearest integer to x * 2/pi
        // xr = x - k*(pi/2)
        // --------------------------------------------
        float k_real = x * INV_PI_2;
        int k = static_cast<int>(std::nearbyint(k_real)); // branchless-ish
        float xr = x - static_cast<float>(k) * PI_2;

        // --------------------------------------------
        // Step 2: Reduced powers
        // --------------------------------------------
        float x2 = xr * xr;

        // --------------------------------------------
        // Step 3: Minimax polynomial for sin(x) on [-pi/4, +pi/4]
        // sin(x) ≈ x + x^3*p1 + x^5*p2 + x^7*p3
        // Coefficients optimized for float32
        // --------------------------------------------
        constexpr float p1 = -0.16666667163f;
        constexpr float p2 = 0.00833334774f;
        constexpr float p3 = -0.00019840903f;

        float sin_poly = xr + x2 * xr * (p1 + x2 * (p2 + x2 * p3));

        // --------------------------------------------
        // Cosine polynomial for branchless quadrant use
        // cos(x) ≈ 1 + x^2*c1 + x^4*c2 + x^6*c3
        // --------------------------------------------
        constexpr float c1 = -0.5f;
        constexpr float c2 = 0.041666623f;
        constexpr float c3 = -0.001388676f;

        float cos_poly = 1.0f + x2 * (c1 + x2 * (c2 + x2 * c3));

        // --------------------------------------------
        // Step 4: Branchless quadrant logic
        // --------------------------------------------
        int q = k & 3;

        // Corrected sign tables for float
        constexpr float S[4] = { +1.0f, +1.0f, -1.0f, +1.0f };
        constexpr float C[4] = { +1.0f, +1.0f, -1.0f, -1.0f };
        constexpr float use_cos[4] = { 0.0f,  1.0f,  0.0f,  1.0f };

        float sin_candidate = sin_poly * S[q];
        float cos_candidate = cos_poly * C[q];

        float result = sin_candidate + use_cos[q] * (cos_candidate - sin_candidate);

        return result;
    }


    template <typename T>
	inline const T Sin(const T x) noexcept
	{
		return std::sin(x);
	}

    inline double Cos (double x) noexcept
    {
        // Constants
        constexpr double INV_PI_2 = 0.63661977236758134308; // 2/pi
        constexpr double PI_2 = 1.57079632679489661923; // pi/2

                                                        // Range reduction: k = round(x * 2/pi), xr = x - k*(pi/2)
        double k_real = x * INV_PI_2;
        long long k = llround(k_real);
        double xr = x - static_cast<double>(k) * PI_2;

        double x2 = xr * xr;

        // Sin polynomial
        constexpr double s1 = -0.16666667163372;
        constexpr double s2 = 0.00833334773742;
        constexpr double s3 = -0.00019840903752;
        double sin_poly = xr + x2 * xr * (s1 + x2 * (s2 + x2 * s3));

        // Cos polynomial
        constexpr double c1 = -0.49999999725103100312;
        constexpr double c2 = 0.04166662332373906319;
        constexpr double c3 = -0.00138867637746099295;
        double cos_poly = 1.0 + x2 * (c1 + x2 * (c2 + x2 * c3));

        int q = static_cast<int>(k & 3);

        // Branchless tables for cos(x)
        constexpr double S[4] = { +1.0, +1.0, -1.0, +1.0 }; // for sin_poly
        constexpr double C[4] = { +1.0, +1.0, -1.0, -1.0 }; // for cos_poly
        constexpr double use_cos[4] = { 1.0, 0.0, 1.0, 0.0 };     // pick cos_poly for cos(x)

        double sin_candidate = sin_poly * S[q];
        double cos_candidate = cos_poly * C[q];

        double result = sin_candidate + use_cos[q] * (cos_candidate - sin_candidate);
        return result;
    }

#if defined(_MSC_VER)
    inline long double Cos (long double x) noexcept { return Cos(static_cast<double>(x)); }
#endif

    inline float Cos (float x) noexcept
    {
        constexpr float INV_PI_2 = 0.63661977236758134308f;
        constexpr float PI_2 = 1.57079632679489661923f;

        float k_real = x * INV_PI_2;
        int k = static_cast<int>(std::nearbyint(k_real));
        float xr = x - static_cast<float>(k) * PI_2;

        float x2 = xr * xr;

        // Sin polynomial
        constexpr float s1 = -0.16666667163f;
        constexpr float s2 = 0.00833334774f;
        constexpr float s3 = -0.00019840903f;
        float sin_poly = xr + x2 * xr * (s1 + x2 * (s2 + x2 * s3));

        // Cos polynomial
        constexpr float c1 = -0.5f;
        constexpr float c2 = 0.041666623f;
        constexpr float c3 = -0.001388676f;
        float cos_poly = 1.0f + x2 * (c1 + x2 * (c2 + x2 * c3));

        int q = k & 3;

        // Branchless tables for cos(x)
        constexpr float S[4] = { +1.0f, +1.0f, -1.0f, +1.0f };
        constexpr float C[4] = { +1.0f, +1.0f, -1.0f, -1.0f };
        constexpr float use_cos[4] = { 1.0f, 0.0f, 1.0f, 0.0f };

        float sin_candidate = sin_poly * S[q];
        float cos_candidate = cos_poly * C[q];

        float result = sin_candidate + use_cos[q] * (cos_candidate - sin_candidate);
        return result;
    }

	template <typename T>
	inline const T Cos (const T x) noexcept
	{
		return std::cos (x);
	}


    inline void SinCos (double x, double& sin_out, double& cos_out) noexcept
    {
	   // Constants
		constexpr double INV_PI_2 = 0.63661977236758134308; // 2/pi

		// Split pi/2 into high+low to improve reduction accuracy:
		// values chosen so PI_2_H + PI_2_L == pi/2 with PI_2_H carries the high 53-/-bits
		constexpr double PI_2_H = 1.5707963267948966;            // high part
		constexpr double PI_2_L = 6.123031769111886e-17;        // low part (pi/2 - PI_2_H)

		// Range reduction
		double k_real = x * INV_PI_2;
		long long k = llround(k_real);                 // nearest integer
		// Improved xr: do two subtractions to preserve precision
		double xr = (x - static_cast<double>(k) * PI_2_H) - static_cast<double>(k) * PI_2_L;

		double x2 = xr * xr;

		// Sin polynomial (minimax, order 7)
		constexpr double p1 = -0.16666667163372;
		constexpr double p2 =  0.00833334773742;
		constexpr double p3 = -0.00019840903752;
	#if defined(__cpp_lib_math_fma) || (defined(__FMA__) || defined(__AVX__))
		// use fma if available: (((p3*x2 + p2)*x2 + p1)*x2)*xr + xr  (but written carefully)
		double t = std::fma(p3, x2, p2);
		t = std::fma(t, x2, p1);
		t = std::fma(t, x2, 1.0);
		double sin_poly = std::fma(t, x2, xr); // close Horner with fma
	#else
		double sin_poly = xr + x2 * xr * (p1 + x2 * (p2 + x2 * p3));
	#endif

		// Cos polynomial (order 6)
		constexpr double c1 = -0.49999999725103100312;
		constexpr double c2 =  0.04166662332373906319;
		constexpr double c3 = -0.00138867637746099295;
	#if defined(__cpp_lib_math_fma) || (defined(__FMA__) || defined(__AVX__))
		double tc = std::fma(c3, x2, c2);
		tc = std::fma(tc, x2, c1);
		tc = std::fma(tc, x2, 1.0);
		double cos_poly = tc;
	#else
		double cos_poly = 1.0 + x2 * (c1 + x2 * (c2 + x2 * c3));
	#endif

		// Quadrant
		int q = static_cast<int>(k & 3);

		// Corrected branchless tables (see explanation above)
		constexpr double sin_sign[4] = { +1.0, -1.0, -1.0, +1.0 };
		constexpr double cos_sign[4] = { +1.0, +1.0, -1.0, -1.0 };
		constexpr bool   swap_tbl[4] = { false, true, false, true };

		// Apply signs
		double s = sin_poly * sin_sign[q];
		double c = cos_poly * cos_sign[q];

		// Branchless swap
		sin_out = swap_tbl[q] ? c : s;
		cos_out = swap_tbl[q] ? s : c;
	}

#if defined(_MSC_VER)
    inline void SinCos (long double x, long double& sin_out, long double& cos_out) noexcept
    {
        static_assert(sizeof(double) == sizeof(long double), "long double size is not equal to double size!");
        SinCos (static_cast<double>(x), sin_out, cos_out);
    }
#endif

    inline void SinCos (float x, float& sin_out, float& cos_out) noexcept
    {
		constexpr float INV_PI_2 = 0.63661977236758134308f; // 2/pi
		constexpr float PI_2     = 1.57079632679489661923f; // pi/2

		float k_real = x * INV_PI_2;
		int k = static_cast<int>(std::nearbyint(k_real));
		float xr = x - static_cast<float>(k) * PI_2;
		float x2 = xr * xr;

		constexpr float p1 = -0.16666667163f;
		constexpr float p2 =  0.00833334774f;
		constexpr float p3 = -0.00019840903f;
		float sin_poly = xr + x2 * xr * (p1 + x2 * (p2 + x2 * p3));

		constexpr float c1 = -0.5f;
		constexpr float c2 =  0.041666623f;
		constexpr float c3 = -0.001388676f;
		float cos_poly = 1.0f + x2 * (c1 + x2 * (c2 + x2 * c3));

		int q = k & 3;

		constexpr float sin_sign[4] = { +1.0f, -1.0f, -1.0f, +1.0f };
		constexpr float cos_sign[4] = { +1.0f, +1.0f, -1.0f, -1.0f };
		constexpr bool  swap_tbl[4]  = { false, true, false, true };

		float s = sin_poly * sin_sign[q];
		float c = cos_poly * cos_sign[q];

		sin_out = swap_tbl[q] ? c : s;
		cos_out = swap_tbl[q] ? s : c;
        return;
    }

    template <typename T>
    inline void SinCos (T x, T& sin_out, T& cos_out) noexcept
    {
        sin_out = std::sin(x);
        cos_out = std::cos(x);
        return;
    }


	template <typename T>
	inline const typename std::enable_if<std::is_floating_point<T>::value, T>::type Exp (const T fVal) noexcept
	{
		constexpr T one{ 1 };
#ifdef FAST_COMPUTE_EXTRA_PRECISION
		constexpr T reciproc1024{ 1.0 / 1024.0 };
		T x = one + fVal * reciproc1024;
		x *= x; x *= x; x *= x; x *= x;
		x *= x; x *= x; x *= x; x *= x;
		x *= x; x *= x;
#else
		constexpr T reciproc256{ 1.0 / 256.0 };
		T x = one + fVal * reciproc256;
		x *= x; x *= x; x *= x; x *= x;
		x *= x; x *= x; x *= x; x *= x;
#endif
		return x;
	}

	template <typename T>
	inline const typename std::enable_if<std::is_floating_point<T>::value, T>::type	Sinh (const T x) noexcept
	{
		constexpr T half{ 0.5 };
		return half * (Exp(x) - Exp(-x));
	}

	template <typename T>
	inline const typename std::enable_if<std::is_floating_point<T>::value, T>::type	Cosh (const T x) noexcept
	{
		constexpr T half{ 0.5 };
		return half * (Exp(x) + Exp(-x));
	}

#ifndef _COMPUTE_PRECISE_TANH_
/* 
	Fast computation of hyperbolic tangent. Rational approximation with clamping.
	Maximum absolute errror = 2.77074604e-3 @ +/-3.29019976
*/
	inline const float Tanh (const float x) noexcept
	{
		constexpr float n0 = -8.73291016e-1f; // -0x1.bf2000p-1
		constexpr float n1 = -2.76107788e-2f; // -0x1.c46000p-6
		constexpr float d0 = 2.79589844e+0f;  //  0x1.65e000p+1
		const  float x2 = x * x;
		const float num = fmaf(n0, x2, n1);
		const float den = x2 + d0;
		const float quot = num / den;
		const float res = fmaf(quot, x, x);
		return fminf(fmaxf(res, -1.0f), 1.0f);
	}
#else // _COMPUTE_PRECISE_TANH_
/* 
	Fast computation of hyperbolic tangent. Rational approximation with clamping
	of the argument. Maximum absolute error = 1.98537030e-5, maximum relative
	error = 1.98540995e-5, maximum ulp error = 333.089863.
*/
	inline const float Tanh (const float x) // 10 operations
	{
		constexpr float cutoff = 5.76110792f; //  0x1.70b5fep+2
		constexpr float n0 = -1.60153955e-4f; // -0x1.4fde00p-13
		constexpr float n1 = -9.34448242e-1f; // -0x1.de7000p-1
		constexpr float n2 = -2.19176636e+1f; // -0x1.5eaec0p+4
		constexpr float d0 = 2.90915985e+1f; //  0x1.d17730p+4
		constexpr float d1 = 6.57667847e+1f; //  0x1.071130p+6
		float y = fminf(fmaxf(x, -cutoff), cutoff);
		float y2 = y * y;
		float num = fmaf(fmaf(n0, y2, n1), y2, n2) * y2;
		float den = fmaf(y2 + d0, y2, d1);
		float quot = num / den;
		float res = fmaf(quot, y, y);
		return res;
	}
#endif // _COMPUTE_PRECISE_TANH_


	template <typename T>
	inline const typename std::enable_if<std::is_floating_point<T>::value, T>::type Sigmoid (const T fVal) noexcept
	{
		constexpr T one{ 1 };
		return one / (one + Exp(-fVal));
	}

	template <typename T>
	inline constexpr T FastSigmoid (const T& fVal) noexcept
	{
		return fVal / (1 + Abs(fVal));
	}


    template<typename T>
    inline constexpr typename std::enable_if<std::is_integral<T>::value || std::is_floating_point<T>::value, T>::type Sign(const T val) noexcept
    {
        return (static_cast<T>(0) < val) - (val < static_cast<T>(0));
    }

    template<typename T> // Computes the dot product of two 2D vectors
    inline constexpr typename std::enable_if<std::is_integral<T>::value || std::is_floating_point<T>::value, T>::type Dot(const T x1, const T x2, const T y1, const T y2) noexcept
    {
        return x1 * x2 + y1 * y2;
    }

    template<typename T> // Computes the determinant of a 2x2 matrix [[a, b], [c, d]]
    inline constexpr typename std::enable_if<std::is_integral<T>::value || std::is_floating_point<T>::value, T>::type Determinant2x2 (const T a, const T b, const T c, const T d) noexcept
    {
        return a * d - b * c;
    }

    template <typename T>
    inline constexpr typename std::enable_if<std::is_floating_point<T>::value, T>::type VectorNorm (const T x, const T y) noexcept
    {
        return Sqrt (x * x + y * y);
    }

} /* namespace FastCompute */