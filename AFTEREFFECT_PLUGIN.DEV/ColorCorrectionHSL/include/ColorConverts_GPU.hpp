#pragma once

#include "FastAriphmetics.hpp"
#include "ColorCorrectionGPU.hpp"

namespace GPU
{
	constexpr float flt_EPSILON = 1.19209290e-07F;
	constexpr float flt_MAX = 3.402823466e+38F;
	constexpr float flt_MIN = 1.17549435e-38F;

	template <typename T>
	inline __device__ T MIN_VALUE(const T& a, const T& b) { return ((a < b) ? a : b); }

	template <typename T>
	inline __device__ T MAX_VALUE(const T& a, const T& b) { return ((a > b) ? a : b); }

	template <typename T>
	inline __device__ T MIN3_VALUE(const T& a, const T& b, const T& c) { return (a < b) ? MIN_VALUE(a, c) : MIN_VALUE(b, c); }

	template <typename T>
	inline __device__ T MAX3_VALUE(const T& a, const T& b, const T& c) { return (a > b) ? MAX_VALUE(a, c) : MAX_VALUE(b, c); }

	template <typename T>
	inline __device__ T CLAMP_VALUE(const T& val, const T& min, const T& max)
	{
		return ((val < min) ? min : ((val > max) ? max : val));
	}


	template<typename T>
	inline __device__ const typename std::enable_if<std::is_floating_point<T>::value, T>::type
		restore_rgb_channel_value(const T& t1, const T& t2, const T& t3)
	{
		T val;

		const T t3mult3 = t3 * 3.0f;

		if (t3mult3 < 0.50f)
			val = t1 + (t2 - t1) * 6.0f * t3;
		else if (t3mult3 < 1.50f)
			val = t2;
		else if (t3mult3 < 2.0f)
			val = t1 + (t2 - t1) * (0.6660f - t3) * 6.0f;
		else
			val = t1;
		return val;
	}


	template<typename T>
	inline __device__ void sRgb2hsl(const T& R, const T& G, const T& B, T& H, T& S, T& L) noexcept
	{
		/* start convert RGB to HSL color space */
		T const& maxVal = MAX3_VALUE(R, G, B);
		T const& minVal = MIN3_VALUE(R, G, B);
		T const sumMaxMin = maxVal + minVal;
		T const luminance = sumMaxMin * (T)50.0; /* luminance value in percents = 100 * (max + min) / 2 */
		T hue, saturation;

		if (maxVal == minVal)
		{
			saturation = hue = (T)(0.0);
		}
		else
		{
			T const& subMaxMin = maxVal - minVal;
			saturation = ((T)100.0 * subMaxMin) / ((luminance < (T)50.0) ? sumMaxMin : ((T)2.0 - sumMaxMin));
			if (R == maxVal)
				hue = ((T)60.0 * (G - B)) / subMaxMin;
			else if (G == maxVal)
				hue = ((T)60.0 * (B - R)) / subMaxMin + (T)120.0;
			else
				hue = ((T)60.0 * (R - G)) / subMaxMin + (T)240.0;
		}

		H = hue;
		S = saturation;
		L = luminance;

		return;
	}


	template<typename T>
	inline __device__ void hsl2sRgb(const T& H, const T& S, const T& L, T& R, T& G, T& B) noexcept
	{
		constexpr T denom = 1.0 / 1e7;
		constexpr T reciproc3 = 1.0 / 3.0;

		T rR, gG, bB;

		/* back convert to RGB space */
		if (denom >= S)
		{
			rR = gG = bB = L;
		}
		else
		{
			T tmpVal1, tmpVal2;
			tmpVal2 = (L < (T)0.50) ? (L * ((T)1.0 + S)) : (L + S - (L * S));
			tmpVal1 = (T)2.0 * L - tmpVal2;

			T const& tmpG = H;
			T  tmpR = H + reciproc3;
			T  tmpB = H - reciproc3;

			tmpR -= ((tmpR > (T)1.0) ? (T)1.0 : (T)0.0);
			tmpB += ((tmpB < (T)0.0) ? (T)1.0 : (T)0.0);

			rR = restore_rgb_channel_value(tmpVal1, tmpVal2, tmpR);
			gG = restore_rgb_channel_value(tmpVal1, tmpVal2, tmpG);
			bB = restore_rgb_channel_value(tmpVal1, tmpVal2, tmpB);
		}

		R = rR;
		G = gG;
		B = bB;

		return;
	}


	template<typename T>
	inline __device__ void sRgb2hsv(const T& R, const T& G, const T& B, T& H, T& S, T& V) noexcept
	{
		constexpr T denom = (T)(1.0 / 1e7);

		/* start convert RGB to HSV color space */
		T const& maxVal = MAX3_VALUE(R, G, B);
		T const& minVal = MIN3_VALUE(R, G, B);
		T const& C = maxVal - minVal;

		T const& value = maxVal;
		T hue, saturation;

		if (denom < C)
		{
			if (R == maxVal)
			{
				hue = (G - B) / C;
				if (G < B)
					hue += (T)6;
			}
			else if (G == maxVal)
				hue = (T)2 + (B - R) / C;
			else
				hue = (T)4 + (R - G) / C;

			hue *= (T)60;
			saturation = C / maxVal;
		}
		else
		{
			hue = saturation = (T)0;
		}

		H = hue;
		S = saturation;
		V = value;

		return;
	}


	template<typename T>
	inline __device__ void hsv2sRgb (const T& H, const T& S, const T& V, T& R, T& G, T& B) noexcept
	{
		T const& c = S * V;
		T const& minV = V - c;

		constexpr T reciproc360 = (T)(1.0 / 360.0);
		constexpr T reciproc60  = (T)(1.0 / 60.0);

		T const& h = (H - (T)360 * floor(H * reciproc360)) * reciproc60;
		T const& X = c * ((T)1 - fabs(h - (T)2 * floor(h * (T)0.5) - (T)1));
		const int& hh = static_cast<int const>(h);

		float fR, fG, fB;

		if (hh & 0x1)
		{
			if (1 == hh)
			{
				fR = minV + X; fG = minV + c; fB = minV; /* 1 */
			}
			else if (3 == hh)
			{
				fR = minV; fG = minV + X; fB = minV + c; /* 3 */
			}
			else
			{
				fR = minV + c; fG = minV; fB = minV + X; /* 5 */
			}
		}
		else
		{
			if (0 == hh)
			{
				fR = minV + c; fG = minV + X; fB = minV; /* 0 */
			}
			else if (2 == hh)
			{
				fR = minV; fG = minV + c; fB = minV + X; /* 2 */
			}
			else
			{
				fR = minV + X; fG = minV; fB = minV + c; /* 4 */
			}
		}

		R = fR;
		G = fG;
		B = fB;

		return;
	}


	template<typename T>
	inline __device__ void sRgb2hsi (const T& R, const T& G, const T& B, T& H, T& S, T& I) noexcept
	{
		constexpr T reciproc3 = (T)(1.0 / 3.0);
		constexpr T denom = (T)(1.0 / 1e7);
		constexpr T reciprocPi180 = (T)(180.0 / 3.14159265358979323846);

		T i = (R + G + B) * reciproc3;
		T h, s;

		if (i > denom)
		{
			T const& alpha = (T)0.5 * ((T)2 * R - G - B) + denom;
			T const& beta = (T)0.8660254037 * (G - B) + denom;
			s = (T)1.0 - MIN3_VALUE(R, G, B) / i;
			h = atan2(beta, alpha) * reciprocPi180;
			if (h < 0)
				h += (T)360;
		}
		else
		{
			i = h = s = (T)0;
		}

		H = h;
		S = s;
		I = i;

		return;
	}


	template<typename T>
	inline __device__ void hsi2sRgb(const T& H, const T& S, const T& I, T& R, T& G, T& B) noexcept
	{
		constexpr T PiDiv180 = (T)(3.14159265358979323846 / 180.0);
		constexpr T reciproc360 = (1.0 / 360.0);
		constexpr T denom = (T)(1.0 / 1e7);

		T h = H - (T)360 * floor(H * reciproc360);
		const T& val1 = I * ((T)1.0 - S);
		const T& tripleI = (T)3.0 * I;

		if (h < (T)120)
		{
			const T& cosTmp = cos(((T)60 - h) * PiDiv180);
			const T& cosDiv = ((T)0. == cosTmp) ? denom : cosTmp;
			B = val1;
			R = I * (1.f + S * cos(h * PiDiv180) / cosDiv);
			G = tripleI - R - B;
		}
		else if (h < (T)240)
		{
			h -= (T)120;
			const T& cosTmp = cos(((T)60 - h) * PiDiv180);
			const T& cosDiv = ((T)0. == cosTmp) ? denom : cosTmp;
			R = val1;
			G = I * (1.f + S * cos(h * PiDiv180) / cosDiv);
			B = tripleI - R - G;
		}
		else
		{
			h -= (T)240;
			const T& cosTmp = cos(((T)60 - h) * PiDiv180);
			const T& cosDiv = ((T)0. == cosTmp) ? denom : cosTmp;
			G = val1;
			B = I * ((T)1.0 + S * cos(h * PiDiv180) / cosDiv);
			R = tripleI - G - B;
		}

		return;
	}


	template<typename T>
	inline __device__ void sRgb2hsp(const T& R, const T& G, const T& B, T& H, T& S, T& P) noexcept
	{
		constexpr T Pr = (T)0.2990;
		constexpr T Pg = (T)0.5870;
		constexpr T Pb = (T)1.0 - Pr - Pg; /* 0.1140f */;
		constexpr T div1on6 = (T)(1.0 / 6.0);
		constexpr T div2on6 = (T)(2.0 / 6.0);
		constexpr T div4on6 = (T)(4.0 / 6.0);

		T const& maxVal = MAX3_VALUE(R, G, B);

		P = sqrt(R * R * Pr + G * G * Pg + B * B * Pb);

		if (R == G && R == B) {
			H = S = (T)0;
		}
		else {
			if (R == maxVal)
			{   //  R is largest
				if (B >= G) {
					H = (T)1.0 - div1on6 * (B - G) / (R - G);
					S = (T)1.0 - G / R;
				}
				else {
					H = div1on6 * (G - B) / (R - B);
					S = (T)1.0 - B / R;
				}
			}
			else if (G == maxVal)
			{   //  G is largest
				if (R >= B) {
					H = div2on6 - div1on6 * (R - B) / (G - B);
					S = (T)1.0 - B / G;
				}
				else {
					H = div2on6 + div1on6 * (B - R) / (G - R);
					S = (T)1.0 - R / G;
				}
			}
			else {   //  B is largest
				if (G >= R) {
					H = div4on6 - div1on6 * (G - R) / (B - R);
					S = (T)1.0 - R / B;
				}
				else {
					H = div4on6 + div1on6 * (R - G) / (B - G);
					S = (T)1.0 - G / B;
				}
			}
		}
		return;
	}


	template<typename T>
	inline __device__ void hsp2sRgb (const T& H, const T& S, const T& P, T& R, T& G, T& B) noexcept
	{
		constexpr T Pr = (T)0.2990;
		constexpr T Pg = (T)0.5870;
		constexpr T Pb = (T)1.0 - Pr - Pg; /* 0.1140f */;
		constexpr T OneDivSix = (T)(1.0 / 6.0);
		constexpr T TwoDivSix = (T)(2.0 / 6.0);
		constexpr T TreDivSix = (T)(3.0 / 6.0);
		constexpr T ForDivSix = (T)(4.0 / 6.0);
		constexpr T FivDivSix = (T)(5.0 / 6.0);

		const T& minOverMax = (T)1.0 - S;
		T part;

		if (minOverMax > (T)0.0)
		{
			if (H < OneDivSix) {   //  R>G>B
				const T& h = 6.f * H;
				constexpr T one = (T)1.0;
				part = one + h * (one / minOverMax - one);
				B = P / sqrt(Pr / minOverMax / minOverMax + Pg * part * part + Pb);
				R = B / minOverMax;
				G = B + h * (R - B);
			}
			else if (H < TwoDivSix) {   //  G>R>B
				const T& h = (T)6.0 * (-H + TwoDivSix);
				part = (T)1.0 + h  * ((T)1.0 / minOverMax - (T)1.0);
				B = P / sqrt(Pg / minOverMax / minOverMax + Pr * part * part + Pb);
				G = B / minOverMax;
				R = B + h * (G - B);
			}
			else if (H < TreDivSix) {   //  G>B>R
				const T& h = (T)6.0 * (H - TwoDivSix);
				part = (T)1.0 + h * ((T)1.0 / minOverMax - (T)1.0);
				R = P / sqrt(Pg / minOverMax / minOverMax + Pb * part * part + Pr);
				G = R / minOverMax;
				B = R + h * (G - R);
			}
			else if (H < ForDivSix) {   //  B>G>R
				const T& h = (T)6.0 * (-H + ForDivSix);
				part = (T)1.0 + h * ((T)1.0 / minOverMax - (T)1.0);
				R = P / sqrt(Pb / minOverMax / minOverMax + Pg * part * part + Pr);
				B = R / minOverMax;
				G = R + h * (B - R);
			}
			else if (H < FivDivSix) {   //  B>R>G
				const T& h = (T)6.0 * (H - ForDivSix);
				part = (T)1.0 + h * ((T)1.0 / minOverMax - (T)1.0);
				G = P / sqrt(Pb / minOverMax / minOverMax + Pr * part * part + Pg);
				B = G / minOverMax;
				R = G + h * (B - G);
			}
			else {   //  R>B>G
				const T& h = (T)6.0 * (-H + (T)1.0);
				part = (T)1.0 + h * ((T)1.0 / minOverMax - (T)1.0);
				G = P / sqrt(Pr / minOverMax / minOverMax + Pb * part * part + Pg);
				R = G / minOverMax;
				B = G + h * (R - G);
			}
		}
		else
		{
			if (H < OneDivSix) {   //  R>G>B
				const T& h = (T)6.0 * H;
				R = sqrt(P * P / (Pr + Pg * h * h));
				G = R * h;
				B = (T)0;
			}
			else if (H < TwoDivSix) {   //  G>R>B
				const T& h = (T)6.0 * (-H + TwoDivSix);
				G = sqrt(P * P / (Pg + Pr * h * h));
				R = G * h;
				B = (T)0;
			}
			else if (H < TreDivSix) {   //  G>B>R
				const T& h = (T)6.0 * (H - TwoDivSix);
				G = sqrt(P * P / (Pg + Pb * h * h));
				B = G * h;
				R = (T)0;
			}
			else if (H < ForDivSix) {   //  B>G>R
				const T& h = (T)6.0 * (-H + ForDivSix);
				B = sqrt(P * P / (Pb + Pg * h * h));
				G = B * h;
				R = (T)0;
			}
			else if (H < FivDivSix) {   //  B>R>G
				const T& h = (T)6.0 * (H - ForDivSix);
				B = sqrt(P * P / (Pb + Pr * h * h));
				R = B * h;
				G = (T)0;
			}
			else {   //  R>B>G
				const T& h = (T)6.0 * (-H + (T)1.0);
				R = sqrt(P * P / (Pr + Pb * h * h));
				B = R * h;
				G = (T)0;
			}
		}
		return;
	}

	constexpr float gKappa = 903.296296296f;
	constexpr float gEpsilon = 0.008856452f;
	constexpr float ref_u = 0.19783000664f;
	constexpr float ref_v = 0.46831999494f;

	namespace HSLuv
	{
		typedef struct Bounds {
			float a;
			float b;
		}Bounds;

		typedef struct Triplet {
			float a;
			float b;
			float c;
		}Triplet;

		inline __device__ float dot_product(const Triplet& t1, const Triplet& t2)
		{
			return (t1.a * t2.a + t1.b * t2.b + t1.c * t2.c);
		}

		inline __device__ float to_linear(const float& c)
		{
			return (c > 0.04045f) ? pow((c + 0.055f) / 1.055f, 2.4f) : c / 12.92f;
		}

		inline __device__ float from_linear(const float& c)
		{
			constexpr float reciproc = 1.0f / 2.4f;
			return (c <= 0.0031308f) ? 12.92f * c : 1.055f * pow(c, reciproc) - 0.055f;
		}

		inline __device__ float y2l(const float& y)
		{
			return (y <= gEpsilon) ? y * gKappa : 116.0f * cbrt(y) - 16.0f;
		}

		inline __device__ void rgb2xyz(const float& R, const float& G, const float& B, float& x, float& y, float& z)
		{
			/* for XYZ */
			constexpr Triplet m_inv[3] = {
				{ 0.412390799f,  0.357584339f,  0.180480788f },
				{ 0.212639006f,  0.715168679f,  0.072192315f },
				{ 0.019330819f,  0.119194780f,  0.950532152f }
			};

			const Triplet& rgbl{ to_linear(R), to_linear(G), to_linear(B) };
			x = dot_product(m_inv[0], rgbl);
			y = dot_product(m_inv[1], rgbl);
			z = dot_product(m_inv[2], rgbl);
			return;
		}

		inline __device__ void xyz2luv(const float& X, const float& Y, const float& Z, float& l, float& u, float& v)
		{
			constexpr float denom = 1.e-7f;
			const float& divider = X + 15.0f * Y + 3.0f * Z;
			const float& var_u = (4.0f * X) / divider;
			const float& var_v = (9.0f * Y) / divider;
			l = y2l(Y);
			u = (l > denom) ? 13.0f * l * (var_u - ref_u) : 0.f;
			v = (l > denom) ? 13.0f * l * (var_v - ref_v) : 0.f;
			return;
		}

		inline __device__ void luv2lch(const float& L, const float& U, const float& V, float& l, float& c, float& h)
		{
			constexpr float denom = 1e-7f;
			constexpr float divider = 180.f / 3.14159265358979323846f;
			l = L;
			c = sqrt(U * U + V * V);
			const float& tH = (c > denom) ? atan2(V, U) * divider : 0.f;
			h = (tH < 0.0f) ? tH + 360.f : tH;
			return;
		}

		inline __device__ void get_bounds(const float& l, Bounds bounds[6])
		{
			/* for RGB */
			constexpr Triplet m[3] = {
				{  3.240969942f, -1.537383178f, -0.498610760f },
				{ -0.969243636f,  1.875967502f,  0.041555057f },
				{  0.055630080f, -0.203976959f,  1.056971514f }
			};

			const float& tl = l + 16.0f;
			const float& sub1 = (tl * tl * tl) / 1560896.0f;
			const float& sub2 = (sub1 > gEpsilon ? sub1 : (l / gKappa));

			for (int channel = 0; channel < 3; channel++) {
				const float& m1 = m[channel].a;
				const float& m2 = m[channel].b;
				const float& m3 = m[channel].c;

				for (int t = 0; t < 2; t++) {
					const float& top1 = (284517.0f * m1 - 94839.0f * m3) * sub2;
					const float& top2 = (838422.0f * m3 + 769860.0f * m2 + 731718.0f * m1) * l * sub2 - 769860.0f * t * l;
					const float& bottom = (632260.0f * m3 - 126452.0f * m2) * sub2 + 126452.0f * t;

					bounds[channel * 2 + t].a = top1 / bottom;
					bounds[channel * 2 + t].b = top2 / bottom;
				}
			}
			return;
		}

		inline __device__ float ray_length_until_intersect(const float& theta, const Bounds& line)
		{
			return line.b / (sin(theta) - line.a * cos(theta));
		}

		inline __device__ float intersect_line_line(const Bounds& line1, const Bounds& line2)
		{
			return (line1.b - line2.b) / (line2.a - line1.a);
		}

		inline __device__ float dist_from_pole_squared(const float& x, const float& y)
		{
			return x * x + y * y;
		}

		inline __device__ float max_safe_chroma_for_l(const float& l)
		{
			float min_len_squared = flt_MAX;
			Bounds bounds[6];

			get_bounds(l, bounds);
			for (int i = 0; i < 6; i++)
			{
				const float& m1 = bounds[i].a;
				const float& b1 = bounds[i].b;

				const Bounds line2 = { -1.0f / m1, 0.0f };
				const float& x = intersect_line_line(bounds[i], line2);
				const float& distance = dist_from_pole_squared(x, b1 + x * m1);

				if (distance < min_len_squared)
					min_len_squared = distance;
			}

			return FastCompute::Sqrt(min_len_squared);
		}

		inline __device__ float max_chroma_for_lh(const float& l, const float& h)
		{
			constexpr float PiDiv180 = 3.14159265358979323846f / 180.f;
			Bounds bounds[6]{};
			float min_len = flt_MAX;
			const float& hrad = h * PiDiv180;
			int i;

			get_bounds(l, bounds);


			for (i = 0; i < 6; i++)
			{
				const float& len = ray_length_until_intersect(hrad, bounds[i]);
				if (len >= 0.f && len < min_len)
					min_len = len;
			}
			return min_len;
		}

		inline __device__ void lch2hsluv(const float& L, const float& C, const float& HH, float& H, float& S, float& Luv)
		{
			constexpr float denom = 1e-7f;
			constexpr float denomHigh = 99.9999f;
			H = HH;
			S = (L > denomHigh || L < denom) ? 0.f : C / max_chroma_for_lh(L, H) * 100.0f;
			Luv = L;
			return;
		}

		inline __device__ void hsluv2lch(const float& H, const float& S, const float& Luv, float& l, float& c, float& h)
		{
			constexpr float denom = 1e-7f;
			constexpr float denomHigh = 99.9999f;
			l = Luv;
			c = (Luv > denomHigh || Luv < denom) ? 0.0f : max_chroma_for_lh(Luv, H) / 100.0f * S;
			h = (S < denom) ? 0.f : H;
		}

		inline __device__ void lch2luv(const float& L, const float& C, const float& H, float& l, float& u, float& v)
		{
			constexpr float PiDiv180 = 3.14159265358979323846f / 180.f;
			const float& hRad = H * PiDiv180;

			l = L;
			u = cos(hRad) * C;
			v = sin(hRad) * C;
			return;
		}

		inline __device__ const float l2y(const float& l)
		{
			if (l <= 8.0f) {
				return l / gKappa;
			}
			else {
				const float& x = (l + 16.0f) / 116.0f;
				return (x * x * x);
			}
		}

		inline __device__ void luv2xyz(const float& L, const float& U, const float& V, float& X, float& Y, float& Z)
		{
			constexpr float denom = 1e-7f;
			if (L > denom)
			{
				const float& var_u = U / (13.0f * L) + ref_u;
				const float& var_v = V / (13.0f * L) + ref_v;
				Y = l2y(L);
				X = -(9.0f * Y * var_u) / ((var_u - 4.0f) * var_v - var_u * var_v);
				Z = (9.0f * Y - (15.0f * var_v * Y) - (var_v * X)) / (3.0f * var_v);
			}
			else
			{
				X = Y = Z = 0.0f;
			}

			return;
		}

		inline __device__ void xyz2rgb(const float& X, const float& Y, const float& Z, float& R, float& G, float& B)
		{
			constexpr Triplet m[3] = {
				{  3.240969942f, -1.537383178f, -0.498610760f },
				{ -0.969243636f,  1.875967502f,  0.041555057f },
				{  0.055630080f, -0.203976959f,  1.056971514f }
			};

			const Triplet& t{ X, Y, Z };
			R = from_linear(dot_product(m[0], t));
			G = from_linear(dot_product(m[1], t));
			B = from_linear(dot_product(m[2], t));
			return;
		}

		inline __device__ void lch2hpluv(const float& L, const float& C, const float& HH, float& H, float& P, float& Luv)
		{
			constexpr float denom = 1e-7f;
			constexpr float denomHigh = 99.9999f;

			H = (C > denom) ? HH : 0.f;
			P = (L > denomHigh || L < denom) ? 0.f : C / max_safe_chroma_for_l(L) * 100.0f;
			Luv = L;
			return;
		}

		inline __device__ void hpluv2lch(const float& H, const float& P, const float& Luv, float& L, float& C, float& HH)
		{
			constexpr float denom = 1e-7f;
			constexpr float denomHigh = 99.9999f;

			C = (Luv > denomHigh || Luv < denom) ? 0.f : max_safe_chroma_for_l(L) / 100.0f * P;
			HH = (P > denom) ? HH : 0.f;
			L = Luv;
		}

	} /* namespace HSLuv */


	inline void __device__ sRgb2hsLuv(const float& R, const float& G, const float& B, float& H, float& S, float& Luv) noexcept
	{
		float X, Y, Z;
		float l, u, v;
		float L_, C, H_;

		/* convert sRGB to XYZ */
		HSLuv::rgb2xyz(R, G, B, X, Y, Z);

		/* convert XYZ to LUV */
		HSLuv::xyz2luv(X, Y, Z, l, u, v);

		/* convert LUV to LCH */
		HSLuv::luv2lch(l, u, v, L_, C, H_);

		/* convert LCH to HSLuv */
		HSLuv::lch2hsluv(L_, C, H_, H, S, Luv);

		return;
	}


	inline __device__ void hsLuv2sRgb(const float& H, const float& S, const float& Luv, float& R, float& G, float& B) noexcept
	{
		float L, C, H_;
		float l, u, v;
		float X, Y, Z;

		/* convert HSLuv to LCH */
		HSLuv::hsluv2lch(H, S, Luv, L, C, H_);

		/* convert LCH to LUV */
		HSLuv::lch2luv(L, C, H_, l, u, v);

		/* convert LUV to XYZ */
		HSLuv::luv2xyz(l, u, v, X, Y, Z);

		/* convert XYZ to RGB */
		HSLuv::xyz2rgb(X, Y, Z, R, G, B);

		return;
	}


	inline __device__ void rgb2hpLuv(const float& R, const float& G, const float& B, float& H, float& P, float& Luv) noexcept
	{
		float X, Y, Z;
		float l, u, v;
		float L_, C, H_;

		/* convert sRGB to XYZ */
		HSLuv::rgb2xyz(R, G, B, X, Y, Z);

		/* convert XYZ to LUV */
		HSLuv::xyz2luv(X, Y, Z, l, u, v);

		/* convert LUV to LCH */
		HSLuv::luv2lch(l, u, v, L_, C, H_);

		/* convert LCH to HPLuv */
		HSLuv::lch2hpluv(L_, C, H_, H, P, Luv);

		return;
	}


	inline __device__ void hpLuv2rgb(const float& H, const float& P, const float& Luv, float& R, float& G, float& B) noexcept
	{
		float L, C, H_;
		float l, u, v;
		float X, Y, Z;

		/* CONVERT HPLuv to LCH */

		/* convert LCH to LUV */
		HSLuv::lch2luv(L, C, H_, l, u, v);

		/* convert LUV to XYZ */
		HSLuv::luv2xyz(l, u, v, X, Y, Z);

		/* convert XYZ to RGB */
		HSLuv::xyz2rgb(X, Y, Z, R, G, B);

		return;
	}

}