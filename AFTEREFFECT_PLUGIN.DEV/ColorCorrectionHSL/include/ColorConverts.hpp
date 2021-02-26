#pragma once

#include "CompileTimeUtils.hpp"
#include "FastAriphmetics.hpp"


template<typename T>
inline const typename std::enable_if<std::is_floating_point<T>::value, T>::type
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


inline void sRgb2hsl(const float& R, const float& G, const float& B, float& H, float& S, float& L) noexcept
{
	/* start convert RGB to HSL color space */
	float const maxVal = MAX3_VALUE(R, G, B);
	float const minVal = MIN3_VALUE(R, G, B);
	float const sumMaxMin = maxVal + minVal;
	float luminance = sumMaxMin * 50.0f; /* luminance value in percents = 100 * (max + min) / 2 */
	float hue, saturation;

	if (maxVal == minVal)
	{
		saturation = hue = 0.0f;
	}
	else
	{
		auto const& subMaxMin = maxVal - minVal;
		saturation = (100.0f * subMaxMin) / ((luminance < 50.0f) ? sumMaxMin : (2.0f - sumMaxMin));
		if (R == maxVal)
			hue = (60.0f * (G - B)) / subMaxMin;
		else if (G == maxVal)
			hue = (60.0f * (B - R)) / subMaxMin + 120.0f;
		else
			hue = (60.0f * (R - G)) / subMaxMin + 240.0f;
	}

	H = hue;
	S = saturation;
	L = luminance;

	return;
}


inline void hsl2sRgb(const float& H, const float& S, const float& L, float& R, float& G, float& B) noexcept
{
	constexpr float denom = 1.f / 1e7f;
	constexpr float reciproc3 = 1.0f / 3.0f;
	constexpr float reciproc360 = 1.0f / 360.f;

	float rR, gG, bB;

	/* back convert to RGB space */
	if (denom >= S)
	{
		rR = gG = bB = L;
	}
	else
	{
		float tmpVal1, tmpVal2;
		tmpVal2 = (L < 0.50f) ? (L * (1.0f + S)) : (L + S - (L * S));
		tmpVal1 = 2.0f * L - tmpVal2;

		auto tmpG = H;
		auto tmpR = H + reciproc3;
		auto tmpB = H - reciproc3;

		tmpR -= ((tmpR > 1.0f) ? 1.0f : 0.0f);
		tmpB += ((tmpB < 0.0f) ? 1.0f : 0.0f);

		rR = restore_rgb_channel_value(tmpVal1, tmpVal2, tmpR);
		gG = restore_rgb_channel_value(tmpVal1, tmpVal2, tmpG);
		bB = restore_rgb_channel_value(tmpVal1, tmpVal2, tmpB);
	}

	R = rR;
	G = gG;
	B = bB;

	return;
}


inline void sRgb2hsv(const float& R, const float& G, const float& B, float& H, float& S, float& V) noexcept
{
	constexpr float denom = 1.f / 1e7f;

	/* start convert RGB to HSV color space */
	float const& maxVal = MAX3_VALUE(R, G, B);
	float const& minVal = MIN3_VALUE(R, G, B);
	float const& C = maxVal - minVal;

	float const& value = maxVal;
	float hue, saturation;

	if (denom < C)
	{
		if (R == maxVal)
		{
			hue = (G - B) / C;
			if (G < B)
				hue += 6.f;
		}
		else if (G == maxVal)
			hue = 2.f + (B - R) / C;
		else
			hue = 4.f + (R - G) / C;

		hue *= 60.f;
		saturation = C / maxVal;
	}
	else
	{
		hue = saturation = 0.f;
	}

	H = hue;
	S = saturation;
	V = value;

	return;
}


inline void hsv2sRgb(const float& H, const float& S, const float& V, float& R, float& G, float& B) noexcept
{
	auto const& c = S * V;
	auto const& minV = V - c;

	constexpr float reciproc360 = 1.0f / 360.f;
	constexpr float reciproc60 = 1.0f  / 60.f;

	float const& h = (H - 360.f * floor(H * reciproc360)) * reciproc60;
	float const& X = c * (1.0f - fabs(h - 2.0f * floor(h * 0.5f) - 1.0f));
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


inline void sRgb2hsi(const float& R, const float& G, const float& B, float& H, float& S, float& I) noexcept
{
	constexpr float reciproc3 = 1.f / 3.f;
	constexpr float denom = 1.f / 1e7f;
	constexpr float reciprocPi180 = 180.f / 3.14159265f;

	float i = (R + G + B) * reciproc3;
	float h, s;

	if (i > denom)
	{
		auto const& alpha = 0.5f * (2.f * R - G - B) + denom;
		auto const& beta = 0.8660254037f * (G - B) + denom;
		s = 1.f - MIN3_VALUE(R, G, B) / i;
		h = FastCompute::Atan2(beta, alpha) * reciprocPi180;
		if (h < 0)
			h += 360.f;
	}
	else
	{
		i = h = s = 0.f;
	}

	H = h;
	S = s;
	I = i;

	return;
}


inline void hsi2sRgb(const float& H, const float& S, const float& I, float& R, float& G, float& B) noexcept
{
	constexpr float PiDiv180 = 3.14159265f / 180.f;
	constexpr float reciproc360 = 1.0f / 360.f;
	constexpr float denom = 1.f / 1e7f;

	float h =  H - 360.f * floor (H * reciproc360);
	const float& val1 = I * (1.f - S);
	const float& tripleI = 3.f * I;

	if (h < 120.f)
	{
		const float& cosTmp = FastCompute::Cos((60.f - h) * PiDiv180);
		const float& cosDiv = (0.f == cosTmp) ? denom : cosTmp;
		B = val1;
		R = I * (1.f + S * FastCompute::Cos(h * PiDiv180) / cosDiv);
		G = tripleI - R - B;
	}
	else if (h < 240.f)
	{
		h -= 120.f;
		const float& cosTmp = FastCompute::Cos((60.f - h) * PiDiv180);
		const float& cosDiv = (0.f == cosTmp) ? denom : cosTmp;
		R = val1;
		G = I * (1.f + S * FastCompute::Cos(h * PiDiv180) / cosDiv);
		B = tripleI - R - G;
	}
	else
	{
		h -= 240.f;
		const float& cosTmp = FastCompute::Cos((60.f - h) * PiDiv180);
		const float& cosDiv = (0.f == cosTmp) ? denom : cosTmp;
		G = val1;
		B = I * (1.f + S * FastCompute::Cos(h * PiDiv180) / cosDiv);
		R = tripleI - G - B;
	}

	return;
}


inline void sRgb2hsp(const float& R, const float& G, const float& B, float& H, float& S, float& P) noexcept
{
	constexpr float Pr = 0.2990f;
	constexpr float Pg = 0.5870f;
	constexpr float Pb = 1.0f - Pr - Pg; /* 0.1140f */;
	constexpr float div1on6 = 1.f / 6.f;
	constexpr float div2on6 = 2.f / 6.f;
	constexpr float div4on6 = 4.f / 6.f;

	float const& maxVal = MAX3_VALUE(R, G, B);

	P = sqrt(R * R * Pr + G * G * Pg + B * B * Pb);

	if (R == G && R == B) {
		H = S = 0.f;
	}
	else {
		if (R == maxVal)
		{   //  R is largest
			if (B >= G) {
				H = 1.f - div1on6 * (B - G) / (R - G);
				S = 1.f - G / R;
			}
			else {
				H = div1on6 * (G - B) / (R - B);
				S = 1.f - B / R;
			}
		}
		else if (G == maxVal) 
		{   //  G is largest
			if (R >= B) {
				H = div2on6 - div1on6 * (R - B) / (G - B);
				S = 1.f - B / G;
			}
			else {
				H = div2on6 + div1on6 * (B - R) / (G - R);
				S = 1.f - R / G;
			}
		}
		else {   //  B is largest
			if (G >= R) {
				H = div4on6 - div1on6 * (G - R) / (B - R);
				S = 1.f - R / B;
			}
			else {
				H = div4on6 + div1on6 * (R - G) / (B - G);
				S = 1.f - G / B;
			}
		}
	}
	return;
}


inline void hsp2sRgb(const float& H, const float& S, const float& P, float& R, float& G, float& B) noexcept
{
	constexpr float Pr = 0.2990f;
	constexpr float Pg = 0.5870f;
	constexpr float Pb = 1.0f - Pr - Pg; /* 0.1140f */;

	const float& minOverMax = 1.0f - S;
	float part;

	if (minOverMax > 0.f)
	{
		if ( H < 1.f / 6.f) {   //  R>G>B
			const float& h = 6.f * H; 
			part = 1.f + h * (1.f / minOverMax - 1.f);
			B = P / FastCompute::Sqrt(Pr / minOverMax / minOverMax + Pg * part * part + Pb);
			R = B / minOverMax;
			G = B + h * (R - B);
		}
		else if ( H < 2.f / 6.f) {   //  G>R>B
			const float& h = 6. * (-H + 2.f / 6.f); 
			part = 1.f + h  * (1.f / minOverMax - 1.f);
			B = P / FastCompute::Sqrt(Pg / minOverMax / minOverMax + Pr * part * part + Pb);
			G = B / minOverMax;
			R = B + h * (G - B);
		}
		else if (H < 3.f / 6.f) {   //  G>B>R
			const float& h = 6.f * (H - 2.f / 6.f);
			part = 1.f + h * (1.f / minOverMax - 1.f);
			R = P / FastCompute::Sqrt(Pg / minOverMax / minOverMax + Pb * part * part + Pr);
			G = R / minOverMax;
			B = R + h * (G - R);
		}
		else if (H < 4.f / 6.f) {   //  B>G>R
			const float& h = 6.f * (-H + 4.f / 6.f);
			part = 1.f + h * (1.f / minOverMax - 1.f);
			R = P / FastCompute::Sqrt(Pb / minOverMax / minOverMax + Pg * part * part + Pr);
			B = R / minOverMax;
			G = R + h * (B - R);
		}
		else if (H < 5.f / 6.f) {   //  B>R>G
			const float& h = 6.f * (H - 4.f / 6.f);
			part = 1.f + h * (1.f / minOverMax - 1.f);
			G = P / FastCompute::Sqrt(Pb / minOverMax / minOverMax + Pr * part * part + Pg);
			B = G / minOverMax;
			R = G + h * (B - G);
		}
		else {   //  R>B>G
			const float& h = 6.f * (-H + 1.f);
			part = 1.f + h * (1.f / minOverMax - 1.f);
			G = P / FastCompute::Sqrt(Pr / minOverMax / minOverMax + Pb * part * part + Pg);
			R = G / minOverMax;
			B = G + h * (R - G);
		}
	}
	else
	{
		if ( H < 1.f / 6.f) {   //  R>G>B
			const float& h = 6.f * H;
			R = FastCompute::Sqrt(P * P / (Pr + Pg * h * h));
			G = R *h;
			B = 0.f;
		}
		else if (H < 2.f / 6.f) {   //  G>R>B
			const float& h = 6.f * (-H + 2.f / 6.f);
			G = FastCompute::Sqrt(P * P / (Pg + Pr * h * h));
			R = G * h; 
			B = 0.f;
		}
		else if (H < 3.f / 6.f) {   //  G>B>R
			const float& h = 6.f * (H - 2.f / 6.f);
			G = FastCompute::Sqrt(P * P / (Pg + Pb * h * h));
			B = G * h;
			R = 0.f;
		}
		else if (H < 4.f / 6.f) {   //  B>G>R
			const float& h = 6.f * (-H + 4.f / 6.f);
			B = FastCompute::Sqrt(P * P / (Pb + Pg * h * h));
			G = B * h; 
			R = 0.f;
		}
		else if (H < 5.f / 6.f) {   //  B>R>G
			const float& h = 6.f * (H - 4.f / 6.f);
			B = FastCompute::Sqrt(P * P / (Pb + Pr * h * h));
			R = B * h;
			G= 0.f;
		}
		else {   //  R>B>G
			const float& h = 6.f * (-H + 1.f);
			R = FastCompute::Sqrt(P * P / (Pr + Pb * h * h));
			B = R * h; 
			G = 0.f;
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
	};

	typedef struct Triplet {
		float a;
		float b;
		float c;
	};

	/* for RGB */
	constexpr static Triplet m[3] = {
		{  3.240969942f, -1.537383178f, -0.498610760f },
		{ -0.969243636f,  1.875967502f,  0.041555057f },
		{  0.055630080f, -0.203976959f,  1.056971514f }
	};

	/* for XYZ */
	constexpr static Triplet m_inv[3] = {
		{  0.412390799f,  0.357584339f,  0.180480788f },
		{  0.212639006f,  0.715168679f,  0.072192315f },
		{  0.019330819f,  0.119194780f,  0.950532152f }
	};

	inline float dot_product (const Triplet& t1, const Triplet& t2)
	{
		return (t1.a * t2.a + t1.b * t2.b + t1.c * t2.c);
	}

	inline float to_linear (const float& c)
	{
//		return (c > 0.04045f) ? FastCompute::Pow((c + 0.055f) / 1.055f, 2.4f) : c / 12.92f;
		return (c > 0.04045f) ? pow((c + 0.055f) / 1.055f, 2.4f) : c / 12.92f;
	}

	inline float from_linear (const float& c)
	{
		constexpr float reciproc = 1.0f / 2.4f;
		return (c <= 0.0031308f) ? 12.92f * c : 1.055f * FastCompute::Pow(c, reciproc) - 0.055f;
	}

	inline float y2l (const float& y)
	{
		return (y <= gEpsilon) ? y * gKappa : 116.0f * FastCompute::Cbrt(y) - 16.0f;
	}

	inline void rgb2xyz (const float& R, const float& G, const float& B, float& x, float& y, float& z)
	{
		const Triplet& rgbl { to_linear(R), to_linear(G), to_linear(B) };
		x = dot_product(m_inv[0], rgbl);
		y = dot_product(m_inv[1], rgbl);
		z = dot_product(m_inv[2], rgbl);
		return;
	}

	inline void xyz2luv (const float& X, const float& Y, const float& Z, float& l, float& u, float& v)
	{
		constexpr float denom = 1.e-7f;
		const float& divider = X + 15.0f * Y + 3.0f * Z;
		const float& var_u = (4.0f * X) / divider;
		const float& var_v = (9.0f * Y) / divider;
		l = y2l (Y);
		u = (l > denom) ? 13.0f * l * (var_u - ref_u) : 0.f;
		v = (l > denom) ? 13.0f * l * (var_v - ref_v) : 0.f;
		return;
	}

	inline void luv2lch (const float& L, const float& U, const float& V, float& l, float& c, float& h)
	{
		constexpr float denom = 1e-7f;
		constexpr float divider = 180.f / FastCompute::PI;
		l = L;
		c = sqrt(U * U + V * V);
		const float& tH = (c > denom) ? FastCompute::Atan2(V, U) * divider : 0.f;
		h = (tH < 0.0f) ? tH + 360.f : tH;
		return;
	}

	inline void get_bounds(const float& l, Bounds bounds[6])
	{
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

	inline float ray_length_until_intersect(const float& theta, const Bounds& line)
	{
		return line.b / (sin(theta) - line.a * cos(theta));
//		return line.b / (FastCompute::Sin(theta) - line.a * FastCompute::Cos(theta));
	}

	inline float max_chroma_for_lh(const float& l, const float& h)
	{
		constexpr float PiDiv180 = FastCompute::PI / 180.f;
		CACHE_ALIGN Bounds bounds[6]{};
		float min_len = FLT_MAX;
		const float& hrad = h * PiDiv180;
		int i;

		get_bounds(l, bounds);

		__VECTOR_ALIGNED__
		for (i = 0; i < 6; i++)
		{
			const float& len = ray_length_until_intersect(hrad, bounds[i]);
				if (len >= 0.f && len < min_len)
					min_len = len;
		}
		return min_len;
	}

	inline void lch2hsluv(const float& L, const float& C, const float& HH, float& H, float& S, float& Luv)
	{
		constexpr float denom = 1e-7f;
		constexpr float denomHigh = 99.9999f;
		H = HH;
		S = (L > denomHigh || L < denom) ? 0.f : C / max_chroma_for_lh(L, H) * 100.0f;
		Luv = L;
		return;
	}

	inline void hsluv2lch (const float& H, const float& S, const float& Luv, float& l, float& c, float& h)
	{
		constexpr float denom = 1e-7f;
		constexpr float denomHigh = 99.9999f;
		l = Luv;
		c = (Luv > denomHigh || Luv < denom) ? 0.0f : max_chroma_for_lh(Luv, H) / 100.0f * S;
		h = (S < denom) ? 0.f : H;
	}

	inline void lch2luv (const float& L, const float& C, const float& H, float& l, float& u, float& v)
	{
		constexpr float PiDiv180 = FastCompute::PI / 180.f;
		const float& hRad = H * PiDiv180;

		l = L;
//		u = FastCompute::Cos(hRad) * C;
//		v = FastCompute::Sin(hRad) * C;
		u = cos(hRad) * C;
		v = sin(hRad) * C;
		return;
	}

	inline const float l2y (const float& l)
	{
		if (l <= 8.0f) {
			return l / gKappa;
		}
		else {
			const float& x = (l + 16.0f) / 116.0f;
			return (x * x * x);
		}
	}

	inline void luv2xyz (const float& L, const float& U, const float& V, float& X, float& Y, float& Z)
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

	inline void xyz2rgb (const float& X, const float& Y, const float& Z, float& R, float& G, float& B)
	{
		const Triplet& t{ X, Y, Z };
		R = from_linear(dot_product(m[0], t));
		G = from_linear(dot_product(m[1], t));
		B = from_linear(dot_product(m[2], t));
		return;
	}

} /* namespace HSLuv */


inline void sRgb2hsLuv(const float& R, const float& G, const float& B, float& H, float& S, float& Luv) noexcept
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


inline void hsLuv2sRgb(const float& H, const float& S, const float& Luv, float& R, float& G, float& B) noexcept
{
	float L, C, H_;
	float l, u, v;
	float X, Y, Z;

	/* convert HSLuv to LCH */
	HSLuv::hsluv2lch (H, S, Luv, L, C, H_);

	/* convert LCH to LUV */
	HSLuv::lch2luv (L, C, H_, l, u, v);

	/* convert LUV to XYZ */
	HSLuv::luv2xyz (l, u, v, X, Y, Z);

	/* convert XYZ to RGB */
	HSLuv::xyz2rgb (X, Y, Z, R, G, B);

	return;
}


inline void rgb2hpLuv(const float& R, const float& G, const float& B, float& H, float& P, float& Luv) noexcept
{
	return;
}


inline void hpLuv2rgb(const float& H, const float& P, const float& Luv, float& R, float& G, float& B) noexcept
{
	return;
}