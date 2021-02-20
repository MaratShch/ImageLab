#pragma once

#include "CompileTimeUtils.hpp"
#include <cmath>

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
		h = atan2(beta, alpha) * reciprocPi180;
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
		const float& cosTmp = cos((60.f - h) * PiDiv180);
		const float& cosDiv = (0.f == cosTmp) ? denom : cosTmp;
		B = val1;
		R = I * (1.f + S * cos(h * PiDiv180) / cosDiv);
		G = tripleI - R - B;
	}
	else if (h < 240.f)
	{
		h -= 120.f;
		const float& cosTmp = cos((60.f - h) * PiDiv180);
		const float& cosDiv = (0.f == cosTmp) ? denom : cosTmp;
		R = val1;
		G = I * (1.f + S * cos(h * PiDiv180) / cosDiv);
		B = tripleI - R - G;
	}
	else
	{
		h -= 240.f;
		const float& cosTmp = cos((60.f - h) * PiDiv180);
		const float& cosDiv = (0.f == cosTmp) ? denom : cosTmp;
		G = val1;
		B = I * (1.f + S * cos(h * PiDiv180) / cosDiv);
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

	const float& minOverMax = 1. - S;
	float part;

	if (minOverMax > 0.f)
	{
		if ( H < 1.f / 6.f) {   //  R>G>B
			const float& h = 6.f * H; 
			part = 1.f + h * (1.f / minOverMax - 1.f);
			B = P / sqrt(Pr / minOverMax / minOverMax + Pg * part * part + Pb);
			R = B / minOverMax;
			G = B + h * (R - B);
		}
		else if ( H < 2.f / 6.f) {   //  G>R>B
			const float& h = 6. * (-H + 2.f / 6.f); 
			part = 1.f + h  * (1.f / minOverMax - 1.f);
			B = P / sqrt(Pg / minOverMax / minOverMax + Pr*part*part + Pb);
			G = B / minOverMax;
			R = B + h * (G - B);
		}
		else if (H < 3.f / 6.f) {   //  G>B>R
			const float& h = 6.f * (H - 2.f / 6.f);
			part = 1.f + h * (1.f / minOverMax - 1.f);
			R = P / sqrt(Pg / minOverMax / minOverMax + Pb*part*part + Pr);
			G = R / minOverMax;
			B = R + h * (G - R);
		}
		else if (H < 4.f / 6.f) {   //  B>G>R
			const float& h = 6.f * (-H + 4.f / 6.f);
			part = 1.f + h * (1.f / minOverMax - 1.f);
			R = P / sqrt(Pb / minOverMax / minOverMax + Pg*part*part + Pr);
			B = R / minOverMax;
			G = R + h * (B - R);
		}
		else if (H < 5.f / 6.f) {   //  B>R>G
			const float& h = 6.f * (H - 4.f / 6.f);
			part = 1.f + h * (1.f / minOverMax - 1.f);
			G = P / sqrt(Pb / minOverMax / minOverMax + Pr*part*part + Pg);
			B = G / minOverMax;
			R = G + h * (B - G);
		}
		else {   //  R>B>G
			const float& h = 6.f * (-H + 1.f);
			part = 1.f + h * (1.f / minOverMax - 1.f);
			G = P / sqrt(Pr / minOverMax / minOverMax + Pb*part*part + Pg);
			R = G / minOverMax;
			B = G + h * (R - G);
		}
	}
	else
	{
		if ( H < 1.f / 6.f) {   //  R>G>B
			const float& h = 6.f * H;
			R = sqrt(P*P / (Pr + Pg * h * h));
			G = R *h;
			B = 0.f;
		}
		else if (H < 2.f / 6.f) {   //  G>R>B
			const float& h = 6.f * (-H + 2.f / 6.f);
			G = sqrt(P*P / (Pg + Pr*h*h));
			R = G * h; 
			B = 0.f;
		}
		else if (H < 3.f / 6.f) {   //  G>B>R
			const float& h = 6.f * (H - 2.f / 6.f);
			G = sqrt(P*P / (Pg + Pb*h*h));
			B = G * h;
			R = 0.f;
		}
		else if (H < 4.f / 6.f) {   //  B>G>R
			const float& h = 6.f * (-H + 4.f / 6.f);
			B = sqrt(P*P / (Pb + Pg*h*h));
			G = B * h; 
			R = 0.f;
		}
		else if (H<5. / 6.) {   //  B>R>G
			const float& h = 6.f * (H - 4.f / 6.f);
			B = sqrt(P*P / (Pb + Pr*h*h));
			R = B * h;
			G= 0.f;
		}
		else {   //  R>B>G
			const float& h = 6.f * (-H + 1.f);
			R = sqrt(P*P / (Pr + Pb*h*h)); 
			B = R *h; 
			G = 0.f;
		}
	}
	return;
}