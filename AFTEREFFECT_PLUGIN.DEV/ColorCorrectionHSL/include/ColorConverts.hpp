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
	constexpr float reciproc3 = 1.0f / 3.0f;
	constexpr float reciproc360 = 1.0f / 360.f;

	float rR, gG, bB;

	/* back convert to RGB space */
	if (0.f == S)
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
	/* start convert RGB to HSV color space */
	float const maxVal = MAX3_VALUE(R, G, B);
	float const minVal = MIN3_VALUE(R, G, B);
	float const C = maxVal - minVal;

	float const& value = maxVal;
	float hue, saturation;

	if (0.f == C)
	{
		hue = saturation = 0.f;
	}
	else
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

	H = hue;
	S = saturation;
	V = value;

	return;
}


inline void hsv2sRgb(const float& H, const float& S, const float& V, float& R, float& G, float& B) noexcept
{
	auto const& c = S * V;
	auto const& minV = V - c;
	float h = H;

	constexpr float reciproc360 = 1.0f / 360.f;
	constexpr float reciproc60 = 1.0f  / 60.f;

	h -= 360.f * floor(h * reciproc360);
	h *= reciproc60;
	float const& X = c * (1.0f - fabs(h - 2.0f * floor(h * 0.5f) - 1.0f));

	float fR, fG, fB;
	const int& hh = static_cast<int const>(h);

	if (hh < 3)
	{
		if (0 == hh)
		{
			fR = minV + c; fG = minV + X; fB = minV;
		}
		else if (1 == hh)
		{
			fR = minV + X; fG = minV + c; fB = minV;
		}
		else
		{
			fR = minV; fG = minV + c; fB = minV + X;
		}
	}
	else {
		if (3 == hh)
		{
			fR = minV; fG = minV + X; fB = minV + c;
		}
		else if (4 == hh)
		{
			fR = minV + X; fG = minV; fB = minV + c;
		}
		else
		{
			fR = minV + c; fG = minV; fB = minV + X;
		}
	}

	R = fR;
	G = fG;
	B = fB;

	return;
}