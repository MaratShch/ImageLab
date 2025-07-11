#include "FastAriphmetics.hpp"
#include "ImageEqualization.hpp"
#include "ColorTransform.hpp"
#include "CommonAuxPixFormat.hpp"
#include "ImageEqualizationColorTransform.hpp"

CACHE_ALIGN constexpr float cstar_gmax[] = {
	#include "ImageCoefficients.txt"
};
constexpr size_t coeffArraySize = sizeof(cstar_gmax);
//https://github.com/ta850-z/limited_hues_enhancement/blob/main/limited_hues_enhancement.m
 
inline const void __xyz2rgb (const float& x, const float& y, const float& z, float& r, float& g, float& b) noexcept
{
	CACHE_ALIGN constexpr float xyz2rgb[9] = 
	{ 
		XYZtosRGB[0], XYZtosRGB[1], XYZtosRGB[2],
		XYZtosRGB[3], XYZtosRGB[4], XYZtosRGB[5],
		XYZtosRGB[6], XYZtosRGB[7], XYZtosRGB[8]
	};
	r = x * xyz2rgb[0] + y * xyz2rgb[1] + z * xyz2rgb[2];
	g = x * xyz2rgb[3] + y * xyz2rgb[4] + z * xyz2rgb[5];
	b = x * xyz2rgb[6] + y * xyz2rgb[7] + z * xyz2rgb[8];
	return;
 };

#ifdef _DEBUG
int32_t dbg_indexH_min = INT_MAX;
int32_t dbg_indexH_max = INT_MIN;
int32_t dbg_indexY_min = INT_MAX;
int32_t dbg_indexY_max = INT_MIN;
#endif

void fCIELabHueImprove
(
	const fCIELabPix* pLabSrc,
	fXYZPix*          xyz_out,
	const float*      cs_out,
	A_long sizeX,
	A_long sizeY,
	float  alpha,
	float  ligntness,
	float  cs_out_max
) noexcept
{
	auto const sign = [&](const float f) noexcept {
		return (f > 0.f ? 1.f : (f < 0.f) ? -1.f : 0.f);
	};

	auto const tone_map_inc = [&](const float w) noexcept {
		constexpr float toneMin = 0.2f;
		constexpr float toneMax = 0.8f;
		constexpr float toneDiff = toneMax - toneMin;
		constexpr float toneDiffReciproc = 1.f / toneDiff;
		return ((w < toneMin) ? 0.f : (w > toneMax) ? 1.f : (w - toneMin) * toneDiffReciproc);
	};

	auto const gamut_descript = [&](const float r, const float g, const float b) noexcept {
		const float gR = (r < -0.001f || r > 1.001f) ? 0 : 1;
		const float gG = (g < -0.001f || g > 1.001f) ? 0 : 1;
		const float gB = (b < -0.001f || b > 1.001f) ? 0 : 1;
		return gR * gG * gB;
	};

	const A_long imgSize = sizeX * sizeY;
	const float reciprocCsOutMax = 1.f / cs_out_max;
	const float lightness_reciproc = 1.f / ligntness;
	constexpr float reciproc180  = 1.f / 180.f;
	constexpr float ligh_reciproc_100 = 1.f / 100.f;

	for (A_long i = 0; i < imgSize; i++)
	{                                 
		auto const& L = pLabSrc[i].L; 
		auto const& a = pLabSrc[i].a; 
		auto const& b = pLabSrc[i].b; 
		const float wc = tone_map_inc (cs_out[i] * reciprocCsOutMax);
		const float hangle_out2 = FastCompute::Atan2d(b, a) * reciproc180;
		constexpr float alpha = 10.f;
		const float wx = 3.f * wc * FastCompute::Exp(-alpha * hangle_out2 * hangle_out2) + 1.f;

		/* change LAB a and b components */
		float pLabSrcA = wx * a;
		float pLabSrcB = wx * b;
		/* lightness enhanecemnt */
		float pLabSrcL = 100.f * FastCompute::Pow (L * ligh_reciproc_100, lightness_reciproc);

		const float csOut = FastCompute::Sqrt (pLabSrcA * pLabSrcA + pLabSrcB * pLabSrcB);
		const float h_out  = ((csOut >= 0.1f) ? pLabSrcB / pLabSrcA : 0.f);
		const float hangle_out = FastCompute::Atan2d(pLabSrcB, pLabSrcA) + ((pLabSrcB < 0.f) ? 360.f : 0.f);

		constexpr float reciproc116 = 1.f / 116.f;
		const float sq_h_out = h_out * h_out;
		float fY = (pLabSrcL + 16.f) * reciproc116;
		float fX =  sign(pLabSrcA) * csOut / (500.f * FastCompute::Sqrt(1.f + sq_h_out)) + fY;
		float fZ = -sign(pLabSrcB) * csOut / (200.f * FastCompute::Sqrt(1.f + 1.f / sq_h_out)) + fY;

		constexpr float reciproc16 = 16.f / 116.f;
		constexpr float reciproc7x = 0.9505f / 7.78f;
		constexpr float reciproc7y = 1.f / 7.78f;
		constexpr float reciproc7z = 1.089f / 7.78f;
		float X = (fX > 0.20689f ? (0.9505f * fX * fX * fX) : (fX - reciproc16) * reciproc7x);
		float Y = (fY > 0.20689f ? (fY * fY * fY)           : (fY - reciproc16) * reciproc7y);
		float Z = (fZ > 0.20689f ? (1.0890f * fZ * fZ * fZ) : (fZ - reciproc16) * reciproc7z);

		constexpr int32_t maxIndexY = 100;
		constexpr int32_t maxIndexH = 360;
		constexpr int32_t minIndexY = 0;
		constexpr int32_t minIndexH = 0;

		const int32_t indexY = FastCompute::Min(maxIndexY - 1, static_cast<const int32_t>(100.f * Y));	/* 0 ... 100 */
		const int32_t indexH = FastCompute::Min(maxIndexH - 1, static_cast<const int32_t>(hangle_out));	/* 0 ... 360 */

		constexpr float Xn = 0.9505f;
		constexpr float Zn = 1.0890f;

		float rr, gg, bb, gamut_desc;
		float fCsOut = 0.f;

#ifdef _DEBUG
		rr = gg = bb = gamut_desc = 0.f;

		dbg_indexH_min = FastCompute::Min(dbg_indexH_min, indexH); // 0
		dbg_indexH_max = FastCompute::Max(dbg_indexH_max, indexH); // 362 !!!
		dbg_indexY_min = FastCompute::Min(dbg_indexY_min, indexY); // 0
		dbg_indexY_max = FastCompute::Max(dbg_indexY_max, indexY); // 99
#endif
		__xyz2rgb(Xn * fX * fX * fX, fY * fY * fY, Zn * fZ * fZ * fZ, rr, gg, bb);

		if (1.f == (gamut_desc = gamut_descript(rr, gg, bb)))
		{
			fCsOut = csOut;
		}
		else
		{
			if (minIndexY == indexY || maxIndexY == indexY)
			{
				if (minIndexH == indexH || maxIndexH == indexH)
					fCsOut = FastCompute::Min(cstar_gmax[indexY], cstar_gmax[indexY * maxIndexH]);
				else
					fCsOut = FastCompute::Min(cstar_gmax[indexY + maxIndexY * indexH], cstar_gmax[indexY + maxIndexY * (indexH + 1)]);
			}
			else if (minIndexH == indexH || maxIndexH == indexH)
			{
				const float cs = FastCompute::Min3(cstar_gmax[indexY], cstar_gmax[indexY + 1], cstar_gmax[indexY * maxIndexH]);
				fCsOut = FastCompute::Min(cs, cstar_gmax[(indexY + 1) * maxIndexH]);
			}
			else
			{
				const float cs = FastCompute::Min3(cstar_gmax[indexY + maxIndexY * indexH], cstar_gmax[indexY + 1 + maxIndexY * indexH], cstar_gmax[indexY + maxIndexY * indexH]);
				fCsOut = FastCompute::Min(cs, cstar_gmax[indexY + 1 + maxIndexY * (1 + indexH)]);
			}
		}

		fX =  sign(pLabSrcA) * fCsOut / (500.f * FastCompute::Sqrt(1.f + sq_h_out)) + fY;
		fZ = -sign(pLabSrcB) * fCsOut / (200.f * FastCompute::Sqrt(1.f + 1.f / sq_h_out)) + fY;

		xyz_out[i].X = (fX > 0.20689f) ? 0.9505f * fX * fX * fX : (fX - reciproc16) * reciproc7x;
		xyz_out[i].Y = (fY > 0.20689f) ? fY * fY * fY           : (fY - reciproc16) * reciproc7y;
		xyz_out[i].Z = (fZ > 0.20689f) ? 1.0890f * fZ * fZ * fZ : (fZ - reciproc16) * reciproc7z;

	}/* for (i = 0; i < imgSize; i++) */ 

	return;
}
