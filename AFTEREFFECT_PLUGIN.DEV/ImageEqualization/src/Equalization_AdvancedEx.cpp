#include "FastAriphmetics.hpp"
#include "ImageEqualization.hpp"
#include "ColorTransform.hpp"
#include "CommonAuxPixFormat.hpp"
#include "ImageEqualizationColorTransform.hpp"


void fCIELabHueImprove
(
	fCIELabPix* __restrict pLabSrc,
	float*      __restrict cs_out,
	A_long sizeX,
	A_long sizeY,
	float  alpha,
	float  ligntness
) noexcept
{
	float cs_out_max{ FLT_MIN };
	const A_long imgSize{ sizeX * sizeY };
	A_long i;

	__VECTOR_ALIGNED__
	for (i = 0; i < imgSize; i++)
	{
		auto const& a = pLabSrc[i].a;
		auto const& b = pLabSrc[i].b;
		cs_out[i] = FastCompute::Sqrt(a * a + b * b);
		cs_out_max = FastCompute::Max(cs_out[i], cs_out_max);
	}

	const float reciprocCsOutMax = 1.f / cs_out_max;
	constexpr float reciproc180 = 1.f / 180.f;

	/* lambda for tone mapping change  */
	auto const tone_map_inc = [&](const float w) {
		constexpr float toneMin = 0.2f;
		constexpr float toneMax = 0.8f;
		constexpr float toneDiff = toneMax - toneMin;
		constexpr float toneDiffReciproc = 1.f / toneDiff;
		return ((w < toneMin) ? 0.f : (w > toneMax) ? 1.f : (w - toneMin) * toneDiffReciproc);
	};

	const float lightness_reciproc = 1.f / ligntness;
	const float ligh_reciproc_100 = 1.f / 100.f;

	__VECTOR_ALIGNED__
	for (i = 0; i < imgSize; i++)
	{
		auto const L = pLabSrc[i].L;
		auto const a = pLabSrc[i].a;
		auto const b = pLabSrc[i].b;
		const float wc = tone_map_inc (cs_out[i] * reciprocCsOutMax);
		const float hangle_out2 = FastCompute::Atan2 (b, a) * reciproc180;
		constexpr float alpha = -10.f;
		const float expVal = FastCompute::Exp (alpha * hangle_out2 * hangle_out2);
		const float wx = 3.f * wc * expVal + 1.f;

		/* change LAB a and b components */
		pLabSrc[i].a = wx * a;
		pLabSrc[i].b = wx * b;
		/* lightness enhanecemnt */
		pLabSrc[i].L = 100.f * FastCompute::Pow (L * ligh_reciproc_100, lightness_reciproc);

		const float cs_out = FastCompute::Sqrt (pLabSrc[i].a * pLabSrc[i].a + pLabSrc[i].b * pLabSrc[i].b);
		const float h_out  = ((cs_out >= 0.1f) ? pLabSrc[i].b / pLabSrc[i].a : 0.f);
		const float hangle_out = FastCompute::Atan2 (pLabSrc[i].b, pLabSrc[i].a) + ((pLabSrc[i].b < 0.f) ? 360.f : 0.f);

		auto const sign = [&](const float f) {
			return (f > 0.f ? 1.f : (f < 0.f) ? -1.f : 0.f);
		};

		constexpr float reciproc116 = 1.f / 116.f;
		const float fY = (pLabSrc[i].L + 16.f) * reciproc116;
		const float sq_h_out = h_out * h_out;
		const float fX =  sign(pLabSrc[i].a) * cs_out / (500.f * FastCompute::Sqrt(1.f + sq_h_out)) + fY;
		const float fZ = -sign(pLabSrc[i].b) * cs_out / (200.f * FastCompute::Sqrt(1.f + 1.f / sq_h_out)) + fY;

	}

	return;
}

