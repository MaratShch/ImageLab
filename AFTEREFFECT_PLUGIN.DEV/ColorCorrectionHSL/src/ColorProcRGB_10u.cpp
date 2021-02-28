#include "ColorCorrectionHSL.hpp"
#include "ColorConverts.hpp"


PF_Err prProcessImage_RGB_444_10u_HSL
(
	PF_InData*		in_data,
	PF_OutData*		out_data,
	PF_ParamDef*	params[],
	PF_LayerDef*	output,
	float           add_hue,
	float           add_sat,
	float           add_lum
) noexcept
{
	const PF_LayerDef*      __restrict pfLayer  = reinterpret_cast<const PF_LayerDef* __restrict>(&params[COLOR_CORRECT_INPUT]->u.ld);
	const PF_Pixel_RGB_10u* __restrict localSrc = reinterpret_cast<const PF_Pixel_RGB_10u* __restrict>(pfLayer->data);
	PF_Pixel_RGB_10u*       __restrict localDst = reinterpret_cast<PF_Pixel_RGB_10u* __restrict>(output->data);

	auto const& height = pfLayer->extent_hint.bottom - pfLayer->extent_hint.top;
	auto const& width = pfLayer->extent_hint.right - pfLayer->extent_hint.left;
	auto const& line_pitch = pfLayer->rowbytes / static_cast<A_long>(PF_Pixel_RGB_10u_size);

	constexpr float reciproc1023 = 1.0f / 1023.f;
	constexpr float reciproc360 = 1.0f / 360.f;

	PF_Pixel_RGB_10u finalPixel{};
	float newR, newG, newB;

	for (auto j = 0; j < height; j++)
	{
		auto const& line_idx = j * line_pitch;

		__VECTOR_ALIGNED__
		for (auto i = 0; i < width; i++)
		{
			PF_Pixel_RGB_10u const& srcPixel = localSrc[line_idx + i];
			float const& R = static_cast<float const>(srcPixel.R) * reciproc1023;
			float const& G = static_cast<float const>(srcPixel.G) * reciproc1023;
			float const& B = static_cast<float const>(srcPixel.B) * reciproc1023;

			/* start convert RGB to HSL color space */
			float hue, saturation, luminance;

			/* convert sRGB to HSL format */
			sRgb2hsl(R, G, B, hue, saturation, luminance);

			/* add values to HSL */
			hue += add_hue;
			saturation += add_sat;
			luminance += add_lum;

			auto const& newHue = CLAMP_H(hue) * reciproc360;
			auto const& newSat = CLAMP_LS(saturation) * 0.01f;
			auto const& newLum = CLAMP_LS(luminance)  * 0.01f;

			/* back convert to sRGB space */
			hsl2sRgb(newHue, newSat, newLum, newR, newG, newB);

			finalPixel.R = static_cast<A_u_short>(CLAMP_VALUE(newR * 1023.f, 0.f, 1023.f));
			finalPixel.G = static_cast<A_u_short>(CLAMP_VALUE(newG * 1023.f, 0.f, 1023.f));
			finalPixel.B = static_cast<A_u_short>(CLAMP_VALUE(newB * 1023.f, 0.f, 1023.f));

			/* put to output buffer updated value */
			localDst[i + line_idx] = finalPixel;

		} /* for (i = 0; i < width; i++) */

	} /* for (j = 0; j < height; j++) */

	return PF_Err_NONE;
}


PF_Err prProcessImage_RGB_444_10u_HSV
(
	PF_InData*		in_data,
	PF_OutData*		out_data,
	PF_ParamDef*	params[],
	PF_LayerDef*	output,
	float           add_hue,
	float           add_sat,
	float           add_val
) noexcept
{
	const PF_LayerDef*      __restrict pfLayer = reinterpret_cast<const PF_LayerDef* __restrict>(&params[COLOR_CORRECT_INPUT]->u.ld);
	const PF_Pixel_RGB_10u* __restrict localSrc = reinterpret_cast<const PF_Pixel_RGB_10u* __restrict>(pfLayer->data);
	PF_Pixel_RGB_10u*       __restrict localDst = reinterpret_cast<PF_Pixel_RGB_10u* __restrict>(output->data);

	auto const& height = pfLayer->extent_hint.bottom - pfLayer->extent_hint.top;
	auto const& width = pfLayer->extent_hint.right - pfLayer->extent_hint.left;
	auto const& line_pitch = pfLayer->rowbytes / static_cast<A_long>(PF_Pixel_RGB_10u_size);

	constexpr float reciproc1023 = 1.0f / 1023.0f;

	PF_Pixel_RGB_10u finalPixel{};
	float hue, saturation, value, fR, fG, fB;

	for (auto j = 0; j < height; j++)
	{
		auto const& line_idx = j * line_pitch;

		__VECTOR_ALIGNED__
		for (auto i = 0; i < width; i++)
		{
			PF_Pixel_RGB_10u const& srcPixel = localSrc[line_idx + i];
			float const& R = static_cast<float const>(srcPixel.R) * reciproc1023;
			float const& G = static_cast<float const>(srcPixel.G) * reciproc1023;
			float const& B = static_cast<float const>(srcPixel.B) * reciproc1023;

			sRgb2hsv(R, G, B, hue, saturation, value);

			/* correct HSV */
			auto newHue = CLAMP_VALUE(hue + add_hue, 0.f, 360.f);
			auto const& newSat = CLAMP_VALUE(saturation + add_sat * 0.01f, 0.0f, 1.0f);
			auto const& newVal = CLAMP_VALUE(value + add_val * 0.01f, 0.0f, 1.0f);

			/* back convert to RGB */
			hsv2sRgb(newHue, newSat, newVal, fR, fG, fB);

			finalPixel.R = static_cast<A_u_long>(CLAMP_VALUE(fR * 1023.f, 0.f, 1023.f));
			finalPixel.G = static_cast<A_u_long>(CLAMP_VALUE(fG * 1023.f, 0.f, 1023.f));
			finalPixel.B = static_cast<A_u_long>(CLAMP_VALUE(fB * 1023.f, 0.f, 1023.f));

			/* put to output buffer updated value */
			localDst[i + line_idx] = finalPixel;

		} /* for (i = 0; i < width; i++) */

	} /* for (j = 0; j < height; j++) */

	return PF_Err_NONE;
}


PF_Err prProcessImage_RGB_444_10u_HSI
(
	PF_InData*		in_data,
	PF_OutData*		out_data,
	PF_ParamDef*	params[],
	PF_LayerDef*	output,
	float           add_hue,
	float           add_sat,
	float           add_int
) noexcept
{
	const PF_LayerDef*       __restrict pfLayer = reinterpret_cast<const PF_LayerDef* __restrict>(&params[COLOR_CORRECT_INPUT]->u.ld);
	const PF_Pixel_RGB_10u*  __restrict localSrc = reinterpret_cast<const PF_Pixel_RGB_10u* __restrict>(pfLayer->data);
	PF_Pixel_RGB_10u*        __restrict localDst = reinterpret_cast<PF_Pixel_RGB_10u* __restrict>(output->data);

	auto const& height = pfLayer->extent_hint.bottom - pfLayer->extent_hint.top;
	auto const& width = pfLayer->extent_hint.right - pfLayer->extent_hint.left;
	auto const& line_pitch = pfLayer->rowbytes / static_cast<A_long>(PF_Pixel_RGB_10u_size);

	constexpr float reciproc1024 = 1.0f / 1024.0f;
	constexpr float reciproc100 = 1.0f / 100.0f;

	PF_Pixel_RGB_10u finalPixel{};
	float newR, newG, newB;

	for (auto j = 0; j < height; j++)
	{
		auto const& line_idx = j * line_pitch;

		__VECTOR_ALIGNED__
		for (auto i = 0; i < width; i++)
		{
			PF_Pixel_RGB_10u const& srcPixel = localSrc[line_idx + i];
			float const& R = static_cast<float const>(srcPixel.R) * reciproc1024;
			float const& G = static_cast<float const>(srcPixel.G) * reciproc1024;
			float const& B = static_cast<float const>(srcPixel.B) * reciproc1024;

			float hue, saturation, intencity;

			/* convert sRGB to HSL format */
			sRgb2hsi(R, G, B, hue, saturation, intencity);

			/* add values to HSL */
			auto const& newHue = CLAMP_VALUE(hue + add_hue, 0.f, 360.f);
			auto const& newSat = CLAMP_VALUE(saturation + add_sat * reciproc100, 0.f, 1.0f);
			auto const& newInt = CLAMP_VALUE(intencity + add_int * reciproc100, 0.f, 1.0f);

			/* back convert to sRGB space */
			hsi2sRgb(newHue, newSat, newInt, newR, newG, newB);

			finalPixel.R = static_cast<A_u_long>(CLAMP_VALUE(newR * 1024.0f, 0.f, 1023.0f));
			finalPixel.G = static_cast<A_u_long>(CLAMP_VALUE(newG * 1024.0f, 0.f, 1023.0f));
			finalPixel.B = static_cast<A_u_long>(CLAMP_VALUE(newB * 1024.0f, 0.f, 1023.0f));

			/* put to output buffer updated value */
			localDst[i + line_idx] = finalPixel;

		} /* for (i = 0; i < width; i++) */

	} /* for (j = 0; j < height; j++) */

	return PF_Err_NONE;
}


PF_Err prProcessImage_RGB_444_10u_HSP
(
	PF_InData*		in_data,
	PF_OutData*		out_data,
	PF_ParamDef*	params[],
	PF_LayerDef*	output,
	float           add_hue,
	float           add_sat,
	float           add_per
) noexcept
{
	const PF_LayerDef*       __restrict pfLayer = reinterpret_cast<const PF_LayerDef* __restrict>(&params[COLOR_CORRECT_INPUT]->u.ld);
	const PF_Pixel_RGB_10u*  __restrict localSrc = reinterpret_cast<const PF_Pixel_RGB_10u* __restrict>(pfLayer->data);
	PF_Pixel_RGB_10u*        __restrict localDst = reinterpret_cast<PF_Pixel_RGB_10u* __restrict>(output->data);

	auto const& height = pfLayer->extent_hint.bottom - pfLayer->extent_hint.top;
	auto const& width = pfLayer->extent_hint.right - pfLayer->extent_hint.left;
	auto const& line_pitch = pfLayer->rowbytes / static_cast<A_long>(PF_Pixel_RGB_10u_size);

	constexpr float reciproc1024 = 1.0f / 1024.0f;
	constexpr float reciproc360 = 1.0f / 360.0f;
	constexpr float reciproc100 = 1.0f / 100.0f;

	PF_Pixel_RGB_10u finalPixel{};
	float newR, newG, newB;

	for (auto j = 0; j < height; j++)
	{
		auto const& line_idx = j * line_pitch;

		__VECTOR_ALIGNED__
			for (auto i = 0; i < width; i++)
			{
				PF_Pixel_RGB_10u const& srcPixel = localSrc[line_idx + i];
				float const& R = static_cast<float const>(srcPixel.R) * reciproc1024;
				float const& G = static_cast<float const>(srcPixel.G) * reciproc1024;
				float const& B = static_cast<float const>(srcPixel.B) * reciproc1024;

				float hue, saturation, percistant_brignthness;

				/* convert sRGB to HSP format */
				sRgb2hsp(R, G, B, hue, saturation, percistant_brignthness);

				/* add values to HSL */
				auto const& newHue = CLAMP_VALUE(hue + add_hue * reciproc360, 0.f, 1.0f);
				auto const& newSat = CLAMP_VALUE(saturation + add_sat * reciproc100, 0.f, 1.0f);
				auto const& newPer = CLAMP_VALUE(percistant_brignthness + add_per * reciproc100, 0.f, 1.0f);

				/* back convert to sRGB space */
				hsp2sRgb(newHue, newSat, newPer, newR, newG, newB);

				finalPixel.R = static_cast<A_u_long>(CLAMP_VALUE(newR * 1024.0f, 0.f, 1023.f));
				finalPixel.G = static_cast<A_u_long>(CLAMP_VALUE(newG * 1024.0f, 0.f, 1023.f));
				finalPixel.B = static_cast<A_u_long>(CLAMP_VALUE(newB * 1024.0f, 0.f, 1023.f));

				/* put to output buffer updated value */
				localDst[i + line_idx] = finalPixel;

			} /* for (i = 0; i < width; i++) */

	} /* for (j = 0; j < height; j++) */

	return PF_Err_NONE;
}


PF_Err prProcessImage_RGB_444_10u_HSLuv
(
	PF_InData*		in_data,
	PF_OutData*		out_data,
	PF_ParamDef*	params[],
	PF_LayerDef*	output,
	float           add_hue,
	float           add_sat,
	float           add_luv
) noexcept
{
	const PF_LayerDef*       __restrict pfLayer = reinterpret_cast<const PF_LayerDef* __restrict>(&params[COLOR_CORRECT_INPUT]->u.ld);
	const PF_Pixel_RGB_10u*  __restrict localSrc = reinterpret_cast<const PF_Pixel_RGB_10u* __restrict>(pfLayer->data);
	PF_Pixel_RGB_10u*        __restrict localDst = reinterpret_cast<PF_Pixel_RGB_10u* __restrict>(output->data);

	auto const& height = pfLayer->extent_hint.bottom - pfLayer->extent_hint.top;
	auto const& width = pfLayer->extent_hint.right - pfLayer->extent_hint.left;
	auto const& line_pitch = pfLayer->rowbytes / static_cast<A_long>(PF_Pixel_RGB_10u_size);

	constexpr float reciproc1024 = 1.0f / 1024.0f;

	PF_Pixel_RGB_10u finalPixel{};
	float newR = 0.f, newG = 0.f, newB = 0.f;

	for (auto j = 0; j < height; j++)
	{
		auto const& line_idx = j * line_pitch;

		__VECTOR_ALIGNED__
		for (auto i = 0; i < width; i++)
		{
			PF_Pixel_RGB_10u const& srcPixel = localSrc[line_idx + i];
			float const& R = static_cast<float const>(srcPixel.R) * reciproc1024;
			float const& G = static_cast<float const>(srcPixel.G) * reciproc1024;
			float const& B = static_cast<float const>(srcPixel.B) * reciproc1024;

			float hue, saturation, luv;

			/* convert sRGB to HSLuv format */
			sRgb2hsLuv(R, G, B, hue, saturation, luv);

			/* add values to HSLuv */
			auto const& newHue = CLAMP_VALUE(hue + add_hue, 0.f, 360.0f);
			auto const& newSat = CLAMP_VALUE(saturation + add_sat, 0.f, 100.0f);
			auto const& newLuv = CLAMP_VALUE(luv + add_luv, 0.f, 100.0f);

			/* back convert to sRGB space */
			hsLuv2sRgb(newHue, newSat, newLuv, newR, newG, newB);

			finalPixel.R = static_cast<A_u_long>(CLAMP_VALUE(newR * 1024.0f, 0.f, 1023.f));
			finalPixel.G = static_cast<A_u_long>(CLAMP_VALUE(newG * 1024.0f, 0.f, 1023.f));
			finalPixel.B = static_cast<A_u_long>(CLAMP_VALUE(newB * 1024.0f, 0.f, 1023.f));

			/* put to output buffer updated value */
			localDst[i + line_idx] = finalPixel;

		} /* for (i = 0; i < width; i++) */

	} /* for (j = 0; j < height; j++) */

	return PF_Err_NONE;
}


PF_Err prProcessImage_RGB_444_10u_HPLuv
(
	PF_InData*		in_data,
	PF_OutData*		out_data,
	PF_ParamDef*	params[],
	PF_LayerDef*	output,
	float           add_hue,
	float           add_p,
	float           add_luv
) noexcept
{
	const PF_LayerDef*       __restrict pfLayer = reinterpret_cast<const PF_LayerDef* __restrict>(&params[COLOR_CORRECT_INPUT]->u.ld);
	const PF_Pixel_RGB_10u*  __restrict localSrc = reinterpret_cast<const PF_Pixel_RGB_10u* __restrict>(pfLayer->data);
	PF_Pixel_RGB_10u*        __restrict localDst = reinterpret_cast<PF_Pixel_RGB_10u* __restrict>(output->data);

	auto const& height = pfLayer->extent_hint.bottom - pfLayer->extent_hint.top;
	auto const& width = pfLayer->extent_hint.right - pfLayer->extent_hint.left;
	auto const& line_pitch = pfLayer->rowbytes / static_cast<A_long>(PF_Pixel_RGB_10u_size);

	constexpr float reciproc1024 = 1.0f / 1024.0f;

	PF_Pixel_RGB_10u finalPixel{};
	float newR = 0.f, newG = 0.f, newB = 0.f;

	for (auto j = 0; j < height; j++)
	{
		auto const& line_idx = j * line_pitch;

		__VECTOR_ALIGNED__
		for (auto i = 0; i < width; i++)
		{
			PF_Pixel_RGB_10u const& srcPixel = localSrc[line_idx + i];
			float const& R = static_cast<float const>(srcPixel.R) * reciproc1024;
			float const& G = static_cast<float const>(srcPixel.G) * reciproc1024;
			float const& B = static_cast<float const>(srcPixel.B) * reciproc1024;

			float hue, per, luv;

			/* convert sRGB to HSLuv format */
			sRgb2hsLuv(R, G, B, hue, per, luv);

			/* add values to HSLuv */
			auto const& newHue = CLAMP_VALUE(hue + add_hue, 0.f, 360.0f);
			auto const& newPer = CLAMP_VALUE(per + add_p, 0.f, 100.0f);
			auto const& newLuv = CLAMP_VALUE(luv + add_luv, 0.f, 100.0f);

			/* back convert to sRGB space */
			hsLuv2sRgb(newHue, newPer, newLuv, newR, newG, newB);

			finalPixel.R = static_cast<A_u_long>(CLAMP_VALUE(newR * 1024.0f, 0.f, 1023.f));
			finalPixel.G = static_cast<A_u_long>(CLAMP_VALUE(newG * 1024.0f, 0.f, 1023.f));
			finalPixel.B = static_cast<A_u_long>(CLAMP_VALUE(newB * 1024.0f, 0.f, 1023.f));

			/* put to output buffer updated value */
			localDst[i + line_idx] = finalPixel;

		} /* for (i = 0; i < width; i++) */

	} /* for (j = 0; j < height; j++) */

	return PF_Err_NONE;
}