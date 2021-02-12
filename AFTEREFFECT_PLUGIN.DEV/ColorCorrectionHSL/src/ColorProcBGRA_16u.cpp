#include "ColorCorrectionHSL.hpp"
#include "ColorConverts.hpp"


PF_Err prProcessImage_BGRA_4444_16u_HSL
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
	const PF_LayerDef*       __restrict pfLayer  = reinterpret_cast<const PF_LayerDef* __restrict>(&params[COLOR_CORRECT_INPUT]->u.ld);
	const PF_Pixel_BGRA_16u* __restrict localSrc = reinterpret_cast<const PF_Pixel_BGRA_16u* __restrict>(pfLayer->data);
	PF_Pixel_BGRA_16u*       __restrict localDst = reinterpret_cast<PF_Pixel_BGRA_16u* __restrict>(output->data);

	constexpr float reciproc32767 = 1.f / 32767.f;
	constexpr float reciproc360 = 1.f / 360.f;

	auto const& height = pfLayer->extent_hint.bottom - pfLayer->extent_hint.top;
	auto const& width  = pfLayer->extent_hint.right - pfLayer->extent_hint.left;
	auto const& line_pitch = pfLayer->rowbytes / static_cast<A_long>(PF_Pixel_BGRA_16u_size);

	PF_Pixel_BGRA_16u finalPixel{};
	float newR, newG, newB;

	for (auto j = 0; j < height; j++)
	{
		auto const& line_idx = j * line_pitch;

		__VECTOR_ALIGNED__
		for (auto i = 0; i < width; i++)
		{
			PF_Pixel_BGRA_16u const& srcPixel = localSrc[line_idx + i];
			float const& R = static_cast<float const>(srcPixel.R) * reciproc32767;
			float const& G = static_cast<float const>(srcPixel.G) * reciproc32767;
			float const& B = static_cast<float const>(srcPixel.B) * reciproc32767;
			auto  const& A = srcPixel.A;

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

			finalPixel.A = A;
			finalPixel.R = static_cast<A_u_short>(CLAMP_VALUE(newR * 32767.f, 0.f, 32767.f));
			finalPixel.G = static_cast<A_u_short>(CLAMP_VALUE(newG * 32767.f, 0.f, 32767.f));
			finalPixel.B = static_cast<A_u_short>(CLAMP_VALUE(newB * 32767.f, 0.f, 32767.f));

			/* put to output buffer updated value */
			localDst[i + line_idx] = finalPixel;

		} /* for (i = 0; i < width; i++) */

	} /* for (j = 0; j < height; j++) */

	return PF_Err_NONE;
}


PF_Err prProcessImage_BGRA_4444_16u_HSV
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
	const PF_LayerDef*       __restrict pfLayer = reinterpret_cast<const PF_LayerDef* __restrict>(&params[COLOR_CORRECT_INPUT]->u.ld);
	const PF_Pixel_BGRA_16u* __restrict localSrc = reinterpret_cast<const PF_Pixel_BGRA_16u* __restrict>(pfLayer->data);
	PF_Pixel_BGRA_16u*       __restrict localDst = reinterpret_cast<PF_Pixel_BGRA_16u* __restrict>(output->data);

	auto const& height = pfLayer->extent_hint.bottom - pfLayer->extent_hint.top;
	auto const& width = pfLayer->extent_hint.right - pfLayer->extent_hint.left;
	auto const& line_pitch = pfLayer->rowbytes / static_cast<A_long>(PF_Pixel_BGRA_16u_size);

	constexpr float reciproc32767 = 1.0f / 32767.0f;

	PF_Pixel_BGRA_16u finalPixel{};
	float hue, saturation, value, fR, fG, fB;

	for (auto j = 0; j < height; j++)
	{
		auto const& line_idx = j * line_pitch;

		__VECTOR_ALIGNED__
		for (auto i = 0; i < width; i++)
		{
			PF_Pixel_BGRA_16u const& srcPixel = localSrc[line_idx + i];
			float const& R = static_cast<float const>(srcPixel.R) * reciproc32767;
			float const& G = static_cast<float const>(srcPixel.G) * reciproc32767;
			float const& B = static_cast<float const>(srcPixel.B) * reciproc32767;
			auto  const& A = srcPixel.A;

			sRgb2hsv(R, G, B, hue, saturation, value);

			/* correct HSV */
			auto newHue = CLAMP_VALUE(hue + add_hue, 0.f, 360.f);
			auto const& newSat = CLAMP_VALUE(saturation + add_sat * 0.01f, 0.0f, 1.0f);
			auto const& newVal = CLAMP_VALUE(value + add_val * 0.01f, 0.0f, 1.0f);

			/* back convert to RGB */
			hsv2sRgb(newHue, newSat, newVal, fR, fG, fB);

			finalPixel.A = A;
			finalPixel.R = static_cast<A_u_short>(CLAMP_VALUE(fR * 32767.f, 0.f, 32767.f));
			finalPixel.G = static_cast<A_u_short>(CLAMP_VALUE(fG * 32767.f, 0.f, 32767.f));
			finalPixel.B = static_cast<A_u_short>(CLAMP_VALUE(fB * 32767.f, 0.f, 32767.f));

			/* put to output buffer updated value */
			localDst[i + line_idx] = finalPixel;

		} /* for (i = 0; i < width; i++) */

	} /* for (j = 0; j < height; j++) */

	return PF_Err_NONE;
}
