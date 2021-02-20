#include "ColorCorrectionHSL.hpp"
#include "ColorTransformMatrix.hpp"
#include "ColorConverts.hpp"

PF_Err prProcessImage_VUYA_4444_8u_HSL
(
	PF_InData*		in_data,
	PF_OutData*		out_data,
	PF_ParamDef*	params[],
	PF_LayerDef*	output,
	float           add_hue,
	float           add_sat,
	float           add_lum,
	const bool&     isBT709
) noexcept
{
	const PF_LayerDef*       __restrict pfLayer  = reinterpret_cast<const PF_LayerDef* __restrict>(&params[COLOR_CORRECT_INPUT]->u.ld);
	const PF_Pixel_VUYA_8u*  __restrict localSrc = reinterpret_cast<const PF_Pixel_VUYA_8u* __restrict>(pfLayer->data);
	PF_Pixel_VUYA_8u*        __restrict localDst = reinterpret_cast<PF_Pixel_VUYA_8u* __restrict>(output->data);

	const float* __restrict yuv2rgb = YUV2RGB[isBT709];
	const float* __restrict rgb2yuv = RGB2YUV[isBT709];

	auto const& height = pfLayer->extent_hint.bottom - pfLayer->extent_hint.top;
	auto const& width  = pfLayer->extent_hint.right - pfLayer->extent_hint.left;
	auto const& line_pitch = pfLayer->rowbytes / static_cast<A_long>(PF_Pixel_VUYA_8u_size);

	PF_Pixel_VUYA_8u finalPixel{};
	float newR, newG, newB;
	constexpr float reciproc255 = 1.0f / 255.0f;
	constexpr float reciproc360 = 1.0f / 360.f;

	for (auto j = 0; j < height; j++)
	{
		auto const& line_idx = j * line_pitch;

		__VECTOR_ALIGNED__
		for (auto i = 0; i < width; i++)
		{
			PF_Pixel_VUYA_8u const& srcPixel = localSrc[line_idx + i];

			float const& Y = static_cast<float>(srcPixel.Y);
			float const& U = static_cast<float>(srcPixel.U) - 128.0f;
			float const& V = static_cast<float>(srcPixel.V) - 128.0f;
			auto  const& A = srcPixel.A;

			auto const& R = (Y * yuv2rgb[0] + U * yuv2rgb[1] + V * yuv2rgb[2]) * reciproc255;
			auto const& G = (Y * yuv2rgb[3] + U * yuv2rgb[4] + V * yuv2rgb[5]) * reciproc255;
			auto const& B = (Y * yuv2rgb[6] + U * yuv2rgb[7] + V * yuv2rgb[8]) * reciproc255;

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

			newR = CLAMP_VALUE(newR * 255.f, 0.f, 255.f);
			newG = CLAMP_VALUE(newG * 255.f, 0.f, 255.f);
			newB = CLAMP_VALUE(newB * 255.f, 0.f, 255.f);

			finalPixel.A = A;
			finalPixel.Y = static_cast<A_u_char>(newR * rgb2yuv[0] + newG * rgb2yuv[1] + newB * rgb2yuv[2]);
			finalPixel.U = static_cast<A_u_char>(newR * rgb2yuv[3] + newG * rgb2yuv[4] + newB * rgb2yuv[5] + 128.f);
			finalPixel.V = static_cast<A_u_char>(newR * rgb2yuv[6] + newG * rgb2yuv[7] + newB * rgb2yuv[8] + 128.f);

			/* put to output buffer updated value */
			localDst[line_idx + i] = finalPixel;

		} /* for (i = 0; i < width; i++) */

	} /* for (j = 0; j < height; j++) */

	return PF_Err_NONE;
}


PF_Err prProcessImage_VUYA_4444_8u_HSV
(
	PF_InData*		in_data,
	PF_OutData*		out_data,
	PF_ParamDef*	params[],
	PF_LayerDef*	output,
	float           add_hue,
	float           add_sat,
	float           add_val,
	const bool&     isBT709
) noexcept
{
	const PF_LayerDef*       __restrict pfLayer = reinterpret_cast<const PF_LayerDef* __restrict>(&params[COLOR_CORRECT_INPUT]->u.ld);
	const PF_Pixel_VUYA_8u*  __restrict localSrc = reinterpret_cast<const PF_Pixel_VUYA_8u* __restrict>(pfLayer->data);
	PF_Pixel_VUYA_8u*        __restrict localDst = reinterpret_cast<PF_Pixel_VUYA_8u* __restrict>(output->data);

	const float* __restrict yuv2rgb = YUV2RGB[isBT709];
	const float* __restrict rgb2yuv = RGB2YUV[isBT709];

	auto const& height = pfLayer->extent_hint.bottom - pfLayer->extent_hint.top;
	auto const& width = pfLayer->extent_hint.right - pfLayer->extent_hint.left;
	auto const& line_pitch = pfLayer->rowbytes / static_cast<A_long>(PF_Pixel_VUYA_8u_size);

	PF_Pixel_VUYA_8u finalPixel{};
	float newR, newG, newB;
	constexpr float reciproc255 = 1.0f / 255.0f;

	for (auto j = 0; j < height; j++)
	{
		auto const& line_idx = j * line_pitch;

		__VECTOR_ALIGNED__
		for (auto i = 0; i < width; i++)
		{
			PF_Pixel_VUYA_8u const& srcPixel = localSrc[line_idx + i];
			
			float const& Y = static_cast<float>(srcPixel.Y);
			float const& U = static_cast<float>(srcPixel.U) - 128.0f;
			float const& V = static_cast<float>(srcPixel.V) - 128.0f;
			auto  const& A = srcPixel.A;

			auto const& R = (Y * yuv2rgb[0] + U * yuv2rgb[1] + V * yuv2rgb[2]) * reciproc255;
			auto const& G = (Y * yuv2rgb[3] + U * yuv2rgb[4] + V * yuv2rgb[5]) * reciproc255;
			auto const& B = (Y * yuv2rgb[6] + U * yuv2rgb[7] + V * yuv2rgb[8]) * reciproc255;

			/* start convert RGB to HSL color space */
			float hue, saturation, value;

			/* convert sRGB to HSL format */
			sRgb2hsv(R, G, B, hue, saturation, value);

			/* add values to HSV */
			auto newHue = CLAMP_VALUE(hue + add_hue, 0.f, 360.f);
			auto const& newSat = CLAMP_VALUE(saturation + add_sat * 0.01f, 0.0f, 1.0f);
			auto const& newVal = CLAMP_VALUE(value + add_val * 0.01f, 0.0f, 1.0f);

			/* back convert to sRGB space */
			hsv2sRgb(newHue, newSat, newVal, newR, newG, newB);

			newR = CLAMP_VALUE(newR * 255.f, 0.f, 255.f);
			newG = CLAMP_VALUE(newG * 255.f, 0.f, 255.f);
			newB = CLAMP_VALUE(newB * 255.f, 0.f, 255.f);

			finalPixel.A = A;
			finalPixel.Y = static_cast<A_u_char>(newR * rgb2yuv[0] + newG * rgb2yuv[1] + newB * rgb2yuv[2]);
			finalPixel.U = static_cast<A_u_char>(newR * rgb2yuv[3] + newG * rgb2yuv[4] + newB * rgb2yuv[5] + 128.f);
			finalPixel.V = static_cast<A_u_char>(newR * rgb2yuv[6] + newG * rgb2yuv[7] + newB * rgb2yuv[8] + 128.f);

			/* put to output buffer updated value */
			localDst[line_idx + i] = finalPixel;

		} /* for (i = 0; i < width; i++) */

	} /* for (j = 0; j < height; j++) */

	return PF_Err_NONE;
}

