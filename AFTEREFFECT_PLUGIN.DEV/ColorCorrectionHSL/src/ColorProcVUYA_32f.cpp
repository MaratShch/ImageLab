#include "ColorCorrectionHSL.hpp"
#include "ColorTransformMatrix.hpp"

PF_Err prProcessImage_VUYA_4444_32f_HSL
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
		const PF_Pixel_VUYA_32f* __restrict localSrc = reinterpret_cast<const PF_Pixel_VUYA_32f* __restrict>(pfLayer->data);
		PF_Pixel_VUYA_32f*       __restrict localDst = reinterpret_cast<PF_Pixel_VUYA_32f* __restrict>(output->data);

		const float* __restrict yuv2rgb = YUV2RGB[isBT709];
		const float* __restrict rgb2yuv = RGB2YUV[isBT709];

		auto const& height = pfLayer->extent_hint.bottom - pfLayer->extent_hint.top;
		auto const& width  = pfLayer->extent_hint.right - pfLayer->extent_hint.left;
		auto const& line_pitch = pfLayer->rowbytes / static_cast<A_long>(PF_Pixel_VUYA_32f_size);
		constexpr float reciproc3 = 1.0f / 3.0f;

		PF_Pixel_VUYA_32f finalPixel{};

		for (auto j = 0; j < height; j++)
		{
			auto const& line_idx = j * line_pitch;

			__VECTOR_ALIGNED__
			for (auto i = 0; i < width; i++)
			{
				PF_Pixel_VUYA_32f const& srcPixel = localSrc[line_idx + i];

				auto const& A = srcPixel.A;
				auto const& R = (srcPixel.Y * yuv2rgb[0] + srcPixel.U * yuv2rgb[1] + srcPixel.V * yuv2rgb[2]);
				auto const& G = (srcPixel.Y * yuv2rgb[3] + srcPixel.U * yuv2rgb[4] + srcPixel.V * yuv2rgb[5]);
				auto const& B = (srcPixel.Y * yuv2rgb[6] + srcPixel.U * yuv2rgb[7] + srcPixel.V * yuv2rgb[8]);

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

				/* add values to HSL */
				hue += add_hue;
				saturation += add_sat;
				luminance  += add_lum;

				auto const& newHue = CLAMP_H(hue) / 360.f;
				auto const& newSat = CLAMP_LS(saturation) / 100.f;
				auto const& newLum = CLAMP_LS(luminance)  / 100.f;

				/* back convert to RGB space */
				if (0.01f > newSat)
				{
					auto const& newValue = CLAMP_VALUE(newLum, f32_value_black, f32_value_white);
					finalPixel.A = A;
					finalPixel.Y = newValue * rgb2yuv[0] + newValue * rgb2yuv[1] + newValue * rgb2yuv[2];
					finalPixel.U = newValue * rgb2yuv[3] + newValue * rgb2yuv[4] + newValue * rgb2yuv[5];
					finalPixel.V = newValue * rgb2yuv[6] + newValue * rgb2yuv[7] + newValue * rgb2yuv[8];
				}
				else
				{
					float tmpVal1, tmpVal2;
					tmpVal2 = (newLum < 0.50f) ? (newLum * (1.0f + newSat)) : (newLum + newSat - (newLum * newSat));
					tmpVal1 = 2.0f * newLum - tmpVal2;

					auto const& tmpG = newHue;
					auto tmpR = newHue + reciproc3;
					auto tmpB = newHue - reciproc3;

					tmpR -= ((tmpR > 1.0f) ? 1.0f : 0.0f);
					tmpB += ((tmpB < 0.0f) ? 1.0f : 0.0f);

					auto const& fR = CLAMP_VALUE(restore_rgb_channel_value(tmpVal1, tmpVal2, tmpR), f32_value_black, f32_value_white);
					auto const& fG = CLAMP_VALUE(restore_rgb_channel_value(tmpVal1, tmpVal2, tmpG), f32_value_black, f32_value_white);
					auto const& fB = CLAMP_VALUE(restore_rgb_channel_value(tmpVal1, tmpVal2, tmpB), f32_value_black, f32_value_white);

					finalPixel.A = A;
					finalPixel.Y = fR * rgb2yuv[0] + fG * rgb2yuv[1] + fB * rgb2yuv[2];
					finalPixel.U = fR * rgb2yuv[3] + fG * rgb2yuv[4] + fB * rgb2yuv[5];
					finalPixel.V = fR * rgb2yuv[6] + fG * rgb2yuv[7] + fB * rgb2yuv[8];
				}
				
				/* put to output buffer updated value */
				localDst[line_idx + i] = finalPixel;

			} /* for (i = 0; i < width; i++) */

		} /* for (j = 0; j < height; j++) */

	return PF_Err_NONE;
}

