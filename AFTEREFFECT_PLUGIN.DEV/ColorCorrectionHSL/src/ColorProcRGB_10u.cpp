#include "ColorCorrectionHSL.hpp"

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
	constexpr float reciproc3 = 1.0f / 3.0f;

	PF_Pixel_RGB_10u finalPixel{};

	for (auto j = 0; j < height; j++)
	{
		auto const& line_idx = j * line_pitch;

		__VECTOR_ALIGNED__
		for (auto i = 0; i < width; i++)
		{
			PF_Pixel_RGB_10u const& srcPixel = localSrc[line_idx + i];
			float const& R = static_cast<float const>(srcPixel.R) / 1024.0f;
			float const& G = static_cast<float const>(srcPixel.G) / 1024.0f;
			float const& B = static_cast<float const>(srcPixel.B) / 1024.0f;

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
			luminance += add_lum;

			auto const& newHue = CLAMP_H(hue) / 360.f;
			auto const& newSat = CLAMP_LS(saturation) / 100.f;
			auto const& newLum = CLAMP_LS(luminance) / 100.f;
			
			/* back convert to RGB space */
			if (0.f == newSat)
			{
				finalPixel.R = finalPixel.G = finalPixel.B = static_cast<A_u_long>(CLAMP_VALUE(newLum * 1024.0f, 0.f, 1023.0f));
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

				auto const& fR = restore_rgb_channel_value(tmpVal1, tmpVal2, tmpR);
				auto const& fG = restore_rgb_channel_value(tmpVal1, tmpVal2, tmpG);
				auto const& fB = restore_rgb_channel_value(tmpVal1, tmpVal2, tmpB);

				finalPixel.R = static_cast<A_u_long>(CLAMP_VALUE(fR * 1024.f, 0.f, 1023.f));
				finalPixel.G = static_cast<A_u_long>(CLAMP_VALUE(fG * 1024.f, 0.f, 1023.f));
				finalPixel.B = static_cast<A_u_long>(CLAMP_VALUE(fB * 1024.f, 0.f, 1023.f));
			}

			/* put to output buffer updated value */
			localDst[i + line_idx] = finalPixel;

		} /* for (i = 0; i < width; i++) */

	} /* for (j = 0; j < height; j++) */

	return PF_Err_NONE;
}

