#include "ImageStylization.hpp"
#include "StylizationStructs.hpp"
#include "PrSDKAESupport.h"
#include "ColorTransformMatrix.hpp"
#include "FastAriphmetics.hpp"

#include <mutex>

constexpr float div255 = 1.f / 255.f;
constexpr float div32768 = 1.f / 32768.f;

constexpr float qH = 6.f;
constexpr float qS = 5.f;
constexpr float qI = 5.f;

constexpr int nbinsH = static_cast<int>(hist_size_H / qH);
constexpr int nbinsS = static_cast<int>(hist_size_S / qS);
constexpr int nbinsI = static_cast<int>(hist_size_I / qI);


inline void sRgb2hsi(const float& R, const float& G, const float& B, float& H, float& S, float& I) noexcept
{
	constexpr float reciproc3 = 1.f / 3.f;
	constexpr float denom = 1.f / 1e7f;
	constexpr float reciprocPi180 = 180.f / FastCompute::PI;

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

	float h = H - 360.f * floor(H * reciproc360);
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



static PF_Err PR_ImageStyle_CartoonEffect_BGRA_8u
(
	PF_InData*   __restrict in_data,
	PF_OutData*  __restrict out_data,
	PF_ParamDef* __restrict params[],
	PF_LayerDef* __restrict output
) noexcept
{
	const PF_LayerDef*      __restrict pfLayer = reinterpret_cast<const PF_LayerDef* __restrict>(&params[IMAGE_STYLE_INPUT]->u.ld);
	const PF_Pixel_BGRA_8u* __restrict localSrc = reinterpret_cast<const PF_Pixel_BGRA_8u* __restrict>(pfLayer->data);
	PF_Pixel_BGRA_8u*       __restrict localDst = reinterpret_cast<PF_Pixel_BGRA_8u* __restrict>(output->data);

	auto const& height = pfLayer->extent_hint.bottom - pfLayer->extent_hint.top;
	auto const& width  = pfLayer->extent_hint.right  - pfLayer->extent_hint.left;
	auto const& line_pitch = pfLayer->rowbytes / static_cast<A_long>(PF_Pixel_BGRA_8u_size);

	constexpr float sMin = static_cast<float>(nbinsH) / FastCompute::PIx2; // compute minimum saturation value that prevents quantization problems 

	int j, i, nhighsat;
	float R, G, B;
	float H, S, I;

	j = i = nhighsat = 0;
	R = G = B = H = S = I = 0.f;

	/* first path - build the statistics about frame [build histogram] */
	for (j = 0; j < height; j++)
	{
		const A_long& line_idx = j * line_pitch;

		for (i = 0; i < width; i++)
		{
			/* convert RGB to sRGB */
			B = static_cast<float>(localSrc[line_idx + i].B) * div255;
			G = static_cast<float>(localSrc[line_idx + i].G) * div255;
			R = static_cast<float>(localSrc[line_idx + i].R) * div255;

			/* convert sRGB to HSI color space */
			sRgb2hsi (R, G, B, H, S, I);


		} /* for (i = 0; i < width; i++) */
	} /* for (j = 0; j < height; j++) */

	return PF_Err_NONE;
}



static PF_Err PR_ImageStyle_CartoonEffect_VUYA_8u
(
	PF_InData*   __restrict in_data,
	PF_OutData*  __restrict out_data,
	PF_ParamDef* __restrict params[],
	PF_LayerDef* __restrict output
) noexcept
{
	return PF_Err_NONE;
}


static PF_Err PR_ImageStyle_CartoonEffect_VUYA_32f
(
	PF_InData*   __restrict in_data,
	PF_OutData*  __restrict out_data,
	PF_ParamDef* __restrict params[],
	PF_LayerDef* __restrict output
) noexcept
{
	return PF_Err_NONE;
}


static PF_Err PR_ImageStyle_CartoonEffect_BGRA_16u
(
	PF_InData*   __restrict in_data,
	PF_OutData*  __restrict out_data,
	PF_ParamDef* __restrict params[],
	PF_LayerDef* __restrict output
) noexcept
{
	return PF_Err_NONE;
}


static PF_Err PR_ImageStyle_CartoonEffect_BGRA_32f
(
	PF_InData*   __restrict in_data,
	PF_OutData*  __restrict out_data,
	PF_ParamDef* __restrict params[],
	PF_LayerDef* __restrict output
) noexcept
{
	return PF_Err_NONE;
}




PF_Err PR_ImageStyle_CartoonEffect
(
	PF_InData*   __restrict in_data,
	PF_OutData*  __restrict out_data,
	PF_ParamDef* __restrict params[],
	PF_LayerDef* __restrict output
) noexcept
{
	PF_Err err = PF_Err_NONE;
	PF_Err errFormat = PF_Err_INVALID_INDEX;

	/* This plugin called frop PR - check video fomat */
	AEFX_SuiteScoper<PF_PixelFormatSuite1> pixelFormatSuite =
		AEFX_SuiteScoper<PF_PixelFormatSuite1>(
			in_data,
			kPFPixelFormatSuite,
			kPFPixelFormatSuiteVersion1,
			out_data);

	PrPixelFormat destinationPixelFormat = PrPixelFormat_Invalid;
	if (PF_Err_NONE == (errFormat = pixelFormatSuite->GetPixelFormat(output, &destinationPixelFormat)))
	{
		switch (destinationPixelFormat)
		{
			case PrPixelFormat_BGRA_4444_8u:
				err = PR_ImageStyle_CartoonEffect_BGRA_8u(in_data, out_data, params, output);
			break;

			case PrPixelFormat_VUYA_4444_8u_709:
			case PrPixelFormat_VUYA_4444_8u:
				err = PR_ImageStyle_CartoonEffect_VUYA_8u(in_data, out_data, params, output);
			break;

			case PrPixelFormat_VUYA_4444_32f_709:
			case PrPixelFormat_VUYA_4444_32f:
				err = PR_ImageStyle_CartoonEffect_VUYA_32f(in_data, out_data, params, output);
			break;

			case PrPixelFormat_BGRA_4444_16u:
				err = PR_ImageStyle_CartoonEffect_BGRA_16u(in_data, out_data, params, output);
			break;

			case PrPixelFormat_BGRA_4444_32f:
				err = PR_ImageStyle_CartoonEffect_BGRA_32f(in_data, out_data, params, output);
			break;

			default:
				err = PF_Err_INVALID_INDEX;
			break;
		}
	}
	else
	{
		err = PF_Err_UNRECOGNIZED_PARAM_TYPE;
	}

	return err;
}