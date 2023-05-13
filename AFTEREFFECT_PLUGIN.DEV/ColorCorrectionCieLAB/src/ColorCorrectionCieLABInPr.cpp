#include "ColorCorrectionCIELab.hpp"
#include "ColorCorrectionCIELabEnums.hpp"
#include "PrSDKAESupport.h"
#include "FastAriphmetics.hpp"

template <typename T, std::enable_if_t<is_RGB_proc<T>::value>* = nullptr>
inline fCIELabPix RGB2CIELab (const T& pixelRGB) noexcept
{
	constexpr float powCoeff = 2.19921875f;
	constexpr float reciproc255 = 1.f / 255.f;

	const float tR = FastCompute::Pow(static_cast<float>(pixelRGB.R) * reciproc255, powCoeff);
	const float tG = FastCompute::Pow(static_cast<float>(pixelRGB.G) * reciproc255, powCoeff);
	const float tB = FastCompute::Pow(static_cast<float>(pixelRGB.B) * reciproc255, powCoeff);

	const float x = tR * 0.60672089f + tG * 0.19521921f + tB * 0.19799678f;
	const float y = tR * 0.29738000f + tG * 0.62735000f + tB * 0.07552700f;
	const float z = tR * 0.02482481f + tG * 0.06492290f + tB * 0.91024310f;

	const float x1 = (x > 0.0088560f) ? FastCompute::Cbrt(x) : 7.7870f * x + 0.1379310f;
	const float y1 = (y > 0.0088560f) ? FastCompute::Cbrt(y) : 7.7870f * y + 0.1379310f;
	const float z1 = (z > 0.0088560f) ? FastCompute::Cbrt(z) : 7.7870f * z + 0.1379310f;

	fCIELabPix pixelLAB;
	pixelLAB.L = 116.0f * y1 - 16.0f;
	pixelLAB.a = 500.0f * (x1 - y1);
	pixelLAB.b = 200.0f * (y1 - z1);

	return pixelLAB;
}

inline fRGB CIELab2RGB (const fCIELabPix& pixelCIELab, const float& black, const float& white) noexcept
{
	constexpr float reciproc116 = 1.f / 116.f;
	constexpr float reciproc500 = 1.f / 500.f;
	constexpr float reciproc200 = 1.f / 200.f;
	constexpr float reciproc7 = 1.f / 7.7870f;

	const float y = (pixelCIELab.L + 16.0f) * reciproc116;
	const float x = pixelCIELab.a * reciproc500 + y;
	const float z = y - pixelCIELab.b * reciproc200;

	const float x1 = ((x > 0.2068930f) ? x * x * x : (x - 0.1379310f) * reciproc7) * 0.950470f;
	const float y1 =  (y > 0.2068930f) ? y * y * y : (y - 0.1379310f) * reciproc7;
	const float z1 = ((z > 0.2068930f) ? z * z * z : (z - 0.1379310f) * reciproc7) * 1.088830f;

	const float rr = x1 *  2.041370f + y1 * -0.564950f + z1 * -0.344690f;
	const float gg = x1 * -0.962700f + y1 *  1.876010f + z1 *  0.041560f;
	const float bb = x1 *  0.013450f + y1 * -0.118390f + z1 *  1.015410f;

	const float r1 = FastCompute::Exp(0.4547070f * FastCompute::Log(rr));
	const float g1 = FastCompute::Exp(0.4547070f * FastCompute::Log(gg));
	const float b1 = FastCompute::Exp(0.4547070f * FastCompute::Log(bb));

	const float iR = CLAMP_VALUE(r1 * 255.0f, black, white);
	const float iG = CLAMP_VALUE(g1 * 255.0f, black, white);
	const float iB = CLAMP_VALUE(b1 * 255.0f, black, white);

	fRGB pixelRGB;
	pixelRGB.R = iR;
	pixelRGB.G = iG;
	pixelRGB.B = iB;

	return pixelRGB;
}


PF_Err CIELabCorrect_BGRA_4444_8u
(
	PF_InData*    in_data,
	PF_OutData*   out_data,
	PF_ParamDef*  params[],
	PF_LayerDef*  output,
	const float   L_level,
	const float   A_level,
	const float   B_level
) noexcept
{
	const PF_LayerDef* pfLayer = reinterpret_cast<const PF_LayerDef*>(&params[eCIELAB_INPUT]->u.ld);
	const PF_Pixel_BGRA_8u* __restrict localSrc = reinterpret_cast<const PF_Pixel_BGRA_8u* __restrict>(pfLayer->data);
	      PF_Pixel_BGRA_8u* __restrict localDst = reinterpret_cast<      PF_Pixel_BGRA_8u* __restrict>(output->data);
	const A_long sizeY = pfLayer->extent_hint.bottom - pfLayer->extent_hint.top;
	const A_long sizeX = pfLayer->extent_hint.right  - pfLayer->extent_hint.left;
	const A_long line_pitch = pfLayer->rowbytes / static_cast<A_long>(PF_Pixel_BGRA_8u_size);
	PF_Err err = PF_Err_NONE;
	
	for (A_long j = 0; j < sizeY; j++)
	{
		const PF_Pixel_BGRA_8u* __restrict pSrcLine = localSrc + j * line_pitch;
  		      PF_Pixel_BGRA_8u* __restrict pDstLine = localDst + j * line_pitch;

		for (A_long i = 0; i < sizeX; i++)
		{
			/* convert RGB to CIELab */
			fCIELabPix pixCIELab = RGB2CIELab (pSrcLine[i]);

			/* add values from sliders */
			pixCIELab.L += L_level;
			pixCIELab.a += A_level;
			pixCIELab.b += B_level;

			constexpr float clampMin = static_cast<float>(u8_value_black);
			constexpr float clampMax = static_cast<float>(u8_value_white);

			/* back convert to RGB */
			fRGB dstPixel = CIELab2RGB (pixCIELab, clampMin, clampMax);
			pDstLine[i].B = static_cast<A_u_char>(dstPixel.B);
			pDstLine[i].G = static_cast<A_u_char>(dstPixel.G);
			pDstLine[i].R = static_cast<A_u_char>(dstPixel.R);
			pDstLine[i].A = pSrcLine[i].A;

		} /* for (A_long i = 0; i < sizeX; i++) */
	} /* for (A_long j = 0; j < sizeY; j++) */


	return err;
}


PF_Err ProcessImgInPR
(
	PF_InData*    in_data,
	PF_OutData*   out_data,
	PF_ParamDef*  params[],
	PF_LayerDef*  output
) noexcept
{
	PF_Err err = PF_Err_NONE;

	/* get sliders values */
	auto const& L_coarse = params[eCIELAB_SLIDER_L_COARSE]->u.sd.value;
	auto const& L_fine   = params[eCIELAB_SLIDER_L_FINE  ]->u.fs_d.value;
	auto const& A_coarse = params[eCIELAB_SLIDER_A_COARSE]->u.sd.value;
	auto const& A_fine   = params[eCIELAB_SLIDER_A_FINE  ]->u.fs_d.value;
	auto const& B_coarse = params[eCIELAB_SLIDER_B_COARSE]->u.sd.value;
	auto const& B_fine   = params[eCIELAB_SLIDER_B_FINE  ]->u.fs_d.value;

	const float L_level = static_cast<float>(static_cast<double>(L_coarse) + L_fine);
	const float A_level = static_cast<float>(static_cast<double>(A_coarse) + A_fine);
	const float B_level = static_cast<float>(static_cast<double>(B_coarse) + B_fine);

	if ((0.f == L_level) && (0.f == A_level) && (0.f == B_level))
	{
		err = PF_COPY(&params[eCIELAB_INPUT]->u.ld, output, NULL, NULL);
	}
	else
	{
		/* This plugin called frop PR - check video fomat */
		PrPixelFormat destinationPixelFormat = PrPixelFormat_Invalid;
		if (PF_Err_NONE == (AEFX_SuiteScoper<PF_PixelFormatSuite1>(in_data, kPFPixelFormatSuite, kPFPixelFormatSuiteVersion1, out_data)->GetPixelFormat(output, &destinationPixelFormat)))
		{
			switch (destinationPixelFormat)
			{
				case PrPixelFormat_BGRA_4444_8u:
					err = CIELabCorrect_BGRA_4444_8u (in_data, out_data, params, output, L_level, A_level, B_level);
				break;

				case PrPixelFormat_VUYA_4444_8u_709:
				case PrPixelFormat_VUYA_4444_8u:
				break;

				case PrPixelFormat_VUYA_4444_32f_709:
				case PrPixelFormat_VUYA_4444_32f:
				break;

				case PrPixelFormat_BGRA_4444_16u:
				break;

				case PrPixelFormat_BGRA_4444_32f:
				break;

				default:
					err = PF_Err_INVALID_INDEX;
				break;
			} /* switch (destinationPixelFormat) */
		} /* if (PF_Err_NONE == (AEFX_SuiteScoper<PF_PixelFormatSuite1>(in_data, kPFPixelFormatSuite, kPFPixelFormatSuiteVersion1, out_data)->GetPixelFormat(output, &destinationPixelFormat))) */
		else
		{
			err = PF_Err_UNRECOGNIZED_PARAM_TYPE;
		}
	}
	return err;
}
