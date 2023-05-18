#include "ColorCorrectionCIELab.hpp"
#include "ColorCorrectionCIELabEnums.hpp"
#include "ColorTransform.hpp"
#include "PrSDKAESupport.h"


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
	constexpr float reciproc = 1.f / static_cast<float>(u8_value_white);

	const float* __restrict fReferences = cCOLOR_ILLUMINANT[CIE_1931][ILLUMINANT_D65];

	for (A_long j = 0; j < sizeY; j++)
	{
		const PF_Pixel_BGRA_8u* __restrict pSrcLine = localSrc + j * line_pitch;
  		      PF_Pixel_BGRA_8u* __restrict pDstLine = localDst + j * line_pitch;
        
		for (A_long i = 0; i < sizeX; i++)
		{
			/* convert RGB to CIELab */
			fRGB pixRGB;
			pixRGB.R = static_cast<float>(pSrcLine[i].R) * reciproc;
			pixRGB.G = static_cast<float>(pSrcLine[i].G) * reciproc;
			pixRGB.B = static_cast<float>(pSrcLine[i].B) * reciproc;

			fCIELabPix pixCIELab = RGB2CIELab (pixRGB, fReferences);

			/* add values from sliders */
			pixCIELab.L += L_level;
			pixCIELab.a += A_level;
			pixCIELab.b += B_level;
			
			pixCIELab.L = CLAMP_VALUE(pixCIELab.L, static_cast<float>(L_coarse_min_level),  static_cast<float>(L_coarse_max_level));
			pixCIELab.a = CLAMP_VALUE(pixCIELab.a, static_cast<float>(AB_coarse_min_level), static_cast<float>(AB_coarse_max_level));
			pixCIELab.b = CLAMP_VALUE(pixCIELab.b, static_cast<float>(AB_coarse_min_level), static_cast<float>(AB_coarse_max_level));

			/* back convert to RGB */
			fRGB pixRGBOut = CIELab2RGB (pixCIELab, fReferences);
			pDstLine[i].B = static_cast<A_u_char>(CLAMP_VALUE(pixRGBOut.B * 255.f, static_cast<float>(u8_value_black), static_cast<float>(u8_value_white)));
			pDstLine[i].G = static_cast<A_u_char>(CLAMP_VALUE(pixRGBOut.G * 255.f, static_cast<float>(u8_value_black), static_cast<float>(u8_value_white)));
			pDstLine[i].R = static_cast<A_u_char>(CLAMP_VALUE(pixRGBOut.R * 255.f, static_cast<float>(u8_value_black), static_cast<float>(u8_value_white)));
			pDstLine[i].A = pSrcLine[i].A;

		} /* for (A_long i = 0; i < sizeX; i++) */
	} /* for (A_long j = 0; j < sizeY; j++) */

	return PF_Err_NONE;
}


PF_Err CIELabCorrect_BGRA_4444_16u
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
	const PF_Pixel_BGRA_16u* __restrict localSrc = reinterpret_cast<const PF_Pixel_BGRA_16u* __restrict>(pfLayer->data);
	      PF_Pixel_BGRA_16u* __restrict localDst = reinterpret_cast<      PF_Pixel_BGRA_16u* __restrict>(output->data);
	const A_long sizeY = pfLayer->extent_hint.bottom - pfLayer->extent_hint.top;
	const A_long sizeX = pfLayer->extent_hint.right  - pfLayer->extent_hint.left;
	const A_long line_pitch = pfLayer->rowbytes / static_cast<A_long>(PF_Pixel_BGRA_16u_size);
	constexpr float reciproc = 1.f / static_cast<float>(u16_value_white);

	const float* __restrict fReferences = cCOLOR_ILLUMINANT[CIE_1931][ILLUMINANT_D65];

	for (A_long j = 0; j < sizeY; j++)
	{
		const PF_Pixel_BGRA_16u* __restrict pSrcLine = localSrc + j * line_pitch;
		      PF_Pixel_BGRA_16u* __restrict pDstLine = localDst + j * line_pitch;

		for (A_long i = 0; i < sizeX; i++)
		{
			/* convert RGB to CIELab */
			fRGB pixRGB;
			pixRGB.R = static_cast<float>(pSrcLine[i].R) * reciproc;
			pixRGB.G = static_cast<float>(pSrcLine[i].G) * reciproc;
			pixRGB.B = static_cast<float>(pSrcLine[i].B) * reciproc;

			fCIELabPix pixCIELab = RGB2CIELab (pixRGB, fReferences);

			/* add values from sliders */
			pixCIELab.L += L_level;
			pixCIELab.a += A_level;
			pixCIELab.b += B_level;

			pixCIELab.L = CLAMP_VALUE(pixCIELab.L, static_cast<float>(L_coarse_min_level),  static_cast<float>(L_coarse_max_level));
			pixCIELab.a = CLAMP_VALUE(pixCIELab.a, static_cast<float>(AB_coarse_min_level), static_cast<float>(AB_coarse_max_level));
			pixCIELab.b = CLAMP_VALUE(pixCIELab.b, static_cast<float>(AB_coarse_min_level), static_cast<float>(AB_coarse_max_level));

			/* back convert to RGB */
			fRGB pixRGBOut = CIELab2RGB (pixCIELab, fReferences);
			pDstLine[i].B = static_cast<A_u_short>(CLAMP_VALUE(pixRGBOut.B * static_cast<float>(u16_value_white), static_cast<float>(u16_value_black), static_cast<float>(u16_value_white)));
			pDstLine[i].G = static_cast<A_u_short>(CLAMP_VALUE(pixRGBOut.G * static_cast<float>(u16_value_white), static_cast<float>(u16_value_black), static_cast<float>(u16_value_white)));
			pDstLine[i].R = static_cast<A_u_short>(CLAMP_VALUE(pixRGBOut.R * static_cast<float>(u16_value_white), static_cast<float>(u16_value_black), static_cast<float>(u16_value_white)));
			pDstLine[i].A = pSrcLine[i].A;

		} /* for (A_long i = 0; i < sizeX; i++) */
	} /* for (A_long j = 0; j < sizeY; j++) */

	return PF_Err_NONE;
}


PF_Err CIELabCorrect_BGRA_4444_32f
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
	const PF_Pixel_BGRA_32f* __restrict localSrc = reinterpret_cast<const PF_Pixel_BGRA_32f* __restrict>(pfLayer->data);
	      PF_Pixel_BGRA_32f* __restrict localDst = reinterpret_cast<      PF_Pixel_BGRA_32f* __restrict>(output->data);
	const A_long sizeY = pfLayer->extent_hint.bottom - pfLayer->extent_hint.top;
	const A_long sizeX = pfLayer->extent_hint.right  - pfLayer->extent_hint.left;
	const A_long line_pitch = pfLayer->rowbytes / static_cast<A_long>(PF_Pixel_BGRA_32f_size);

	const float* __restrict fReferences = cCOLOR_ILLUMINANT[CIE_1931][ILLUMINANT_D65];

	for (A_long j = 0; j < sizeY; j++)
	{
		const PF_Pixel_BGRA_32f* __restrict pSrcLine = localSrc + j * line_pitch;
		      PF_Pixel_BGRA_32f* __restrict pDstLine = localDst + j * line_pitch;

		for (A_long i = 0; i < sizeX; i++)
		{
			/* convert RGB to CIELab */
			fRGB pixRGB;
			pixRGB.R = pSrcLine[i].R;
			pixRGB.G = pSrcLine[i].G;
			pixRGB.B = pSrcLine[i].B;

			fCIELabPix pixCIELab = RGB2CIELab(pixRGB, fReferences);

			/* add values from sliders */
			pixCIELab.L += L_level;
			pixCIELab.a += A_level;
			pixCIELab.b += B_level;

			pixCIELab.L = CLAMP_VALUE(pixCIELab.L, static_cast<float>(L_coarse_min_level),  static_cast<float>(L_coarse_max_level));
			pixCIELab.a = CLAMP_VALUE(pixCIELab.a, static_cast<float>(AB_coarse_min_level), static_cast<float>(AB_coarse_max_level));
			pixCIELab.b = CLAMP_VALUE(pixCIELab.b, static_cast<float>(AB_coarse_min_level), static_cast<float>(AB_coarse_max_level));

			/* back convert to RGB */
			fRGB pixRGBOut = CIELab2RGB(pixCIELab, fReferences);
			pDstLine[i].B = CLAMP_VALUE(pixRGBOut.B, f32_value_black, f32_value_white);
			pDstLine[i].G = CLAMP_VALUE(pixRGBOut.G, f32_value_black, f32_value_white);
			pDstLine[i].R = CLAMP_VALUE(pixRGBOut.R, f32_value_black, f32_value_white);
			pDstLine[i].A = pSrcLine[i].A;

		} /* for (A_long i = 0; i < sizeX; i++) */
	} /* for (A_long j = 0; j < sizeY; j++) */

	return PF_Err_NONE;
}


PF_Err CIELabCorrect_VUYA_4444_8u
(
	PF_InData*    in_data,
	PF_OutData*   out_data,
	PF_ParamDef*  params[],
	PF_LayerDef*  output,
	const float   L_level,
	const float   A_level,
	const float   B_level,
	const bool    isBT709 = true
) noexcept
{
	const PF_LayerDef* pfLayer = reinterpret_cast<const PF_LayerDef*>(&params[eCIELAB_INPUT]->u.ld);
	const PF_Pixel_VUYA_8u* __restrict localSrc = reinterpret_cast<const PF_Pixel_VUYA_8u* __restrict>(pfLayer->data);
	      PF_Pixel_VUYA_8u* __restrict localDst = reinterpret_cast<      PF_Pixel_VUYA_8u* __restrict>(output->data);
	const A_long sizeY = pfLayer->extent_hint.bottom - pfLayer->extent_hint.top;
	const A_long sizeX = pfLayer->extent_hint.right  - pfLayer->extent_hint.left;
	const A_long line_pitch = pfLayer->rowbytes / static_cast<A_long>(PF_Pixel_VUYA_8u_size);
	constexpr float reciproc = 1.f / static_cast<float>(u8_value_white);

	const float* __restrict fReferences = cCOLOR_ILLUMINANT[CIE_1931][ILLUMINANT_D65];
	const float* __restrict pYUV2RGB = YUV2RGB[true == isBT709 ? BT709 : BT601];
	const float* __restrict pRGB2YUV = RGB2YUV[true == isBT709 ? BT709 : BT601];

	CACHE_ALIGN const float fYUV2RGB[9] {
		/* because we need RGB values in range 0...1 let's scale color transform matrix for decrease number of multiplications in main loop */
		pYUV2RGB[0] * reciproc, pYUV2RGB[1] * reciproc, pYUV2RGB[2] * reciproc,
		pYUV2RGB[3] * reciproc, pYUV2RGB[4] * reciproc, pYUV2RGB[5] * reciproc,
		pYUV2RGB[6] * reciproc, pYUV2RGB[7] * reciproc, pYUV2RGB[8] * reciproc,
	};

	for (A_long j = 0; j < sizeY; j++)
	{
		const PF_Pixel_VUYA_8u* __restrict pSrcLine = localSrc + j * line_pitch;
		      PF_Pixel_VUYA_8u* __restrict pDstLine = localDst + j * line_pitch;

		for (A_long i = 0; i < sizeX; i++)
		{
			/* convert RGB to CIELab */
			const float Y = static_cast<float>(pSrcLine[i].Y);
			const float U = static_cast<float>(pSrcLine[i].U) - 128.f;
			const float V = static_cast<float>(pSrcLine[i].V) - 128.f;

			fRGB pixRGB;
			pixRGB.R = Y * fYUV2RGB[0] + U * fYUV2RGB[1] + V * fYUV2RGB[2];
			pixRGB.G = Y * fYUV2RGB[3] + U * fYUV2RGB[4] + V * fYUV2RGB[5];
			pixRGB.B = Y * fYUV2RGB[6] + U * fYUV2RGB[7] + V * fYUV2RGB[8];

			fCIELabPix pixCIELab = RGB2CIELab(pixRGB, fReferences);

			/* add values from sliders */
			pixCIELab.L += L_level;
			pixCIELab.a += A_level;
			pixCIELab.b += B_level;

			pixCIELab.L = CLAMP_VALUE(pixCIELab.L, static_cast<float>(L_coarse_min_level),  static_cast<float>(L_coarse_max_level));
			pixCIELab.a = CLAMP_VALUE(pixCIELab.a, static_cast<float>(AB_coarse_min_level), static_cast<float>(AB_coarse_max_level));
			pixCIELab.b = CLAMP_VALUE(pixCIELab.b, static_cast<float>(AB_coarse_min_level), static_cast<float>(AB_coarse_max_level));

			/* back convert to RGB */
			fRGB pixRGBOut = CIELab2RGB(pixCIELab, fReferences);
			constexpr float scaler1 = static_cast<float>(u8_value_black);
			constexpr float scaler2 = static_cast<float>(u8_value_white);
			const float YY = (pixRGBOut.R * pRGB2YUV[0] + pixRGBOut.G * pRGB2YUV[1] + pixRGBOut.B * pRGB2YUV[2]) * scaler2;
			const float UU = (pixRGBOut.R * pRGB2YUV[3] + pixRGBOut.G * pRGB2YUV[4] + pixRGBOut.B * pRGB2YUV[5]) * scaler2 + 128.f;
			const float VV = (pixRGBOut.R * pRGB2YUV[6] + pixRGBOut.G * pRGB2YUV[7] + pixRGBOut.B * pRGB2YUV[8]) * scaler2 + 128.f;

			pDstLine[i].Y = static_cast<A_u_char>(CLAMP_VALUE(YY, scaler1, scaler2));
			pDstLine[i].U = static_cast<A_u_char>(CLAMP_VALUE(UU, scaler1, scaler2));
			pDstLine[i].V = static_cast<A_u_char>(CLAMP_VALUE(VV, scaler1, scaler2));
			pDstLine[i].A = pSrcLine[i].A;

		} /* for (A_long i = 0; i < sizeX; i++) */
	} /* for (A_long j = 0; j < sizeY; j++) */

	return PF_Err_NONE;
}


PF_Err CIELabCorrect_VUYA_4444_32f
(
	PF_InData*    in_data,
	PF_OutData*   out_data,
	PF_ParamDef*  params[],
	PF_LayerDef*  output,
	const float   L_level,
	const float   A_level,
	const float   B_level,
	const bool    isBT709 = true
) noexcept
{
	const PF_LayerDef* pfLayer = reinterpret_cast<const PF_LayerDef*>(&params[eCIELAB_INPUT]->u.ld);
	const PF_Pixel_VUYA_32f* __restrict localSrc = reinterpret_cast<const PF_Pixel_VUYA_32f* __restrict>(pfLayer->data);
	      PF_Pixel_VUYA_32f* __restrict localDst = reinterpret_cast<      PF_Pixel_VUYA_32f* __restrict>(output->data);
	const A_long sizeY = pfLayer->extent_hint.bottom - pfLayer->extent_hint.top;
	const A_long sizeX = pfLayer->extent_hint.right  - pfLayer->extent_hint.left;
	const A_long line_pitch = pfLayer->rowbytes / static_cast<A_long>(PF_Pixel_VUYA_32f_size);

	const float* __restrict fReferences = cCOLOR_ILLUMINANT[CIE_1931][ILLUMINANT_D65];
	const float* __restrict pYUV2RGB = YUV2RGB[true == isBT709 ? BT709 : BT601];
	const float* __restrict pRGB2YUV = RGB2YUV[true == isBT709 ? BT709 : BT601];

	for (A_long j = 0; j < sizeY; j++)
	{
		const PF_Pixel_VUYA_32f* __restrict pSrcLine = localSrc + j * line_pitch;
		      PF_Pixel_VUYA_32f* __restrict pDstLine = localDst + j * line_pitch;

		for (A_long i = 0; i < sizeX; i++)
		{
			/* convert RGB to CIELab */
			fRGB pixRGB;
			pixRGB.R = pSrcLine[i].Y * pYUV2RGB[0] + pSrcLine[i].U * pYUV2RGB[1] + pSrcLine[i].V * pYUV2RGB[2];
			pixRGB.G = pSrcLine[i].Y * pYUV2RGB[3] + pSrcLine[i].U * pYUV2RGB[4] + pSrcLine[i].V * pYUV2RGB[5];
			pixRGB.B = pSrcLine[i].Y * pYUV2RGB[6] + pSrcLine[i].U * pYUV2RGB[7] + pSrcLine[i].V * pYUV2RGB[8];

			fCIELabPix pixCIELab = RGB2CIELab (pixRGB, fReferences);

			/* add values from sliders */
			pixCIELab.L += L_level;
			pixCIELab.a += A_level;
			pixCIELab.b += B_level;

			pixCIELab.L = CLAMP_VALUE(pixCIELab.L, static_cast<float>(L_coarse_min_level),  static_cast<float>(L_coarse_max_level));
			pixCIELab.a = CLAMP_VALUE(pixCIELab.a, static_cast<float>(AB_coarse_min_level), static_cast<float>(AB_coarse_max_level));
			pixCIELab.b = CLAMP_VALUE(pixCIELab.b, static_cast<float>(AB_coarse_min_level), static_cast<float>(AB_coarse_max_level));

			/* back convert to RGB */
			fRGB pixRGBOut = CIELab2RGB (pixCIELab, fReferences);
			const float YY = pixRGBOut.R * pRGB2YUV[0] + pixRGBOut.G * pRGB2YUV[1] + pixRGBOut.B * pRGB2YUV[2];
			const float UU = pixRGBOut.R * pRGB2YUV[3] + pixRGBOut.G * pRGB2YUV[4] + pixRGBOut.B * pRGB2YUV[5];
			const float VV = pixRGBOut.R * pRGB2YUV[6] + pixRGBOut.G * pRGB2YUV[7] + pixRGBOut.B * pRGB2YUV[8];

			pDstLine[i].Y = CLAMP_VALUE(YY, f32_value_black, f32_value_white);
			pDstLine[i].U = CLAMP_VALUE(UU, f32_value_black, f32_value_white);
			pDstLine[i].V = CLAMP_VALUE(VV, f32_value_black, f32_value_white);
			pDstLine[i].A = pSrcLine[i].A;

		} /* for (A_long i = 0; i < sizeX; i++) */
	} /* for (A_long j = 0; j < sizeY; j++) */

	return PF_Err_NONE;
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
				{
					auto const isBT709 = (destinationPixelFormat == PrPixelFormat_VUYA_4444_8u_709);
					err = CIELabCorrect_VUYA_4444_8u (in_data, out_data, params, output, L_level, A_level, B_level, isBT709);
				}
				break;

				case PrPixelFormat_VUYA_4444_32f_709:
				case PrPixelFormat_VUYA_4444_32f:
				{
					auto const isBT709 = (destinationPixelFormat == PrPixelFormat_VUYA_4444_8u_709);
					err = CIELabCorrect_VUYA_4444_32f (in_data, out_data, params, output, L_level, A_level, B_level, isBT709);
				}
				break;

				case PrPixelFormat_BGRA_4444_16u:
					err = CIELabCorrect_BGRA_4444_16u (in_data, out_data, params, output, L_level, A_level, B_level);
				break;

				case PrPixelFormat_BGRA_4444_32f:
					err = CIELabCorrect_BGRA_4444_32f (in_data, out_data, params, output, L_level, A_level, B_level);
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
