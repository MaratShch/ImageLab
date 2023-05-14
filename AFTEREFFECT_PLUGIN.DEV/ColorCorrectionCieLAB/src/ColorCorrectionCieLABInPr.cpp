#include "ColorCorrectionCIELab.hpp"
#include "ColorCorrectionCIELabEnums.hpp"
#include "ColorTransformMatrix.hpp"
#include "PrSDKAESupport.h"
#include "FastAriphmetics.hpp"

/* 
 MATLAB reference:
 % lab = rgb2lab([10/255 124/255 240/255]) <- RGB
 % 52.6946   15.7288  -65.8706             <- LAB

 http://www.easyrgb.com/en/math.php

*/

inline fCIELabPix RGB2CIELab (const fRGB& pixelRGB, const float* __restrict fIlluminant) noexcept
{
	/* in first convert: sRGB -> XYZ */
	constexpr float reciproc12 = 1.f  / 12.92f;
	constexpr float reciproc16 = 16.f / 116.f;
	constexpr float reciproc1  = 1.f  / 1.055f;

	const float varR = ((pixelRGB.R > 0.04045f) ? FastCompute::Pow((pixelRGB.R + 0.055f) * reciproc1, 2.40f) : pixelRGB.R * reciproc12);
	const float varG = ((pixelRGB.G > 0.04045f) ? FastCompute::Pow((pixelRGB.G + 0.055f) * reciproc1, 2.40f) : pixelRGB.G * reciproc12);
	const float varB = ((pixelRGB.B > 0.04045f) ? FastCompute::Pow((pixelRGB.B + 0.055f) * reciproc1, 2.40f) : pixelRGB.B * reciproc12);

	const float X = varR * 41.24f + varG * 35.76f + varB * 18.05f;
	const float Y = varR * 21.26f + varG * 71.52f + varB * 7.220f;
	const float Z = varR * 1.930f + varG * 11.92f + varB * 95.05f;

	/* convert: XYZ - > Cie-L*ab */
	const float varX = X / fIlluminant[0];
	const float varY = Y / fIlluminant[1];
	const float varZ = Z / fIlluminant[2];

	const float vX = (varX > 0.0088560f) ? FastCompute::Cbrt(varX) : 7.7870f * varX + reciproc16;
	const float vY = (varY > 0.0088560f) ? FastCompute::Cbrt(varY) : 7.7870f * varY + reciproc16;
	const float vZ = (varZ > 0.0088560f) ? FastCompute::Cbrt(varZ) : 7.7870f * varZ + reciproc16;

	fCIELabPix pixelLAB;
	pixelLAB.L = 116.f * vX - 16.f;
	pixelLAB.a = 500.f * (vX - vY);
	pixelLAB.b = 200.f * (vY - vZ);

	return pixelLAB;
}

inline fRGB CIELab2RGB (const fCIELabPix& pixelCIELab) noexcept
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

	fRGB pixelRGB;
	pixelRGB.R = r1;
	pixelRGB.G = g1;
	pixelRGB.B = b1;

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
	constexpr float reciproc255 = 1.f / 255.f;

	const float* __restrict cIlluminant = cCOLOR_ILLUMINANT[CIE_1931][ILLUMINANT_D65];

	for (A_long j = 0; j < sizeY; j++)
	{
		const PF_Pixel_BGRA_8u* __restrict pSrcLine = localSrc + j * line_pitch;
  		      PF_Pixel_BGRA_8u* __restrict pDstLine = localDst + j * line_pitch;

        __VECTORIZATION__
		for (A_long i = 0; i < sizeX; i++)
		{
			/* convert RGB to CIELab */
			fRGB pixRGB;
			pixRGB.R = static_cast<float>(pSrcLine[i].R) * reciproc255;
			pixRGB.G = static_cast<float>(pSrcLine[i].G) * reciproc255;
			pixRGB.B = static_cast<float>(pSrcLine[i].B) * reciproc255;

			fCIELabPix pixCIELab = RGB2CIELab (pixRGB, cIlluminant);

			/* add values from sliders */
			pixCIELab.L += L_level;
			pixCIELab.a += A_level;
			pixCIELab.b += B_level;
			
			pixCIELab.L = CLAMP_VALUE(pixCIELab.L, static_cast<float>(L_coarse_min_level),  static_cast<float>(L_coarse_max_level));
			pixCIELab.a = CLAMP_VALUE(pixCIELab.a, static_cast<float>(AB_coarse_min_level), static_cast<float>(AB_coarse_max_level));
			pixCIELab.b = CLAMP_VALUE(pixCIELab.b, static_cast<float>(AB_coarse_min_level), static_cast<float>(AB_coarse_max_level));

			/* back convert to RGB */
			fRGB pixRGBOut = CIELab2RGB (pixCIELab);
			pDstLine[i].B = static_cast<A_u_char>(pixRGBOut.B * 255.f);
			pDstLine[i].G = static_cast<A_u_char>(pixRGBOut.G * 255.f);
			pDstLine[i].R = static_cast<A_u_char>(pixRGBOut.R * 255.f);
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
