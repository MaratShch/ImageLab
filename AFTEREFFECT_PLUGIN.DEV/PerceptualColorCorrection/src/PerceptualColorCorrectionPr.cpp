#include "PerceptualColorCorrection.hpp"
#include "PerceptualColorCorrectionAlgo.hpp"
#include "PrSDKAESupport.h"


PF_Err RenderInPremier
(
	PF_InData*   __restrict in_data,
	PF_OutData*  __restrict out_data,
	PF_ParamDef* __restrict params[],
	PF_LayerDef* __restrict output
) 
{
	PF_Err err = PF_Err_NONE;
	PF_Err errFormat = PF_Err_INVALID_INDEX;
	PrPixelFormat destinationPixelFormat = PrPixelFormat_Invalid;

	/* This plugin called frop PR - check video fomat */
	auto const pixelFormatSuite{ AEFX_SuiteScoper<PF_PixelFormatSuite1>(in_data, kPFPixelFormatSuite, kPFPixelFormatSuiteVersion1, out_data) };

	if (PF_Err_NONE == (errFormat = pixelFormatSuite->GetPixelFormat(output, &destinationPixelFormat)))
	{
		const PF_LayerDef* __restrict pfLayer = reinterpret_cast<const PF_LayerDef* __restrict>(&params[0]->u.ld);
		const A_long sizeY = pfLayer->extent_hint.bottom - pfLayer->extent_hint.top;
		const A_long sizeX = pfLayer->extent_hint.right  - pfLayer->extent_hint.left;

		//// this allocation will be replaced by memory storage usage
		fRGB* pTmpImg = new fRGB[sizeY * sizeX];
//		memset(pTmpImg, 0, sizeof(fRGB) * sizeY * sizeX);
		////

		switch (destinationPixelFormat)
		{
			case PrPixelFormat_BGRA_4444_8u:
			{
				const PF_Pixel_BGRA_8u* __restrict pSrc = reinterpret_cast<const PF_Pixel_BGRA_8u* __restrict>(pfLayer->data);
				      PF_Pixel_BGRA_8u* __restrict pDst = reinterpret_cast<      PF_Pixel_BGRA_8u* __restrict>(output->data);
				const A_long linePitch = pfLayer->rowbytes / static_cast<A_long>(PF_Pixel_BGRA_8u_size);

				QuickWhiteBalance (pSrc, pTmpImg, sizeX, sizeY, linePitch);
				dbgBufferShow (pSrc, pTmpImg, pDst, sizeX, sizeY, linePitch, linePitch);
			}
			break;

			case PrPixelFormat_BGRA_4444_16u:
			{
				const PF_Pixel_BGRA_16u* __restrict pSrc = reinterpret_cast<const PF_Pixel_BGRA_16u* __restrict>(pfLayer->data);
				      PF_Pixel_BGRA_16u* __restrict pDst = reinterpret_cast<      PF_Pixel_BGRA_16u* __restrict>(output->data);
				const A_long linePitch = pfLayer->rowbytes / static_cast<A_long>(PF_Pixel_BGRA_16u_size);

				QuickWhiteBalance (pSrc, pTmpImg, sizeX, sizeY, linePitch);
				dbgBufferShow (pSrc, pTmpImg, pDst, sizeX, sizeY, linePitch, linePitch);
			}
			break;

			case PrPixelFormat_BGRA_4444_32f:
			{
				const PF_Pixel_BGRA_32f* __restrict pSrc = reinterpret_cast<const PF_Pixel_BGRA_32f* __restrict>(pfLayer->data);
				      PF_Pixel_BGRA_32f* __restrict pDst = reinterpret_cast<      PF_Pixel_BGRA_32f* __restrict>(output->data);
				const A_long linePitch = pfLayer->rowbytes / static_cast<A_long>(PF_Pixel_BGRA_32f_size);

				QuickWhiteBalance (pSrc, pTmpImg, sizeX, sizeY, linePitch);
				dbgBufferShow (pSrc, pTmpImg, pDst, sizeX, sizeY, linePitch, linePitch);
			}
			break;

			case PrPixelFormat_VUYA_4444_8u_709:
			case PrPixelFormat_VUYA_4444_8u:
			break;
			
			case PrPixelFormat_VUYA_4444_32f_709:
			case PrPixelFormat_VUYA_4444_32f:
			break;

			case PrPixelFormat_RGB_444_10u:
			break;

			default:
			break;
		} /* switch (destinationPixelFormat) */

		/////
		delete[] pTmpImg;
		pTmpImg = nullptr;
		/////

	} /* if (PF_Err_NONE == (errFormat = pixelFormatSuite->GetPixelFormat(output, &destinationPixelFormat))) */
	else
	{
		/* error in determine pixel format */
		err = PF_Err_UNRECOGNIZED_PARAM_TYPE;
	}

	return err;
}
