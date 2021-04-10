#include "NoiseClean.hpp"
#include "PrSDKAESupport.h"


PF_Err ProcessImgInPR
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
					err = NoiseCleanPr_BGRA_4444_8u(in_data, out_data, params, output);
			break;

			case PrPixelFormat_BGRA_4444_16u:
			//		bValue = ProcessPrImage_BGRA_4444_16u(in_data, out_data, params, output, choosedKernel);
			break;

			case PrPixelFormat_VUYA_4444_8u:
			case PrPixelFormat_VUYA_4444_8u_709:
			{
				auto const& isBT709 = (PrPixelFormat_VUYA_4444_8u_709 == destinationPixelFormat);
			}
			break;

			case PrPixelFormat_BGRA_4444_32f:
			//		bValue = ProcessPrImage_BGRA_4444_32f(in_data, out_data, params, output, choosedKernel);
			break;

			case PrPixelFormat_VUYA_4444_32f:
			case PrPixelFormat_VUYA_4444_32f_709:
			{
				auto const& isBT709 = (PrPixelFormat_VUYA_4444_8u_709 == destinationPixelFormat);
			}
			break;

			case PrPixelFormat_RGB_444_10u:
			//		bValue = ProcessPrImage_RGB_444_10u(in_data, out_data, params, output, choosedKernel);
			break;

			default:
			break;
		} /* switch (destinationPixelFormat) */

	} /* if (PF_Err_NONE == (errFormat = pixelFormatSuite->GetPixelFormat(output, &destinationPixelFormat))) */
	else
	{
		/* error in determine pixel format */
		err = PF_Err_UNRECOGNIZED_PARAM_TYPE;
	}

	return err;
}
