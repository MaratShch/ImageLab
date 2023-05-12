#include "ColorCorrectionCIELab.hpp"
#include "PrSDKAESupport.h"


PF_Err ProcessImgInPR
(
	PF_InData*    in_data,
	PF_OutData*   out_data,
	PF_ParamDef*  params[],
	PF_LayerDef*  output
) noexcept
{
	PF_Err err = PF_Err_NONE;

	/* This plugin called frop PR - check video fomat */
	PrPixelFormat destinationPixelFormat = PrPixelFormat_Invalid;
	if (PF_Err_NONE == (AEFX_SuiteScoper<PF_PixelFormatSuite1>(in_data, kPFPixelFormatSuite, kPFPixelFormatSuiteVersion1, out_data)->GetPixelFormat(output, &destinationPixelFormat)))
	{
		switch (destinationPixelFormat)
		{
			case PrPixelFormat_BGRA_4444_8u:
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

	return err;
}
