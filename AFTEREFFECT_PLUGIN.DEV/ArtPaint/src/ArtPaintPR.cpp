#include "ArtPaint.hpp"
#include "PrSDKAESupport.h"


PF_Err ProcessImgInPR
(
	PF_InData*   RESTRICT in_data,
	PF_OutData*  RESTRICT out_data,
	PF_ParamDef* RESTRICT params[],
	PF_LayerDef* RESTRICT output
) 
{
	PF_Err err{ PF_Err_NONE };
	PF_Err errFormat{ PF_Err_INVALID_INDEX };
	PrPixelFormat destinationPixelFormat{ PrPixelFormat_Invalid };

	/* This plugin called frop PR - check video fomat */
	auto const pixelFormatSuite{ AEFX_SuiteScoper<PF_PixelFormatSuite1>(in_data, kPFPixelFormatSuite, kPFPixelFormatSuiteVersion1, out_data) };

	if (PF_Err_NONE == (errFormat = pixelFormatSuite->GetPixelFormat(output, &destinationPixelFormat)))
	{
		switch (destinationPixelFormat)
		{
            case PrPixelFormat_BGRA_4444_8u:
            case PrPixelFormat_BGRP_4444_8u:
            case PrPixelFormat_BGRX_4444_8u:
            break;

            case PrPixelFormat_BGRA_4444_16u:
            case PrPixelFormat_BGRP_4444_16u:
            case PrPixelFormat_BGRX_4444_16u:
            break;

            case PrPixelFormat_BGRA_4444_32f:
            case PrPixelFormat_BGRX_4444_32f:
            case PrPixelFormat_BGRP_4444_32f:
            break;

            case PrPixelFormat_BGRA_4444_32f_Linear:
            case PrPixelFormat_BGRX_4444_32f_Linear:
            case PrPixelFormat_BGRP_4444_32f_Linear:
            break;

            case PrPixelFormat_VUYA_4444_8u_709:
            case PrPixelFormat_VUYA_4444_8u:
            case PrPixelFormat_VUYX_4444_8u_709:
            case PrPixelFormat_VUYX_4444_8u:
            break;

            case PrPixelFormat_VUYA_4444_32f_709:
            case PrPixelFormat_VUYA_4444_32f:
            case PrPixelFormat_VUYX_4444_32f_709:
            case PrPixelFormat_VUYX_4444_32f:
            break;

            case PrPixelFormat_RGB_444_10u:
            break;

            case PrPixelFormat_ARGB_4444_8u:
            case PrPixelFormat_XRGB_4444_8u:
            case PrPixelFormat_PRGB_4444_8u:
            break;

            case PrPixelFormat_ARGB_4444_16u:
            case PrPixelFormat_XRGB_4444_16u:
            case PrPixelFormat_PRGB_4444_16u:
            break;

            case PrPixelFormat_ARGB_4444_32f:
            case PrPixelFormat_XRGB_4444_32f:
            case PrPixelFormat_PRGB_4444_32f:
            break;
    
            case PrPixelFormat_ARGB_4444_32f_Linear:
            case PrPixelFormat_PRGB_4444_32f_Linear:
            case PrPixelFormat_XRGB_4444_32f_Linear:
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
