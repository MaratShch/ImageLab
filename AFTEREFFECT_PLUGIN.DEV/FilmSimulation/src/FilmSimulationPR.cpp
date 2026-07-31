#include "FilmSimulation.hpp"
#include "AlgoControlEnums.hpp"
#include "AlgoMemHandler.hpp"
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

    // This plugin called from PR - check video fomat
    const PF_LayerDef* pfLayer = reinterpret_cast<const PF_LayerDef*>(&params[UnderlyingType(FilmSimulationCtrl::VIDEO_INPUT)]->u.ld);
    const A_long sizeY = pfLayer->extent_hint.bottom - pfLayer->extent_hint.top;
    const A_long sizeX = pfLayer->extent_hint.right  - pfLayer->extent_hint.left;
    const A_long rowBytes = pfLayer->rowbytes;

    MemHandler memHndl = alloc_memory_buffers (sizeX, sizeY);
    if (true == mem_handler_valid(memHndl))
    {
        PF_Err errFormat = PF_Err_INVALID_INDEX;
        PrPixelFormat destinationPixelFormat = PrPixelFormat_Invalid;

        // This plugin called frop PR - check video fomat
        auto const pixelFormatSuite{ AEFX_SuiteScoper<PF_PixelFormatSuite1>(in_data, kPFPixelFormatSuite, kPFPixelFormatSuiteVersion1, out_data) };

        if (PF_Err_NONE == (errFormat = pixelFormatSuite->GetPixelFormat(output, &destinationPixelFormat)))
        {
            const double fpsRate = image_lab_get_fps(in_data);

            switch (destinationPixelFormat)
            {
                case PrPixelFormat_BGRA_4444_8u:
                case PrPixelFormat_BGRA_4444_16u:
                case PrPixelFormat_BGRA_4444_32f:
                case PrPixelFormat_BGRA_4444_32f_Linear:
                case PrPixelFormat_BGRP_4444_8u:
                case PrPixelFormat_BGRP_4444_16u:
                case PrPixelFormat_BGRP_4444_32f:
                case PrPixelFormat_BGRP_4444_32f_Linear:
                case PrPixelFormat_BGRX_4444_8u:
                case PrPixelFormat_BGRX_4444_16u:
                case PrPixelFormat_BGRX_4444_32f:
                case PrPixelFormat_BGRX_4444_32f_Linear:
                case PrPixelFormat_VUYA_4444_8u_709:
                case PrPixelFormat_VUYA_4444_8u:
                case PrPixelFormat_VUYA_4444_32f_709:
                case PrPixelFormat_VUYA_4444_32f:
                case PrPixelFormat_VUYP_4444_8u_709:
                case PrPixelFormat_VUYP_4444_8u:
                case PrPixelFormat_VUYP_4444_32f_709:
                case PrPixelFormat_VUYP_4444_32f:
                case PrPixelFormat_VUYX_4444_8u_709:
                case PrPixelFormat_VUYX_4444_8u:
                case PrPixelFormat_VUYX_4444_32f_709:
                case PrPixelFormat_VUYX_4444_32f:
                case PrPixelFormat_ARGB_4444_8u:
                case PrPixelFormat_ARGB_4444_16u:
                case PrPixelFormat_ARGB_4444_32f:
                case PrPixelFormat_PRGB_4444_32f:
                case PrPixelFormat_XRGB_4444_32f:
                case PrPixelFormat_ARGB_4444_32f_Linear:
                case PrPixelFormat_PRGB_4444_32f_Linear:
                case PrPixelFormat_XRGB_4444_32f_Linear:
                case PrPixelFormat_RGB_444_10u:
                case PrPixelFormat_RGB_444_12u_PQ_709:
                case PrPixelFormat_RGB_444_12u_PQ_P3:
                case PrPixelFormat_RGB_444_12u_PQ_2020:
                    err = PF_Err_NONE;
                    break;

                default:
                    err = PF_Err_INTERNAL_STRUCT_DAMAGED;
                    break;
            } /* switch (destinationPixelFormat) */

        } /* if (PF_Err_NONE == (errFormat = pixelFormatSuite->GetPixelFormat(output, &destinationPixelFormat))) */
        else
        {
            /* error in determine pixel format */
            err = PF_Err_UNRECOGNIZED_PARAM_TYPE;
        }

        free_memory_buffers (memHndl);

    } // if (true == mem_handler_valid(memHndl))
    else
        err = PF_Err_OUT_OF_MEMORY;

	return err;
}
