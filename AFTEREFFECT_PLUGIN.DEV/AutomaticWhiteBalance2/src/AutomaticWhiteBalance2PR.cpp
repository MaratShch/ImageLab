#include "AutomaticWhiteBalance2.hpp"
#include "AutomaticWhiteBalance2Enum.hpp"



PF_Err ProcessImgInPR
(
    PF_InData*   in_data,
    PF_OutData*  out_data,
    PF_ParamDef* params[],
    PF_LayerDef* output
)
{
    PF_Err err = PF_Err_NONE;
    PF_Err errFormat = PF_Err_INVALID_INDEX;
    PrPixelFormat destinationPixelFormat = PrPixelFormat_Invalid;

    const PF_LayerDef* pfLayer = reinterpret_cast<const PF_LayerDef*>(&params[UnderlyingType(eAWB2::eIMAGE_AWB2_INPUT)]->u.ld);
    const A_long sizeY = pfLayer->extent_hint.bottom - pfLayer->extent_hint.top;
    const A_long sizeX = pfLayer->extent_hint.right - pfLayer->extent_hint.left;

//    const AlgoControls algoControls = GetControlParametersStruct(params);

//    MemHandler algoMemHandler = alloc_memory_buffers(sizeX, sizeY, algoControls.sliderIterCnt);
//    if (true == mem_handler_valid(algoMemHandler))
    {
        // This plugin called frop PR - check video format
        auto const pixelFormatSuite{ AEFX_SuiteScoper<PF_PixelFormatSuite1>(in_data, kPFPixelFormatSuite, kPFPixelFormatSuiteVersion1, out_data) };

        if (PF_Err_NONE == (errFormat = pixelFormatSuite->GetPixelFormat(output, &destinationPixelFormat)))
        {
            switch (destinationPixelFormat)
            {
                case PrPixelFormat_BGRA_4444_8u:
                {
                }
                break;

                case PrPixelFormat_BGRA_4444_16u:
                {
                }
                break;

                case PrPixelFormat_BGRX_4444_8u:
                {
                }
                break;

                case PrPixelFormat_BGRA_4444_32f:
                {
                }
                break;

                case PrPixelFormat_BGRA_4444_32f_Linear:
                {
                }
                break;

                case PrPixelFormat_BGRP_4444_8u:
                {
                }
                break;

                case PrPixelFormat_BGRP_4444_16u:
                {
                }
                break;

                case PrPixelFormat_BGRP_4444_32f:
                {
                }
                break;

                case PrPixelFormat_BGRP_4444_32f_Linear:
                {
                }
                break;

                case PrPixelFormat_BGRX_4444_16u:
                {
                }
                break;

                case PrPixelFormat_BGRX_4444_32f:
                {
                }
                break;

                case PrPixelFormat_BGRX_4444_32f_Linear:
                {
                }
                break;

                case PrPixelFormat_VUYA_4444_8u_709:
                {
                }
                break;

                case PrPixelFormat_VUYA_4444_8u:
                {
                }
                break;

                case PrPixelFormat_VUYA_4444_32f_709:
                {
                }
                break;

                case PrPixelFormat_VUYA_4444_32f:
                {
                }
                break;

                case PrPixelFormat_VUYP_4444_8u_709:
                {
                }
                break;

                case PrPixelFormat_VUYP_4444_8u:
                {
                }
                break;

                case PrPixelFormat_VUYP_4444_32f_709:
                {
                }
                break;

                case PrPixelFormat_VUYP_4444_32f:
                {
                }
                break;

                case PrPixelFormat_VUYX_4444_8u_709:
                {
                }
                break;

                case PrPixelFormat_VUYX_4444_8u:
                {
                }
                break;

                case PrPixelFormat_VUYX_4444_32f_709:
                {
                }
                break;

                case PrPixelFormat_VUYX_4444_32f:
                {
                }
                break;

                case PrPixelFormat_RGB_444_10u:
                {
                }
                break;

                case PrPixelFormat_ARGB_4444_8u:
                {
                }
                break;

                case PrPixelFormat_ARGB_4444_16u:
                {
                }
                break;

                case PrPixelFormat_ARGB_4444_32f:
                {
                }
                break;

                case PrPixelFormat_PRGB_4444_32f:
                {
                }
                break;

                case PrPixelFormat_XRGB_4444_32f:
                {
                }
                break;

                case PrPixelFormat_ARGB_4444_32f_Linear:
                {
                }
                break;

                case PrPixelFormat_PRGB_4444_32f_Linear:
                {
                }
                break;

                case PrPixelFormat_XRGB_4444_32f_Linear:
                {
                }
                break;

                default:
                    err = PF_Err_INTERNAL_STRUCT_DAMAGED;
                break;
            } // switch (destinationPixelFormat)

        } // if (PF_Err_NONE == (errFormat = pixelFormatSuite->GetPixelFormat(output, &destinationPixelFormat)))
        else
        {
            // error in determine pixel format
            err = PF_Err_UNRECOGNIZED_PARAM_TYPE;
        }

//        free_memory_buffers(algoMemHandler);

    } // if (true == mem_handler_valid (algoMemHandler))
//    else
//        err = PF_Err_OUT_OF_MEMORY;

    return err;
}