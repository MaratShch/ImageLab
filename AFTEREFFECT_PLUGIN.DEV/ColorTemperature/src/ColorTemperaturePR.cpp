#include "ColorTemperature.hpp"
#include "ColorTemperatureEnums.hpp"
#include "ColorTemperatureControlsPresets.hpp"
#include "CompileTimeUtils.hpp"
#include "CommonAuxPixFormat.hpp"
#include "PrSDKAESupport.h"
#include "AlgorithmMain.hpp"
#include "AlgoPrFormatIngest.hpp"
#include "LinearLut/LinearLut.hpp"


PF_Err ProcessImgInPR
(
	PF_InData*   in_data,
	PF_OutData*  out_data,
	PF_ParamDef* params[],
	PF_LayerDef* output
) 
{
    PF_Err err{ PF_Err_NONE };

    // This plugin called from PR - check video fomat
    const PF_LayerDef* pfLayer = reinterpret_cast<const PF_LayerDef*>(&params[COLOR_TEMPERATURE_FILTER_INPUT]->u.ld);
    const A_long sizeY = pfLayer->extent_hint.bottom - pfLayer->extent_hint.top;
    const A_long sizeX = pfLayer->extent_hint.right  - pfLayer->extent_hint.left;

    static const auto& lut8  = LinLut_srgb_8bit_double::LINEARIZE_LUT_SRGB_8BIT_F64;
    static const auto& lut16 = LinLut_srgb_16bit_double::LINEARIZE_LUT_SRGB_16BIT_F64;
    static const auto& lut10 = LinLut_srgb_10bit_double::LINEARIZE_LUT_SRGB_10BIT_F64;

    MemHandler algoMemHandler = alloc_memory_buffers(sizeX, sizeY);
    if (true == mem_handler_valid(algoMemHandler))
    {
        // will be replaced byreal control values captured from effect control items
        const AlgoControls algoCtrl = getAlgoControlsDefault();

        /* This plugin called frop PR - check video fomat */
        auto const pixelFormatSuite{ AEFX_SuiteScoper<PF_PixelFormatSuite1>(in_data, kPFPixelFormatSuite, kPFPixelFormatSuiteVersion1, out_data) };
        PrPixelFormat destinationPixelFormat{ PrPixelFormat_Invalid };
        PF_Err errFormat{ PF_Err_INVALID_INDEX };

        if (PF_Err_NONE == (errFormat = pixelFormatSuite->GetPixelFormat(output, &destinationPixelFormat)))
        {
            SuperPixel<double> super{};     // computed in double, max accuracy
            CctDuv<double> cct_duv{};       // computed in double, max accuracy

            switch (destinationPixelFormat)
            {
                case PrPixelFormat_BGRA_4444_8u:
                {
                    const A_long srcStride = pfLayer->rowbytes / static_cast<A_long>(PF_Pixel_BGRA_8u_size);
                    const A_long dstStride = srcStride;

//                    ingest_and_superpixel(imgInBuffer, sizeX, sizeY, srcPitch, fmt, lut8, lut16, lut10,
//                        locusGate, srcRGB_f32, super, algoControls.confidenceMap);

//                    AlgorithMain (pfLayer->data, output->data, algoMemHandler, algoCtrl, sizeX, sizeY, srcStride, dstStride, AlgoPrIngest::fmt_BGRA_4444_8u);
                }
                break;

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
                case PrPixelFormat_PRGB_4444_8u:
                case PrPixelFormat_XRGB_4444_8u:
                case PrPixelFormat_ARGB_4444_16u:
                case PrPixelFormat_PRGB_4444_16u:
                case PrPixelFormat_XRGB_4444_16u:
                case PrPixelFormat_ARGB_4444_32f:
                case PrPixelFormat_PRGB_4444_32f:
                case PrPixelFormat_XRGB_4444_32f:
                case PrPixelFormat_ARGB_4444_32f_Linear:
                case PrPixelFormat_PRGB_4444_32f_Linear:
                case PrPixelFormat_XRGB_4444_32f_Linear:
                case PrPixelFormat_RGB_444_10u:
                break;

                default:
                    err = PF_Err_INVALID_INDEX;
                break;
            } // switch (destinationPixelFormat)

        } // if (PF_Err_NONE == (errFormat = pixelFormatSuite->GetPixelFormat(output, &destinationPixelFormat)))
        else
        {
            // error in determine pixel format
            err = PF_Err_UNRECOGNIZED_PARAM_TYPE;
        }

        free_memory_buffers (algoMemHandler);
    }
    else
    {
        err = PF_Err_OUT_OF_MEMORY;
    }

    return err;
}
