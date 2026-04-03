#include "ArtPaint.hpp"
#include "ArtPaintEnums.hpp"
#include "PrSDKAESupport.h"
#include "PaintMemHandler.hpp"
#include "PaintColorConvert.hpp"
#include "PaintAlgoMain.hpp"


PF_Err ProcessImgInPR
(
	PF_InData*   RESTRICT in_data,
	PF_OutData*  RESTRICT out_data,
	PF_ParamDef* RESTRICT params[],
	PF_LayerDef* RESTRICT output
) 
{
	PF_Err err{ PF_Err_NONE };
    const PF_LayerDef* pfLayer = reinterpret_cast<const PF_LayerDef*>(&params[UnderlyingType(ArtPaintControls::ART_PAINT_INPUT)]->u.ld);
    const A_long rawLinePitch = pfLayer->rowbytes;
    const A_long sizeY = pfLayer->extent_hint.bottom - pfLayer->extent_hint.top;
    const A_long sizeX = pfLayer->extent_hint.right  - pfLayer->extent_hint.left;

    MemHandler algoMemHandler = alloc_memory_buffers (sizeX, sizeY);
    if (true == check_memory_buffers(algoMemHandler))
    {
        /* This plugin called frop PR - check video fomat */
        auto const pixelFormatSuite{ AEFX_SuiteScoper<PF_PixelFormatSuite1>(in_data, kPFPixelFormatSuite, kPFPixelFormatSuiteVersion1, out_data) };

        PF_Err errFormat{ PF_Err_INVALID_INDEX };
        PrPixelFormat destinationPixelFormat{ PrPixelFormat_Invalid };

        if (PF_Err_NONE == (errFormat = pixelFormatSuite->GetPixelFormat(output, &destinationPixelFormat)))
        {
            const AlgoControls algoParams = getControlsValues (params);

            switch (destinationPixelFormat)
            {
                case PrPixelFormat_BGRA_4444_8u:
                case PrPixelFormat_BGRP_4444_8u:
                case PrPixelFormat_BGRX_4444_8u:
                {
                    const PF_Pixel_BGRA_8u* RESTRICT localSrc = reinterpret_cast<const PF_Pixel_BGRA_8u* RESTRICT>(pfLayer->data);
                          PF_Pixel_BGRA_8u* RESTRICT localDst = reinterpret_cast<      PF_Pixel_BGRA_8u* RESTRICT>(output->data);
                    const A_long linePitch = rawLinePitch / static_cast<A_long>(PF_Pixel_BGRA_8u_size);

                    convert_BGRA8_to_PlanarYUV_AVX2 (localSrc, algoMemHandler, sizeX, sizeY, linePitch);
                    PaintAlgorithmMain (algoMemHandler, algoParams, sizeX, sizeY);
                    convert_PlanarYUV_to_BGRA8_AVX2 (algoMemHandler, localSrc, localDst, sizeX, sizeY, linePitch, linePitch);
                }
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

        free_memory_buffers(algoMemHandler);
    }
    else
        err = PF_Err_OUT_OF_MEMORY;

	return err;
}
