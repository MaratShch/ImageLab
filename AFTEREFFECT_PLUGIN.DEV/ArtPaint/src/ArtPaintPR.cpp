#include "ArtPaint.hpp"
#include "ArtPaintEnums.hpp"
#include "PrSDKAESupport.h"
#include "PaintMemHandler.hpp"
#include "PaintColorDispatcher.hpp"
#include "PaintColorDispatcherOut.hpp"
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

    const AlgoControls algoParams = getControlsValues(params);

    MemHandler algoMemHandler = alloc_memory_buffers (sizeX, sizeY, algoParams.quality);
    if (true == check_memory_buffers(algoMemHandler))
    {
        /* This plugin called frop PR - check video fomat */
        auto const pixelFormatSuite{ AEFX_SuiteScoper<PF_PixelFormatSuite1>(in_data, kPFPixelFormatSuite, kPFPixelFormatSuiteVersion1, out_data) };

        PF_Err errFormat{ PF_Err_INVALID_INDEX };
        PrPixelFormat destinationPixelFormat{ PrPixelFormat_Invalid };

        if (PF_Err_NONE == (errFormat = pixelFormatSuite->GetPixelFormat(output, &destinationPixelFormat)))
        {
            switch (destinationPixelFormat)
            {
                case PrPixelFormat_BGRA_4444_8u:
                {
                    const PF_Pixel_BGRA_8u* RESTRICT localSrc = reinterpret_cast<const PF_Pixel_BGRA_8u* RESTRICT>(pfLayer->data);
                          PF_Pixel_BGRA_8u* RESTRICT localDst = reinterpret_cast<      PF_Pixel_BGRA_8u* RESTRICT>(output->data);
                    const A_long linePitch = rawLinePitch / static_cast<A_long>(PF_Pixel_BGRA_8u_size);

                    dispatch_convert_to_planar (localSrc, algoMemHandler, sizeX, sizeY, linePitch, PixelFormat::BGRA_8u, algoParams.quality);
                    PaintAlgorithmMain (algoMemHandler, algoParams);
                    dispatch_convert_to_interleaved (algoMemHandler, localSrc, localDst, sizeX, sizeY, linePitch, linePitch, PixelFormat::BGRA_8u, algoParams.quality);
                }
                break;

                case PrPixelFormat_BGRP_4444_8u:
                {
                    const PF_Pixel_BGRP_8u* RESTRICT localSrc = reinterpret_cast<const PF_Pixel_BGRP_8u* RESTRICT>(pfLayer->data);
                          PF_Pixel_BGRP_8u* RESTRICT localDst = reinterpret_cast<      PF_Pixel_BGRP_8u* RESTRICT>(output->data);
                    const A_long linePitch = rawLinePitch / static_cast<A_long>(PF_Pixel_BGRP_8u_size);

                    dispatch_convert_to_planar (localSrc, algoMemHandler, sizeX, sizeY, linePitch, PixelFormat::BGRP_8u, algoParams.quality);
                    PaintAlgorithmMain (algoMemHandler, algoParams);
                    dispatch_convert_to_interleaved (algoMemHandler, localSrc, localDst, sizeX, sizeY, linePitch, linePitch, PixelFormat::BGRP_8u, algoParams.quality);
                }
                break;
                case PrPixelFormat_BGRX_4444_8u:
                {
                    const PF_Pixel_BGRX_8u* RESTRICT localSrc = reinterpret_cast<const PF_Pixel_BGRX_8u* RESTRICT>(pfLayer->data);
                          PF_Pixel_BGRX_8u* RESTRICT localDst = reinterpret_cast<      PF_Pixel_BGRX_8u* RESTRICT>(output->data);
                    const A_long linePitch = rawLinePitch / static_cast<A_long>(PF_Pixel_BGRX_8u_size);

                    dispatch_convert_to_planar (localSrc, algoMemHandler, sizeX, sizeY, linePitch, PixelFormat::BGRX_8u, algoParams.quality);
                    PaintAlgorithmMain (algoMemHandler, algoParams);
                    dispatch_convert_to_interleaved (algoMemHandler, localSrc, localDst, sizeX, sizeY, linePitch, linePitch, PixelFormat::BGRX_8u, algoParams.quality);
                }
                break;

                case PrPixelFormat_BGRA_4444_16u:
                {
                    const PF_Pixel_BGRA_16u* RESTRICT localSrc = reinterpret_cast<const PF_Pixel_BGRA_16u* RESTRICT>(pfLayer->data);
                          PF_Pixel_BGRA_16u* RESTRICT localDst = reinterpret_cast<      PF_Pixel_BGRA_16u* RESTRICT>(output->data);
                    const A_long linePitch = rawLinePitch / static_cast<A_long>(PF_Pixel_BGRA_16u_size);

                    dispatch_convert_to_planar (localSrc, algoMemHandler, sizeX, sizeY, linePitch, PixelFormat::BGRA_16u, algoParams.quality);
                    PaintAlgorithmMain (algoMemHandler, algoParams);
                    dispatch_convert_to_interleaved (algoMemHandler, localSrc, localDst, sizeX, sizeY, linePitch, linePitch, PixelFormat::BGRA_16u, algoParams.quality);
                }
                break;

                case PrPixelFormat_BGRP_4444_16u:
                {
                    const PF_Pixel_BGRP_16u* RESTRICT localSrc = reinterpret_cast<const PF_Pixel_BGRP_16u* RESTRICT>(pfLayer->data);
                          PF_Pixel_BGRP_16u* RESTRICT localDst = reinterpret_cast<      PF_Pixel_BGRP_16u* RESTRICT>(output->data);
                    const A_long linePitch = rawLinePitch / static_cast<A_long>(PF_Pixel_BGRP_16u_size);

                    dispatch_convert_to_planar (localSrc, algoMemHandler, sizeX, sizeY, linePitch, PixelFormat::BGRP_16u, algoParams.quality);
                    PaintAlgorithmMain (algoMemHandler, algoParams);
                    dispatch_convert_to_interleaved (algoMemHandler, localSrc, localDst, sizeX, sizeY, linePitch, linePitch, PixelFormat::BGRP_16u, algoParams.quality);
                }
                break;

                case PrPixelFormat_BGRX_4444_16u:
                {
                    const PF_Pixel_BGRX_16u* RESTRICT localSrc = reinterpret_cast<const PF_Pixel_BGRX_16u* RESTRICT>(pfLayer->data);
                          PF_Pixel_BGRX_16u* RESTRICT localDst = reinterpret_cast<      PF_Pixel_BGRX_16u* RESTRICT>(output->data);
                    const A_long linePitch = rawLinePitch / static_cast<A_long>(PF_Pixel_BGRX_16u_size);

                    dispatch_convert_to_planar (localSrc, algoMemHandler, sizeX, sizeY, linePitch, PixelFormat::BGRX_16u, algoParams.quality);
                    PaintAlgorithmMain (algoMemHandler, algoParams);
                    dispatch_convert_to_interleaved (algoMemHandler, localSrc, localDst, sizeX, sizeY, linePitch, linePitch, PixelFormat::BGRX_16u, algoParams.quality);
                }
                break;

                case PrPixelFormat_BGRA_4444_32f:
                case PrPixelFormat_BGRA_4444_32f_Linear:
                {
                    const PF_Pixel_BGRA_32f* RESTRICT localSrc = reinterpret_cast<const PF_Pixel_BGRA_32f* RESTRICT>(pfLayer->data);
                          PF_Pixel_BGRA_32f* RESTRICT localDst = reinterpret_cast<      PF_Pixel_BGRA_32f* RESTRICT>(output->data);
                    const A_long linePitch = rawLinePitch / static_cast<A_long>(PF_Pixel_BGRA_32f_size);

                    dispatch_convert_to_planar (localSrc, algoMemHandler, sizeX, sizeY, linePitch, PixelFormat::BGRA_32f, algoParams.quality);
                    PaintAlgorithmMain (algoMemHandler, algoParams);
                    dispatch_convert_to_interleaved (algoMemHandler, localSrc, localDst, sizeX, sizeY, linePitch, linePitch, PixelFormat::BGRA_32f, algoParams.quality);
                }
                break;

                case PrPixelFormat_BGRX_4444_32f:
                case PrPixelFormat_BGRX_4444_32f_Linear:
                {
                    const PF_Pixel_BGRX_32f* RESTRICT localSrc = reinterpret_cast<const PF_Pixel_BGRX_32f* RESTRICT>(pfLayer->data);
                          PF_Pixel_BGRX_32f* RESTRICT localDst = reinterpret_cast<      PF_Pixel_BGRX_32f* RESTRICT>(output->data);
                    const A_long linePitch = rawLinePitch / static_cast<A_long>(PF_Pixel_BGRX_32f_size);

                    dispatch_convert_to_planar (localSrc, algoMemHandler, sizeX, sizeY, linePitch, PixelFormat::BGRX_32f, algoParams.quality);
                    PaintAlgorithmMain (algoMemHandler, algoParams);
                    dispatch_convert_to_interleaved (algoMemHandler, localSrc, localDst, sizeX, sizeY, linePitch, linePitch, PixelFormat::BGRX_32f, algoParams.quality);
                }
                break;

                case PrPixelFormat_BGRP_4444_32f:
                case PrPixelFormat_BGRP_4444_32f_Linear:
                {
                    const PF_Pixel_BGRP_32f* RESTRICT localSrc = reinterpret_cast<const PF_Pixel_BGRP_32f* RESTRICT>(pfLayer->data);
                          PF_Pixel_BGRP_32f* RESTRICT localDst = reinterpret_cast<      PF_Pixel_BGRP_32f* RESTRICT>(output->data);
                    const A_long linePitch = rawLinePitch / static_cast<A_long>(PF_Pixel_BGRP_32f_size);

                    dispatch_convert_to_planar (localSrc, algoMemHandler, sizeX, sizeY, linePitch, PixelFormat::BGRP_32f, algoParams.quality);
                    PaintAlgorithmMain (algoMemHandler, algoParams);
                    dispatch_convert_to_interleaved (algoMemHandler, localSrc, localDst, sizeX, sizeY, linePitch, linePitch, PixelFormat::BGRP_32f, algoParams.quality);
                }
                break;

                case PrPixelFormat_VUYA_4444_8u_709:
                case PrPixelFormat_VUYA_4444_8u:
                {
                    const PF_Pixel_VUYA_8u* RESTRICT localSrc = reinterpret_cast<const PF_Pixel_VUYA_8u* RESTRICT>(pfLayer->data);
                          PF_Pixel_VUYA_8u* RESTRICT localDst = reinterpret_cast<      PF_Pixel_VUYA_8u* RESTRICT>(output->data);
                    const A_long linePitch = rawLinePitch / static_cast<A_long>(PF_Pixel_VUYA_8u_size);

                    dispatch_convert_to_planar (localSrc, algoMemHandler, sizeX, sizeY, linePitch, PixelFormat::VUYA_8u, algoParams.quality);
                    PaintAlgorithmMain (algoMemHandler, algoParams);
                    dispatch_convert_to_interleaved (algoMemHandler, localSrc, localDst, sizeX, sizeY, linePitch, linePitch, PixelFormat::VUYA_8u, algoParams.quality);
                }
                break;

                case PrPixelFormat_VUYP_4444_8u_709:
                case PrPixelFormat_VUYP_4444_8u:
                {
                    const PF_Pixel_VUYP_8u* RESTRICT localSrc = reinterpret_cast<const PF_Pixel_VUYP_8u* RESTRICT>(pfLayer->data);
                          PF_Pixel_VUYP_8u* RESTRICT localDst = reinterpret_cast<      PF_Pixel_VUYP_8u* RESTRICT>(output->data);
                    const A_long linePitch = rawLinePitch / static_cast<A_long>(PF_Pixel_VUYP_8u_size);

                    dispatch_convert_to_planar (localSrc, algoMemHandler, sizeX, sizeY, linePitch, PixelFormat::VUYP_8u, algoParams.quality);
                    PaintAlgorithmMain (algoMemHandler, algoParams);
                    dispatch_convert_to_interleaved (algoMemHandler, localSrc, localDst, sizeX, sizeY, linePitch, linePitch, PixelFormat::VUYP_8u, algoParams.quality);
                }
                break;

                case PrPixelFormat_VUYA_4444_32f_709:
                case PrPixelFormat_VUYA_4444_32f:
                {
                    const PF_Pixel_VUYA_32f* RESTRICT localSrc = reinterpret_cast<const PF_Pixel_VUYA_32f* RESTRICT>(pfLayer->data);
                          PF_Pixel_VUYA_32f* RESTRICT localDst = reinterpret_cast<      PF_Pixel_VUYA_32f* RESTRICT>(output->data);
                    const A_long linePitch = rawLinePitch / static_cast<A_long>(PF_Pixel_VUYA_32f_size);

                    dispatch_convert_to_planar (localSrc, algoMemHandler, sizeX, sizeY, linePitch, PixelFormat::VUYA_32f, algoParams.quality);
                    PaintAlgorithmMain (algoMemHandler, algoParams);
                    dispatch_convert_to_interleaved (algoMemHandler, localSrc, localDst, sizeX, sizeY, linePitch, linePitch, PixelFormat::VUYA_32f, algoParams.quality);
                }
                break;

                case PrPixelFormat_VUYP_4444_32f_709:
                case PrPixelFormat_VUYP_4444_32f:
                {
                    const PF_Pixel_VUYP_32f* RESTRICT localSrc = reinterpret_cast<const PF_Pixel_VUYP_32f* RESTRICT>(pfLayer->data);
                          PF_Pixel_VUYP_32f* RESTRICT localDst = reinterpret_cast<      PF_Pixel_VUYP_32f* RESTRICT>(output->data);
                    const A_long linePitch = rawLinePitch / static_cast<A_long>(PF_Pixel_VUYP_32f_size);

                    dispatch_convert_to_planar (localSrc, algoMemHandler, sizeX, sizeY, linePitch, PixelFormat::VUYP_32f, algoParams.quality);
                    PaintAlgorithmMain (algoMemHandler, algoParams);
                    dispatch_convert_to_interleaved (algoMemHandler, localSrc, localDst, sizeX, sizeY, linePitch, linePitch, PixelFormat::VUYP_32f, algoParams.quality);
                }
                break;

                case PrPixelFormat_RGB_444_10u:
                {
                    const PF_Pixel_RGB_10u* RESTRICT localSrc = reinterpret_cast<const PF_Pixel_RGB_10u* RESTRICT>(pfLayer->data);
                          PF_Pixel_RGB_10u* RESTRICT localDst = reinterpret_cast<      PF_Pixel_RGB_10u* RESTRICT>(output->data);
                    const A_long linePitch = rawLinePitch / static_cast<A_long>(PF_Pixel_RGB_10u_size);

                    dispatch_convert_to_planar (localSrc, algoMemHandler, sizeX, sizeY, linePitch, PixelFormat::RGB_10u, algoParams.quality);
                    PaintAlgorithmMain (algoMemHandler, algoParams);
                    dispatch_convert_to_interleaved (algoMemHandler, localSrc, localDst, sizeX, sizeY, linePitch, linePitch, PixelFormat::RGB_10u, algoParams.quality);
                }
                break;

                case PrPixelFormat_ARGB_4444_8u:
                {
                    const PF_Pixel_ARGB_8u* RESTRICT localSrc = reinterpret_cast<const PF_Pixel_ARGB_8u* RESTRICT>(pfLayer->data);
                          PF_Pixel_ARGB_8u* RESTRICT localDst = reinterpret_cast<      PF_Pixel_ARGB_8u* RESTRICT>(output->data);
                    const A_long linePitch = rawLinePitch / static_cast<A_long>(PF_Pixel_ARGB_8u_size);

                    dispatch_convert_to_planar (localSrc, algoMemHandler, sizeX, sizeY, linePitch, PixelFormat::ARGB_8u, algoParams.quality);
                    PaintAlgorithmMain (algoMemHandler, algoParams);
                    dispatch_convert_to_interleaved (algoMemHandler, localSrc, localDst, sizeX, sizeY, linePitch, linePitch, PixelFormat::ARGB_8u, algoParams.quality);
                }
                break;

                case PrPixelFormat_ARGB_4444_16u:
                {
                    const PF_Pixel_ARGB_16u* RESTRICT localSrc = reinterpret_cast<const PF_Pixel_ARGB_16u* RESTRICT>(pfLayer->data);
                          PF_Pixel_ARGB_16u* RESTRICT localDst = reinterpret_cast<      PF_Pixel_ARGB_16u* RESTRICT>(output->data);
                    const A_long linePitch = rawLinePitch / static_cast<A_long>(PF_Pixel_ARGB_16u_size);

                    dispatch_convert_to_planar (localSrc, algoMemHandler, sizeX, sizeY, linePitch, PixelFormat::ARGB_16u, algoParams.quality);
                    PaintAlgorithmMain (algoMemHandler, algoParams);
                    dispatch_convert_to_interleaved (algoMemHandler, localSrc, localDst, sizeX, sizeY, linePitch, linePitch, PixelFormat::ARGB_16u, algoParams.quality);
                }
                break;

                case PrPixelFormat_ARGB_4444_32f:
                case PrPixelFormat_ARGB_4444_32f_Linear:
                {
                    const PF_Pixel_ARGB_32f* RESTRICT localSrc = reinterpret_cast<const PF_Pixel_ARGB_32f* RESTRICT>(pfLayer->data);
                          PF_Pixel_ARGB_32f* RESTRICT localDst = reinterpret_cast<      PF_Pixel_ARGB_32f* RESTRICT>(output->data);
                    const A_long linePitch = rawLinePitch / static_cast<A_long>(PF_Pixel_ARGB_32f_size);

                    dispatch_convert_to_planar (localSrc, algoMemHandler, sizeX, sizeY, linePitch, PixelFormat::ARGB_32f, algoParams.quality);
                    PaintAlgorithmMain (algoMemHandler, algoParams);
                    dispatch_convert_to_interleaved (algoMemHandler, localSrc, localDst, sizeX, sizeY, linePitch, linePitch, PixelFormat::ARGB_32f, algoParams.quality);
                }

                default:
                    err = PF_Err_INVALID_INDEX;
                break;
            } /* switch (destinationPixelFormat) */

        } /* if (PF_Err_NONE == (errFormat = pixelFormatSuite->GetPixelFormat(output, &destinationPixelFormat))) */
        else
        {
            /* error in determine pixel format */
            err = PF_Err_UNRECOGNIZED_PARAM_TYPE;
        }
        free_memory_buffers (algoMemHandler);
    }
    else
        err = PF_Err_OUT_OF_MEMORY;

	return err;
}
