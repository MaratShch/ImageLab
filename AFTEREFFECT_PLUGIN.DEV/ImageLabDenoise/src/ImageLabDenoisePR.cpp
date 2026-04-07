#include "ImageLabDenoise.hpp"
#include "ImageLabDenoiseEnum.hpp"
#include "AlgoMemHandler.hpp"
#include "AlgoControls.hpp"
#include "AlgorithmMain.hpp"
#include "DenoiseColorDispatcher.hpp"
#include "DenoiseColorDispatcherOut.hpp"
#include "PrSDKAESupport.h"


PF_Err ProcessImgInPR
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

    const PF_LayerDef* pfLayer = reinterpret_cast<const PF_LayerDef*>(&params[UnderlyingType(eDenoiseControl::eIMAGE_LAB_DENOISE_INPUT)]->u.ld);
    const A_long sizeY = pfLayer->extent_hint.bottom - pfLayer->extent_hint.top;
    const A_long sizeX = pfLayer->extent_hint.right - pfLayer->extent_hint.left;

    MemHandler algoMemHandler = alloc_memory_buffers(sizeX, sizeY);
    if (true == mem_handler_valid(algoMemHandler))
    {
        // This plugin called frop PR - check video fomat
        auto const pixelFormatSuite{ AEFX_SuiteScoper<PF_PixelFormatSuite1>(in_data, kPFPixelFormatSuite, kPFPixelFormatSuiteVersion1, out_data) };

        if (PF_Err_NONE == (errFormat = pixelFormatSuite->GetPixelFormat(output, &destinationPixelFormat)))
        {
            const AlgoControls algoControls = GetControlParametersStruct (params);

            switch (destinationPixelFormat)
            {
                case PrPixelFormat_BGRA_4444_8u:
                {
                    const PF_Pixel_BGRA_8u* __restrict localSrc = reinterpret_cast<const PF_Pixel_BGRA_8u* __restrict>(pfLayer->data);
                          PF_Pixel_BGRA_8u* __restrict localDst = reinterpret_cast<      PF_Pixel_BGRA_8u* __restrict>(output->data);
                    const A_long linePitch = pfLayer->rowbytes / static_cast<A_long>(PF_Pixel_BGRA_8u_size);

                    dispatch_convert_to_planar (localSrc, algoMemHandler, sizeX, sizeY, linePitch, PixelFormat::BGRA_8u);

                    // call algorithm flow
                    Algorithm_Main (algoMemHandler, sizeX, sizeY, algoControls);

                    dispatch_convert_to_interleaved (algoMemHandler, localSrc, localDst, sizeX, sizeY, linePitch, linePitch, PixelFormat::BGRA_8u);
                }
                break;

                case PrPixelFormat_BGRP_4444_8u:
                {
                    const PF_Pixel_BGRP_8u* __restrict localSrc = reinterpret_cast<const PF_Pixel_BGRP_8u* __restrict>(pfLayer->data);
                          PF_Pixel_BGRP_8u* __restrict localDst = reinterpret_cast<      PF_Pixel_BGRP_8u* __restrict>(output->data);
                    const A_long linePitch = pfLayer->rowbytes / static_cast<A_long>(PF_Pixel_BGRP_8u_size);

                    dispatch_convert_to_planar(localSrc, algoMemHandler, sizeX, sizeY, linePitch, PixelFormat::BGRP_8u);

                    // call algorithm flow
                    Algorithm_Main(algoMemHandler, sizeX, sizeY, algoControls);

                    dispatch_convert_to_interleaved(algoMemHandler, localSrc, localDst, sizeX, sizeY, linePitch, linePitch, PixelFormat::BGRP_8u);
                }
                break;

                case PrPixelFormat_BGRX_4444_8u:
                {
                    const PF_Pixel_BGRX_8u* __restrict localSrc = reinterpret_cast<const PF_Pixel_BGRX_8u* __restrict>(pfLayer->data);
                          PF_Pixel_BGRX_8u* __restrict localDst = reinterpret_cast<      PF_Pixel_BGRX_8u* __restrict>(output->data);
                    const A_long linePitch = pfLayer->rowbytes / static_cast<A_long>(PF_Pixel_BGRX_8u_size);

                    dispatch_convert_to_planar (localSrc, algoMemHandler, sizeX, sizeY, linePitch, PixelFormat::BGRX_8u);

                    // call algorithm flow
                    Algorithm_Main (algoMemHandler, sizeX, sizeY, algoControls);

                    dispatch_convert_to_interleaved (algoMemHandler, localSrc, localDst, sizeX, sizeY, linePitch, linePitch, PixelFormat::BGRX_8u);
                }
                break;

                case PrPixelFormat_BGRA_4444_16u:
                {
                    const PF_Pixel_BGRA_16u* __restrict localSrc = reinterpret_cast<const PF_Pixel_BGRA_16u* __restrict>(pfLayer->data);
                          PF_Pixel_BGRA_16u* __restrict localDst = reinterpret_cast<      PF_Pixel_BGRA_16u* __restrict>(output->data);
                    const A_long linePitch = pfLayer->rowbytes / static_cast<A_long>(PF_Pixel_BGRA_16u_size);

                    dispatch_convert_to_planar (localSrc, algoMemHandler, sizeX, sizeY, linePitch, PixelFormat::BGRA_16u);

                    // call algorithm flow
                    Algorithm_Main (algoMemHandler, sizeX, sizeY, algoControls);

                    dispatch_convert_to_interleaved (algoMemHandler, localSrc, localDst, sizeX, sizeY, linePitch, linePitch, PixelFormat::BGRA_16u);
                }
                break;

                case PrPixelFormat_BGRP_4444_16u:
                {
                    const PF_Pixel_BGRP_16u* __restrict localSrc = reinterpret_cast<const PF_Pixel_BGRP_16u* __restrict>(pfLayer->data);
                          PF_Pixel_BGRP_16u* __restrict localDst = reinterpret_cast<      PF_Pixel_BGRP_16u* __restrict>(output->data);
                    const A_long linePitch = pfLayer->rowbytes / static_cast<A_long>(PF_Pixel_BGRP_16u_size);

                    dispatch_convert_to_planar (localSrc, algoMemHandler, sizeX, sizeY, linePitch, PixelFormat::BGRP_16u);

                    // call algorithm flow
                    Algorithm_Main (algoMemHandler, sizeX, sizeY, algoControls);

                    dispatch_convert_to_interleaved (algoMemHandler, localSrc, localDst, sizeX, sizeY, linePitch, linePitch, PixelFormat::BGRP_16u);
                }
                break;

                case PrPixelFormat_BGRX_4444_16u:
                {
                    const PF_Pixel_BGRX_16u* __restrict localSrc = reinterpret_cast<const PF_Pixel_BGRX_16u* __restrict>(pfLayer->data);
                          PF_Pixel_BGRX_16u* __restrict localDst = reinterpret_cast<      PF_Pixel_BGRX_16u* __restrict>(output->data);
                    const A_long linePitch = pfLayer->rowbytes / static_cast<A_long>(PF_Pixel_BGRX_16u_size);
                    
                    dispatch_convert_to_planar (localSrc, algoMemHandler, sizeX, sizeY, linePitch, PixelFormat::BGRX_16u);

                    // call algorithm flow
                    Algorithm_Main (algoMemHandler, sizeX, sizeY, algoControls);

                    dispatch_convert_to_interleaved (algoMemHandler, localSrc, localDst, sizeX, sizeY, linePitch, linePitch, PixelFormat::BGRX_16u);
                }
                break;

                case PrPixelFormat_BGRA_4444_32f:
                {
                    const PF_Pixel_BGRA_32f* __restrict localSrc = reinterpret_cast<const PF_Pixel_BGRA_32f* __restrict>(pfLayer->data);
                          PF_Pixel_BGRA_32f* __restrict localDst = reinterpret_cast<      PF_Pixel_BGRA_32f* __restrict>(output->data);
                    const A_long linePitch = pfLayer->rowbytes / static_cast<A_long>(PF_Pixel_BGRA_32f_size);
                    
                    dispatch_convert_to_planar (localSrc, algoMemHandler, sizeX, sizeY, linePitch, PixelFormat::BGRA_32f);

                    // call algorithm flow
                    Algorithm_Main (algoMemHandler, sizeX, sizeY, algoControls);

                    dispatch_convert_to_interleaved (algoMemHandler, localSrc, localDst, sizeX, sizeY, linePitch, linePitch, PixelFormat::BGRA_32f);
                }
                break;

                case PrPixelFormat_BGRP_4444_32f:
                {
                    const PF_Pixel_BGRP_32f* __restrict localSrc = reinterpret_cast<const PF_Pixel_BGRP_32f* __restrict>(pfLayer->data);
                          PF_Pixel_BGRP_32f* __restrict localDst = reinterpret_cast<      PF_Pixel_BGRP_32f* __restrict>(output->data);
                    const A_long linePitch = pfLayer->rowbytes / static_cast<A_long>(PF_Pixel_BGRP_32f_size);
                    
                    dispatch_convert_to_planar (localSrc, algoMemHandler, sizeX, sizeY, linePitch, PixelFormat::BGRP_32f);

                    // call algorithm flow
                    Algorithm_Main (algoMemHandler, sizeX, sizeY, algoControls);

                    dispatch_convert_to_interleaved (algoMemHandler, localSrc, localDst, sizeX, sizeY, linePitch, linePitch, PixelFormat::BGRP_32f);
                }
                break;

                case PrPixelFormat_BGRX_4444_32f:
                {
                    const PF_Pixel_BGRX_32f* __restrict localSrc = reinterpret_cast<const PF_Pixel_BGRX_32f* __restrict>(pfLayer->data);
                          PF_Pixel_BGRX_32f* __restrict localDst = reinterpret_cast<      PF_Pixel_BGRX_32f* __restrict>(output->data);
                    const A_long linePitch = pfLayer->rowbytes / static_cast<A_long>(PF_Pixel_BGRX_32f_size);

                    dispatch_convert_to_planar (localSrc, algoMemHandler, sizeX, sizeY, linePitch, PixelFormat::BGRX_32f);

                    // call algorithm flow
                    Algorithm_Main (algoMemHandler, sizeX, sizeY, algoControls);

                    dispatch_convert_to_interleaved (algoMemHandler, localSrc, localDst, sizeX, sizeY, linePitch, linePitch, PixelFormat::BGRX_32f);
                }
                break;

                case PrPixelFormat_BGRA_4444_32f_Linear:
                {
                    const PF_Pixel_BGRA_32f* __restrict localSrc = reinterpret_cast<const PF_Pixel_BGRA_32f* __restrict>(pfLayer->data);
                          PF_Pixel_BGRA_32f* __restrict localDst = reinterpret_cast<      PF_Pixel_BGRA_32f* __restrict>(output->data);
                    const A_long linePitch = pfLayer->rowbytes / static_cast<A_long>(PF_Pixel_BGRA_32f_size);

                    dispatch_convert_to_planar (localSrc, algoMemHandler, sizeX, sizeY, linePitch, PixelFormat::BGRA_32f_Linear);

                    // call algorithm flow
                    Algorithm_Main (algoMemHandler, sizeX, sizeY, algoControls);

                    dispatch_convert_to_interleaved (algoMemHandler, localSrc, localDst, sizeX, sizeY, linePitch, linePitch, PixelFormat::BGRA_32f_Linear);
                }
                break;

                case PrPixelFormat_BGRP_4444_32f_Linear:
                {
                    const PF_Pixel_BGRP_32f* __restrict localSrc = reinterpret_cast<const PF_Pixel_BGRP_32f* __restrict>(pfLayer->data);
                          PF_Pixel_BGRP_32f* __restrict localDst = reinterpret_cast<      PF_Pixel_BGRP_32f* __restrict>(output->data);
                    const A_long linePitch = pfLayer->rowbytes / static_cast<A_long>(PF_Pixel_BGRP_32f_size);

                    dispatch_convert_to_planar (localSrc, algoMemHandler, sizeX, sizeY, linePitch, PixelFormat::BGRP_32f_Linear);

                    // call algorithm flow
                    Algorithm_Main (algoMemHandler, sizeX, sizeY, algoControls);

                    dispatch_convert_to_interleaved (algoMemHandler, localSrc, localDst, sizeX, sizeY, linePitch, linePitch, PixelFormat::BGRP_32f_Linear);
                }
                break;

                case PrPixelFormat_BGRX_4444_32f_Linear:
                {
                    const PF_Pixel_BGRX_32f* __restrict localSrc = reinterpret_cast<const PF_Pixel_BGRX_32f* __restrict>(pfLayer->data);
                          PF_Pixel_BGRX_32f* __restrict localDst = reinterpret_cast<      PF_Pixel_BGRX_32f* __restrict>(output->data);
                    const A_long linePitch = pfLayer->rowbytes / static_cast<A_long>(PF_Pixel_BGRX_32f_size);

                    dispatch_convert_to_planar (localSrc, algoMemHandler, sizeX, sizeY, linePitch, PixelFormat::BGRX_32f_Linear);

                    // call algorithm flow
                    Algorithm_Main (algoMemHandler, sizeX, sizeY, algoControls);
                     
                    dispatch_convert_to_interleaved (algoMemHandler, localSrc, localDst, sizeX, sizeY, linePitch, linePitch, PixelFormat::BGRX_32f_Linear);
                }
                break;

                case PrPixelFormat_VUYA_4444_8u:
                {
                    const PF_Pixel_VUYA_8u* __restrict localSrc = reinterpret_cast<const PF_Pixel_VUYA_8u* __restrict>(pfLayer->data);
                          PF_Pixel_VUYA_8u* __restrict localDst = reinterpret_cast<      PF_Pixel_VUYA_8u* __restrict>(output->data);
                    const A_long linePitch = pfLayer->rowbytes / static_cast<A_long>(PF_Pixel_VUYA_8u_size);

                    dispatch_convert_to_planar (localSrc, algoMemHandler, sizeX, sizeY, linePitch, PixelFormat::VUYA_8u);

                    // call algorithm flow
                    Algorithm_Main (algoMemHandler, sizeX, sizeY, algoControls);
                     
                    dispatch_convert_to_interleaved (algoMemHandler, localSrc, localDst, sizeX, sizeY, linePitch, linePitch, PixelFormat::VUYA_8u);
                }
                break;

                case PrPixelFormat_VUYA_4444_8u_709:
                {
                    const PF_Pixel_VUYA_8u* __restrict localSrc = reinterpret_cast<const PF_Pixel_VUYA_8u* __restrict>(pfLayer->data);
                          PF_Pixel_VUYA_8u* __restrict localDst = reinterpret_cast<      PF_Pixel_VUYA_8u* __restrict>(output->data);
                    const A_long linePitch = pfLayer->rowbytes / static_cast<A_long>(PF_Pixel_VUYA_8u_size);
                    
                    dispatch_convert_to_planar (localSrc, algoMemHandler, sizeX, sizeY, linePitch, PixelFormat::VUYA_8u_709);

                    // call algorithm flow
                    Algorithm_Main (algoMemHandler, sizeX, sizeY, algoControls);
                     
                    dispatch_convert_to_interleaved (algoMemHandler, localSrc, localDst, sizeX, sizeY, linePitch, linePitch, PixelFormat::VUYA_8u_709);
                }
                break;

                case PrPixelFormat_VUYP_4444_8u:
                {
                    const PF_Pixel_VUYP_8u* __restrict localSrc = reinterpret_cast<const PF_Pixel_VUYP_8u* __restrict>(pfLayer->data);
                          PF_Pixel_VUYP_8u* __restrict localDst = reinterpret_cast<      PF_Pixel_VUYP_8u* __restrict>(output->data);
                    const A_long linePitch = pfLayer->rowbytes / static_cast<A_long>(PF_Pixel_VUYP_8u_size);
                    
                    dispatch_convert_to_planar (localSrc, algoMemHandler, sizeX, sizeY, linePitch, PixelFormat::VUYP_8u);

                    // call algorithm flow
                    Algorithm_Main (algoMemHandler, sizeX, sizeY, algoControls);
                     
                    dispatch_convert_to_interleaved (algoMemHandler, localSrc, localDst, sizeX, sizeY, linePitch, linePitch, PixelFormat::VUYP_8u);
                }
                break;

                case PrPixelFormat_VUYP_4444_8u_709:
                {
                    const PF_Pixel_VUYP_8u* __restrict localSrc = reinterpret_cast<const PF_Pixel_VUYP_8u* __restrict>(pfLayer->data);
                          PF_Pixel_VUYP_8u* __restrict localDst = reinterpret_cast<      PF_Pixel_VUYP_8u* __restrict>(output->data);
                    const A_long linePitch = pfLayer->rowbytes / static_cast<A_long>(PF_Pixel_VUYP_8u_size);

                    dispatch_convert_to_planar (localSrc, algoMemHandler, sizeX, sizeY, linePitch, PixelFormat::VUYP_8u_709);

                    // call algorithm flow
                    Algorithm_Main (algoMemHandler, sizeX, sizeY, algoControls);
                     
                    dispatch_convert_to_interleaved (algoMemHandler, localSrc, localDst, sizeX, sizeY, linePitch, linePitch, PixelFormat::VUYP_8u_709);
                }
                break;

                case PrPixelFormat_VUYA_4444_32f:
                {
                    const PF_Pixel_VUYA_32f* __restrict localSrc = reinterpret_cast<const PF_Pixel_VUYA_32f* __restrict>(pfLayer->data);
                          PF_Pixel_VUYA_32f* __restrict localDst = reinterpret_cast<      PF_Pixel_VUYA_32f* __restrict>(output->data);
                    const A_long linePitch = pfLayer->rowbytes / static_cast<A_long>(PF_Pixel_VUYA_32f_size);
                    
                    dispatch_convert_to_planar(localSrc, algoMemHandler, sizeX, sizeY, linePitch, PixelFormat::VUYA_32f);

                    // call algorithm flow
                    Algorithm_Main(algoMemHandler, sizeX, sizeY, algoControls);

                    dispatch_convert_to_interleaved(algoMemHandler, localSrc, localDst, sizeX, sizeY, linePitch, linePitch, PixelFormat::VUYA_32f);
                }
                break;

                case PrPixelFormat_VUYA_4444_32f_709:
                {
                    const PF_Pixel_VUYA_32f* __restrict localSrc = reinterpret_cast<const PF_Pixel_VUYA_32f* __restrict>(pfLayer->data);
                          PF_Pixel_VUYA_32f* __restrict localDst = reinterpret_cast<      PF_Pixel_VUYA_32f* __restrict>(output->data);
                    const A_long linePitch = pfLayer->rowbytes / static_cast<A_long>(PF_Pixel_VUYA_32f_size);

                    dispatch_convert_to_planar (localSrc, algoMemHandler, sizeX, sizeY, linePitch, PixelFormat::VUYA_32f_709);

                    // call algorithm flow
                    Algorithm_Main (algoMemHandler, sizeX, sizeY, algoControls);

                    dispatch_convert_to_interleaved (algoMemHandler, localSrc, localDst, sizeX, sizeY, linePitch, linePitch, PixelFormat::VUYA_32f_709);
                }
                break;

                case PrPixelFormat_VUYP_4444_32f:
                {
                    const PF_Pixel_VUYP_32f* __restrict localSrc = reinterpret_cast<const PF_Pixel_VUYP_32f* __restrict>(pfLayer->data);
                          PF_Pixel_VUYP_32f* __restrict localDst = reinterpret_cast<      PF_Pixel_VUYP_32f* __restrict>(output->data);
                    const A_long linePitch = pfLayer->rowbytes / static_cast<A_long>(PF_Pixel_VUYP_32f_size);

                    dispatch_convert_to_planar (localSrc, algoMemHandler, sizeX, sizeY, linePitch, PixelFormat::VUYP_32f);

                    // call algorithm flow
                    Algorithm_Main (algoMemHandler, sizeX, sizeY, algoControls);

                    dispatch_convert_to_interleaved (algoMemHandler, localSrc, localDst, sizeX, sizeY, linePitch, linePitch, PixelFormat::VUYP_32f);
                }
                break;

                case PrPixelFormat_VUYP_4444_32f_709:
                {
                    const PF_Pixel_VUYP_32f* __restrict localSrc = reinterpret_cast<const PF_Pixel_VUYP_32f* __restrict>(pfLayer->data);
                          PF_Pixel_VUYP_32f* __restrict localDst = reinterpret_cast<      PF_Pixel_VUYP_32f* __restrict>(output->data);
                    const A_long linePitch = pfLayer->rowbytes / static_cast<A_long>(PF_Pixel_VUYP_32f_size);

                    dispatch_convert_to_planar (localSrc, algoMemHandler, sizeX, sizeY, linePitch, PixelFormat::VUYP_32f_709);

                    // call algorithm flow
                    Algorithm_Main (algoMemHandler, sizeX, sizeY, algoControls);

                    dispatch_convert_to_interleaved (algoMemHandler, localSrc, localDst, sizeX, sizeY, linePitch, linePitch, PixelFormat::VUYP_32f_709);
                }
                break;

                case PrPixelFormat_RGB_444_10u:
                {
                    const PF_Pixel_RGB_10u* __restrict localSrc = reinterpret_cast<const PF_Pixel_RGB_10u* __restrict>(pfLayer->data);
                          PF_Pixel_RGB_10u* __restrict localDst = reinterpret_cast<      PF_Pixel_RGB_10u* __restrict>(output->data);
                    const A_long linePitch = pfLayer->rowbytes / static_cast<A_long>(PF_Pixel_RGB_10u_size);

                    dispatch_convert_to_planar(localSrc, algoMemHandler, sizeX, sizeY, linePitch, PixelFormat::RGB_10u);

                    // call algorithm flow
                    Algorithm_Main(algoMemHandler, sizeX, sizeY, algoControls);

                    dispatch_convert_to_interleaved(algoMemHandler, localSrc, localDst, sizeX, sizeY, linePitch, linePitch, PixelFormat::RGB_10u);
                }
                break;

                case PrPixelFormat_ARGB_4444_8u:
                {
                    const PF_Pixel_ARGB_8u* __restrict localSrc = reinterpret_cast<const PF_Pixel_ARGB_8u* __restrict>(pfLayer->data);
                          PF_Pixel_ARGB_8u* __restrict localDst = reinterpret_cast<      PF_Pixel_ARGB_8u* __restrict>(output->data);
                    const A_long linePitch = pfLayer->rowbytes / static_cast<A_long>(PF_Pixel_ARGB_8u_size);

                    dispatch_convert_to_planar (localSrc, algoMemHandler, sizeX, sizeY, linePitch, PixelFormat::ARGB_8u);

                    // call algorithm flow
                    Algorithm_Main (algoMemHandler, sizeX, sizeY, algoControls);

                    dispatch_convert_to_interleaved (algoMemHandler, localSrc, localDst, sizeX, sizeY, linePitch, linePitch, PixelFormat::ARGB_8u);
                }
                break;

                case PrPixelFormat_ARGB_4444_16u:
                {
                    const PF_Pixel_ARGB_16u* __restrict localSrc = reinterpret_cast<const PF_Pixel_ARGB_16u* __restrict>(pfLayer->data);
                          PF_Pixel_ARGB_16u* __restrict localDst = reinterpret_cast<      PF_Pixel_ARGB_16u* __restrict>(output->data);
                    const A_long linePitch = pfLayer->rowbytes / static_cast<A_long>(PF_Pixel_ARGB_16u_size);

                    dispatch_convert_to_planar (localSrc, algoMemHandler, sizeX, sizeY, linePitch, PixelFormat::ARGB_16u);

                    // call algorithm flow
                    Algorithm_Main (algoMemHandler, sizeX, sizeY, algoControls);

                    dispatch_convert_to_interleaved (algoMemHandler, localSrc, localDst, sizeX, sizeY, linePitch, linePitch, PixelFormat::ARGB_16u);
                }
                break;

                case PrPixelFormat_ARGB_4444_32f:
                {
                    const PF_Pixel_ARGB_32f* __restrict localSrc = reinterpret_cast<const PF_Pixel_ARGB_32f* __restrict>(pfLayer->data);
                          PF_Pixel_ARGB_32f* __restrict localDst = reinterpret_cast<      PF_Pixel_ARGB_32f* __restrict>(output->data);
                    const A_long linePitch = pfLayer->rowbytes / static_cast<A_long>(PF_Pixel_ARGB_32f_size);

                    dispatch_convert_to_planar (localSrc, algoMemHandler, sizeX, sizeY, linePitch, PixelFormat::ARGB_32f);

                    // call algorithm flow
                    Algorithm_Main (algoMemHandler, sizeX, sizeY, algoControls);

                    dispatch_convert_to_interleaved (algoMemHandler, localSrc, localDst, sizeX, sizeY, linePitch, linePitch, PixelFormat::ARGB_32f);
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

        free_memory_buffers (algoMemHandler);

    } // if (true == mem_handler_valid (algoMemHandler))
    else
        err = PF_Err_OUT_OF_MEMORY;

    return err;
}
