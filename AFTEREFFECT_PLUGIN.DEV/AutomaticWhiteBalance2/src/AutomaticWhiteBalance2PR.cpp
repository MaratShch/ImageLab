#include "AutomaticWhiteBalance2.hpp"
#include "AutomaticWhiteBalance2Enum.hpp"
#include "AlgoMemHandler.hpp"
#include "AlgoControl.hpp"
#include "AlgorithmMain.hpp"
#include "AlgConvertDispatcher.hpp"
#include "AlgConvertDispatcherOut.hpp"


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
    const A_long sizeX = pfLayer->extent_hint.right  - pfLayer->extent_hint.left;
    const A_long rowBytes = pfLayer->rowbytes;

    MemHandler algoMemHandler = alloc_memory_buffers (sizeX, sizeY);
    if (true == mem_handler_valid(algoMemHandler))
    {
        // This plugin called frop PR - check video format
        auto const pixelFormatSuite{ AEFX_SuiteScoper<PF_PixelFormatSuite1>(in_data, kPFPixelFormatSuite, kPFPixelFormatSuiteVersion1, out_data) };

        if (PF_Err_NONE == (errFormat = pixelFormatSuite->GetPixelFormat(output, &destinationPixelFormat)))
        {
            const AlgoControls algoControls = getAlgoControlsParameters (params);

            switch (destinationPixelFormat)
            {
                case PrPixelFormat_BGRA_4444_8u:
                {
                    const PF_Pixel_BGRA_8u* __restrict localSrc = reinterpret_cast<const PF_Pixel_BGRA_8u* __restrict>(pfLayer->data);
                          PF_Pixel_BGRA_8u* __restrict localDst = reinterpret_cast<      PF_Pixel_BGRA_8u* __restrict>(output->data);
                    const A_long srcLinePitch = rowBytes / static_cast<A_long>(PF_Pixel_BGRA_8u_size);
                    const A_long dstLinePitch = srcLinePitch;

                    dispatch_convert_to_planar (localSrc, algoMemHandler, sizeX, sizeY, srcLinePitch, PixelFormat::BGRA_8u);
                    Algorithm_Main (algoMemHandler, sizeX, sizeY, algoControls);
                    dispatch_convert_to_interleaved (algoMemHandler, localSrc, localDst, sizeX, sizeY, srcLinePitch, dstLinePitch, PixelFormat::BGRA_8u);
                }
                break;

                case PrPixelFormat_BGRA_4444_16u:
                {
                    const PF_Pixel_BGRA_16u* __restrict localSrc = reinterpret_cast<const PF_Pixel_BGRA_16u* __restrict>(pfLayer->data);
                          PF_Pixel_BGRA_16u* __restrict localDst = reinterpret_cast<      PF_Pixel_BGRA_16u* __restrict>(output->data);
                    const A_long srcLinePitch = rowBytes / static_cast<A_long>(PF_Pixel_BGRA_16u_size);
                    const A_long dstLinePitch = srcLinePitch;

                    dispatch_convert_to_planar (localSrc, algoMemHandler, sizeX, sizeY, srcLinePitch, PixelFormat::BGRA_16u);
                    Algorithm_Main (algoMemHandler, sizeX, sizeY, algoControls);
                    dispatch_convert_to_interleaved (algoMemHandler, localSrc, localDst, sizeX, sizeY, srcLinePitch, dstLinePitch, PixelFormat::BGRA_16u);
                }
                break;

                case PrPixelFormat_BGRX_4444_8u:
                {
                    const PF_Pixel_BGRX_8u* __restrict localSrc = reinterpret_cast<const PF_Pixel_BGRX_8u* __restrict>(pfLayer->data);
                          PF_Pixel_BGRX_8u* __restrict localDst = reinterpret_cast<      PF_Pixel_BGRX_8u* __restrict>(output->data);
                    const A_long srcLinePitch = rowBytes / static_cast<A_long>(PF_Pixel_BGRX_8u_size);
                    const A_long dstLinePitch = srcLinePitch;

                    dispatch_convert_to_planar (localSrc, algoMemHandler, sizeX, sizeY, srcLinePitch, PixelFormat::BGRX_8u);
                    Algorithm_Main (algoMemHandler, sizeX, sizeY, algoControls);
                    dispatch_convert_to_interleaved (algoMemHandler, localSrc, localDst, sizeX, sizeY, srcLinePitch, dstLinePitch, PixelFormat::BGRX_8u);
                }
                break;

                case PrPixelFormat_BGRA_4444_32f:
                {
                    const PF_Pixel_BGRA_32f* __restrict localSrc = reinterpret_cast<const PF_Pixel_BGRA_32f* __restrict>(pfLayer->data);
                          PF_Pixel_BGRA_32f* __restrict localDst = reinterpret_cast<      PF_Pixel_BGRA_32f* __restrict>(output->data);
                    const A_long srcLinePitch = rowBytes / static_cast<A_long>(PF_Pixel_BGRA_32f_size);
                    const A_long dstLinePitch = srcLinePitch;

                    dispatch_convert_to_planar (localSrc, algoMemHandler, sizeX, sizeY, srcLinePitch, PixelFormat::BGRA_32f);
                    Algorithm_Main (algoMemHandler, sizeX, sizeY, algoControls);
                    dispatch_convert_to_interleaved (algoMemHandler, localSrc, localDst, sizeX, sizeY, srcLinePitch, dstLinePitch, PixelFormat::BGRA_32f);
                }
                break;

                case PrPixelFormat_BGRA_4444_32f_Linear:
                {
                    const PF_Pixel_BGRA_32f* __restrict localSrc = reinterpret_cast<const PF_Pixel_BGRA_32f* __restrict>(pfLayer->data);
                          PF_Pixel_BGRA_32f* __restrict localDst = reinterpret_cast<      PF_Pixel_BGRA_32f* __restrict>(output->data);
                    const A_long srcLinePitch = rowBytes / static_cast<A_long>(PF_Pixel_BGRA_32f_size);
                    const A_long dstLinePitch = srcLinePitch;

                    dispatch_convert_to_planar (localSrc, algoMemHandler, sizeX, sizeY, srcLinePitch, PixelFormat::BGRA_32f_Linear);
                    Algorithm_Main (algoMemHandler, sizeX, sizeY, algoControls);
                    dispatch_convert_to_interleaved (algoMemHandler, localSrc, localDst, sizeX, sizeY, srcLinePitch, dstLinePitch, PixelFormat::BGRA_32f_Linear);
                }
                break;

                case PrPixelFormat_BGRP_4444_8u:
                {
                    const PF_Pixel_BGRP_8u* __restrict localSrc = reinterpret_cast<const PF_Pixel_BGRP_8u* __restrict>(pfLayer->data);
                          PF_Pixel_BGRP_8u* __restrict localDst = reinterpret_cast<      PF_Pixel_BGRP_8u* __restrict>(output->data);
                    const A_long srcLinePitch = rowBytes / static_cast<A_long>(PF_Pixel_BGRP_8u_size);
                    const A_long dstLinePitch = srcLinePitch;

                    dispatch_convert_to_planar (localSrc, algoMemHandler, sizeX, sizeY, srcLinePitch, PixelFormat::BGRP_8u);
                    Algorithm_Main (algoMemHandler, sizeX, sizeY, algoControls);
                    dispatch_convert_to_interleaved (algoMemHandler, localSrc, localDst, sizeX, sizeY, srcLinePitch, dstLinePitch, PixelFormat::BGRP_8u);
                }
                break;

                case PrPixelFormat_BGRP_4444_16u:
                {
                    const PF_Pixel_BGRP_16u* __restrict localSrc = reinterpret_cast<const PF_Pixel_BGRP_16u* __restrict>(pfLayer->data);
                          PF_Pixel_BGRP_16u* __restrict localDst = reinterpret_cast<      PF_Pixel_BGRP_16u* __restrict>(output->data);
                    const A_long srcLinePitch = rowBytes / static_cast<A_long>(PF_Pixel_BGRP_16u_size);
                    const A_long dstLinePitch = srcLinePitch;

                    dispatch_convert_to_planar (localSrc, algoMemHandler, sizeX, sizeY, srcLinePitch, PixelFormat::BGRP_16u);
                    Algorithm_Main (algoMemHandler, sizeX, sizeY, algoControls);
                    dispatch_convert_to_interleaved (algoMemHandler, localSrc, localDst, sizeX, sizeY, srcLinePitch, dstLinePitch, PixelFormat::BGRP_16u);
                }
                break;

                case PrPixelFormat_BGRP_4444_32f:
                {
                    const PF_Pixel_BGRP_32f* __restrict localSrc = reinterpret_cast<const PF_Pixel_BGRP_32f* __restrict>(pfLayer->data);
                          PF_Pixel_BGRP_32f* __restrict localDst = reinterpret_cast<      PF_Pixel_BGRP_32f* __restrict>(output->data);
                    const A_long srcLinePitch = rowBytes / static_cast<A_long>(PF_Pixel_BGRP_32f_size);
                    const A_long dstLinePitch = srcLinePitch;

                    dispatch_convert_to_planar (localSrc, algoMemHandler, sizeX, sizeY, srcLinePitch, PixelFormat::BGRP_32f);
                    Algorithm_Main (algoMemHandler, sizeX, sizeY, algoControls);
                    dispatch_convert_to_interleaved (algoMemHandler, localSrc, localDst, sizeX, sizeY, srcLinePitch, dstLinePitch, PixelFormat::BGRP_32f);
                }
                break;

                case PrPixelFormat_BGRP_4444_32f_Linear:
                {
                    const PF_Pixel_BGRP_32f* __restrict localSrc = reinterpret_cast<const PF_Pixel_BGRP_32f* __restrict>(pfLayer->data);
                          PF_Pixel_BGRP_32f* __restrict localDst = reinterpret_cast<      PF_Pixel_BGRP_32f* __restrict>(output->data);
                    const A_long srcLinePitch = rowBytes / static_cast<A_long>(PF_Pixel_BGRP_32f_size);
                    const A_long dstLinePitch = srcLinePitch;

                    dispatch_convert_to_planar (localSrc, algoMemHandler, sizeX, sizeY, srcLinePitch, PixelFormat::BGRP_32f_Linear);
                    Algorithm_Main (algoMemHandler, sizeX, sizeY, algoControls);
                    dispatch_convert_to_interleaved (algoMemHandler, localSrc, localDst, sizeX, sizeY, srcLinePitch, dstLinePitch, PixelFormat::BGRP_32f_Linear);
                }
                break;

                case PrPixelFormat_BGRX_4444_16u:
                {
                    const PF_Pixel_BGRX_16u* __restrict localSrc = reinterpret_cast<const PF_Pixel_BGRX_16u* __restrict>(pfLayer->data);
                          PF_Pixel_BGRX_16u* __restrict localDst = reinterpret_cast<      PF_Pixel_BGRX_16u* __restrict>(output->data);
                    const A_long srcLinePitch = rowBytes / static_cast<A_long>(PF_Pixel_BGRX_16u_size);
                    const A_long dstLinePitch = srcLinePitch;

                    dispatch_convert_to_planar (localSrc, algoMemHandler, sizeX, sizeY, srcLinePitch, PixelFormat::BGRX_16u);
                    Algorithm_Main (algoMemHandler, sizeX, sizeY, algoControls);
                    dispatch_convert_to_interleaved (algoMemHandler, localSrc, localDst, sizeX, sizeY, srcLinePitch, dstLinePitch, PixelFormat::BGRX_16u);
                }
                break;

                case PrPixelFormat_BGRX_4444_32f:
                {
                    const PF_Pixel_BGRX_32f* __restrict localSrc = reinterpret_cast<const PF_Pixel_BGRX_32f* __restrict>(pfLayer->data);
                          PF_Pixel_BGRX_32f* __restrict localDst = reinterpret_cast<      PF_Pixel_BGRX_32f* __restrict>(output->data);
                    const A_long srcLinePitch = rowBytes / static_cast<A_long>(PF_Pixel_BGRX_32f_size);
                    const A_long dstLinePitch = srcLinePitch;

                    dispatch_convert_to_planar (localSrc, algoMemHandler, sizeX, sizeY, srcLinePitch, PixelFormat::BGRX_32f);
                    Algorithm_Main (algoMemHandler, sizeX, sizeY, algoControls);
                    dispatch_convert_to_interleaved (algoMemHandler, localSrc, localDst, sizeX, sizeY, srcLinePitch, dstLinePitch, PixelFormat::BGRX_32f);
                }
                break;

                case PrPixelFormat_BGRX_4444_32f_Linear:
                {
                    const PF_Pixel_BGRX_32f* __restrict localSrc = reinterpret_cast<const PF_Pixel_BGRX_32f* __restrict>(pfLayer->data);
                          PF_Pixel_BGRX_32f* __restrict localDst = reinterpret_cast<      PF_Pixel_BGRX_32f* __restrict>(output->data);
                    const A_long srcLinePitch = rowBytes / static_cast<A_long>(PF_Pixel_BGRX_32f_size);
                    const A_long dstLinePitch = srcLinePitch;

                    dispatch_convert_to_planar (localSrc, algoMemHandler, sizeX, sizeY, srcLinePitch, PixelFormat::BGRX_32f_Linear);
                    Algorithm_Main (algoMemHandler, sizeX, sizeY, algoControls);
                    dispatch_convert_to_interleaved (algoMemHandler, localSrc, localDst, sizeX, sizeY, srcLinePitch, dstLinePitch, PixelFormat::BGRX_32f_Linear);
                }
                break;

                case PrPixelFormat_VUYA_4444_8u_709:
                {
                    const PF_Pixel_VUYA_8u* __restrict localSrc = reinterpret_cast<const PF_Pixel_VUYA_8u* __restrict>(pfLayer->data);
                          PF_Pixel_VUYA_8u* __restrict localDst = reinterpret_cast<      PF_Pixel_VUYA_8u* __restrict>(output->data);
                    const A_long srcLinePitch = rowBytes / static_cast<A_long>(PF_Pixel_VUYA_8u_size);
                    const A_long dstLinePitch = srcLinePitch;

                    dispatch_convert_to_planar (localSrc, algoMemHandler, sizeX, sizeY, srcLinePitch, PixelFormat::VUYA_8u_709);
                    Algorithm_Main (algoMemHandler, sizeX, sizeY, algoControls);
                    dispatch_convert_to_interleaved (algoMemHandler, localSrc, localDst, sizeX, sizeY, srcLinePitch, dstLinePitch, PixelFormat::VUYA_8u_709);
                }
                break;

                case PrPixelFormat_VUYA_4444_8u:
                {
                    const PF_Pixel_VUYA_8u* __restrict localSrc = reinterpret_cast<const PF_Pixel_VUYA_8u* __restrict>(pfLayer->data);
                          PF_Pixel_VUYA_8u* __restrict localDst = reinterpret_cast<      PF_Pixel_VUYA_8u* __restrict>(output->data);
                    const A_long srcLinePitch = rowBytes / static_cast<A_long>(PF_Pixel_VUYA_8u_size);
                    const A_long dstLinePitch = srcLinePitch;

                    dispatch_convert_to_planar (localSrc, algoMemHandler, sizeX, sizeY, srcLinePitch, PixelFormat::VUYA_8u);
                    Algorithm_Main (algoMemHandler, sizeX, sizeY, algoControls);
                    dispatch_convert_to_interleaved (algoMemHandler, localSrc, localDst, sizeX, sizeY, srcLinePitch, dstLinePitch, PixelFormat::VUYA_8u);
                }
                break;

                case PrPixelFormat_VUYA_4444_32f_709:
                {
                    const PF_Pixel_VUYA_32f* __restrict localSrc = reinterpret_cast<const PF_Pixel_VUYA_32f* __restrict>(pfLayer->data);
                          PF_Pixel_VUYA_32f* __restrict localDst = reinterpret_cast<      PF_Pixel_VUYA_32f* __restrict>(output->data);
                    const A_long srcLinePitch = rowBytes / static_cast<A_long>(PF_Pixel_VUYA_32f_size);
                    const A_long dstLinePitch = srcLinePitch;

                    dispatch_convert_to_planar (localSrc, algoMemHandler, sizeX, sizeY, srcLinePitch, PixelFormat::VUYA_32f_709);
                    Algorithm_Main (algoMemHandler, sizeX, sizeY, algoControls);
                    dispatch_convert_to_interleaved (algoMemHandler, localSrc, localDst, sizeX, sizeY, srcLinePitch, dstLinePitch, PixelFormat::VUYA_32f_709);
                }
                break;

                case PrPixelFormat_VUYA_4444_32f:
                {
                    const PF_Pixel_VUYA_32f* __restrict localSrc = reinterpret_cast<const PF_Pixel_VUYA_32f* __restrict>(pfLayer->data);
                          PF_Pixel_VUYA_32f* __restrict localDst = reinterpret_cast<      PF_Pixel_VUYA_32f* __restrict>(output->data);
                    const A_long srcLinePitch = rowBytes / static_cast<A_long>(PF_Pixel_VUYA_32f_size);
                    const A_long dstLinePitch = srcLinePitch;

                    dispatch_convert_to_planar (localSrc, algoMemHandler, sizeX, sizeY, srcLinePitch, PixelFormat::VUYA_32f);
                    Algorithm_Main (algoMemHandler, sizeX, sizeY, algoControls);
                    dispatch_convert_to_interleaved (algoMemHandler, localSrc, localDst, sizeX, sizeY, srcLinePitch, dstLinePitch, PixelFormat::VUYA_32f);
                }
                break;

                case PrPixelFormat_VUYP_4444_8u_709:
                {
                    const PF_Pixel_VUYP_8u* __restrict localSrc = reinterpret_cast<const PF_Pixel_VUYP_8u* __restrict>(pfLayer->data);
                          PF_Pixel_VUYP_8u* __restrict localDst = reinterpret_cast<      PF_Pixel_VUYP_8u* __restrict>(output->data);
                    const A_long srcLinePitch = rowBytes / static_cast<A_long>(PF_Pixel_VUYA_8u_size);
                    const A_long dstLinePitch = srcLinePitch;

                    dispatch_convert_to_planar (localSrc, algoMemHandler, sizeX, sizeY, srcLinePitch, PixelFormat::VUYP_8u_709);
                    Algorithm_Main (algoMemHandler, sizeX, sizeY, algoControls);
                    dispatch_convert_to_interleaved (algoMemHandler, localSrc, localDst, sizeX, sizeY, srcLinePitch, dstLinePitch, PixelFormat::VUYP_8u_709);
                }
                break;

                case PrPixelFormat_VUYP_4444_8u:
                {
                    const PF_Pixel_VUYP_8u* __restrict localSrc = reinterpret_cast<const PF_Pixel_VUYP_8u* __restrict>(pfLayer->data);
                          PF_Pixel_VUYP_8u* __restrict localDst = reinterpret_cast<      PF_Pixel_VUYP_8u* __restrict>(output->data);
                    const A_long srcLinePitch = rowBytes / static_cast<A_long>(PF_Pixel_VUYP_8u_size);
                    const A_long dstLinePitch = srcLinePitch;

                    dispatch_convert_to_planar(localSrc, algoMemHandler, sizeX, sizeY, srcLinePitch, PixelFormat::VUYP_8u);
                    Algorithm_Main(algoMemHandler, sizeX, sizeY, algoControls);
                    dispatch_convert_to_interleaved(algoMemHandler, localSrc, localDst, sizeX, sizeY, srcLinePitch, dstLinePitch, PixelFormat::VUYP_8u);
                }
                break;

                case PrPixelFormat_VUYP_4444_32f_709:
                {
                    const PF_Pixel_VUYP_32f* __restrict localSrc = reinterpret_cast<const PF_Pixel_VUYP_32f* __restrict>(pfLayer->data);
                          PF_Pixel_VUYP_32f* __restrict localDst = reinterpret_cast<      PF_Pixel_VUYP_32f* __restrict>(output->data);
                    const A_long srcLinePitch = rowBytes / static_cast<A_long>(PF_Pixel_VUYP_32f_size);
                    const A_long dstLinePitch = srcLinePitch;

                    dispatch_convert_to_planar (localSrc, algoMemHandler, sizeX, sizeY, srcLinePitch, PixelFormat::VUYP_32f_709);
                    Algorithm_Main (algoMemHandler, sizeX, sizeY, algoControls);
                    dispatch_convert_to_interleaved (algoMemHandler, localSrc, localDst, sizeX, sizeY, srcLinePitch, dstLinePitch, PixelFormat::VUYP_32f_709);
                }
                break;

                case PrPixelFormat_VUYP_4444_32f:
                {
                    const PF_Pixel_VUYP_32f* __restrict localSrc = reinterpret_cast<const PF_Pixel_VUYP_32f* __restrict>(pfLayer->data);
                          PF_Pixel_VUYP_32f* __restrict localDst = reinterpret_cast<      PF_Pixel_VUYP_32f* __restrict>(output->data);
                    const A_long srcLinePitch = rowBytes / static_cast<A_long>(PF_Pixel_VUYP_32f_size);
                    const A_long dstLinePitch = srcLinePitch;

                    dispatch_convert_to_planar (localSrc, algoMemHandler, sizeX, sizeY, srcLinePitch, PixelFormat::VUYP_32f);
                    Algorithm_Main (algoMemHandler, sizeX, sizeY, algoControls);
                    dispatch_convert_to_interleaved (algoMemHandler, localSrc, localDst, sizeX, sizeY, srcLinePitch, dstLinePitch, PixelFormat::VUYP_32f);
                }
                break;

                case PrPixelFormat_VUYX_4444_8u_709:
                {
                    const PF_Pixel_VUYX_8u* __restrict localSrc = reinterpret_cast<const PF_Pixel_VUYX_8u* __restrict>(pfLayer->data);
                          PF_Pixel_VUYX_8u* __restrict localDst = reinterpret_cast<      PF_Pixel_VUYX_8u* __restrict>(output->data);
                    const A_long srcLinePitch = rowBytes / static_cast<A_long>(PF_Pixel_VUYX_8u_size);
                    const A_long dstLinePitch = srcLinePitch;

                    dispatch_convert_to_planar (localSrc, algoMemHandler, sizeX, sizeY, srcLinePitch, PixelFormat::VUYX_8u_709);
                    Algorithm_Main (algoMemHandler, sizeX, sizeY, algoControls);
                    dispatch_convert_to_interleaved (algoMemHandler, localSrc, localDst, sizeX, sizeY, srcLinePitch, dstLinePitch, PixelFormat::VUYX_8u_709);
                }
                break;

                case PrPixelFormat_VUYX_4444_8u:
                {
                    const PF_Pixel_VUYX_8u* __restrict localSrc = reinterpret_cast<const PF_Pixel_VUYX_8u* __restrict>(pfLayer->data);
                          PF_Pixel_VUYX_8u* __restrict localDst = reinterpret_cast<      PF_Pixel_VUYX_8u* __restrict>(output->data);
                    const A_long srcLinePitch = rowBytes / static_cast<A_long>(PF_Pixel_VUYX_8u_size);
                    const A_long dstLinePitch = srcLinePitch;

                    dispatch_convert_to_planar (localSrc, algoMemHandler, sizeX, sizeY, srcLinePitch, PixelFormat::VUYX_8u);
                    Algorithm_Main (algoMemHandler, sizeX, sizeY, algoControls);
                    dispatch_convert_to_interleaved (algoMemHandler, localSrc, localDst, sizeX, sizeY, srcLinePitch, dstLinePitch, PixelFormat::VUYX_8u);
                }
                break;

                case PrPixelFormat_VUYX_4444_32f_709:
                {
                    const PF_Pixel_VUYX_32f* __restrict localSrc = reinterpret_cast<const PF_Pixel_VUYX_32f* __restrict>(pfLayer->data);
                          PF_Pixel_VUYX_32f* __restrict localDst = reinterpret_cast<      PF_Pixel_VUYX_32f* __restrict>(output->data);
                    const A_long srcLinePitch = rowBytes / static_cast<A_long>(PF_Pixel_VUYX_32f_size);
                    const A_long dstLinePitch = srcLinePitch;

                    dispatch_convert_to_planar (localSrc, algoMemHandler, sizeX, sizeY, srcLinePitch, PixelFormat::VUYX_32f_709);
                    Algorithm_Main (algoMemHandler, sizeX, sizeY, algoControls);
                    dispatch_convert_to_interleaved (algoMemHandler, localSrc, localDst, sizeX, sizeY, srcLinePitch, dstLinePitch, PixelFormat::VUYX_32f_709);
                }
                break;

                case PrPixelFormat_VUYX_4444_32f:
                {
                    const PF_Pixel_VUYX_32f* __restrict localSrc = reinterpret_cast<const PF_Pixel_VUYX_32f* __restrict>(pfLayer->data);
                          PF_Pixel_VUYX_32f* __restrict localDst = reinterpret_cast<      PF_Pixel_VUYX_32f* __restrict>(output->data);
                    const A_long srcLinePitch = rowBytes / static_cast<A_long>(PF_Pixel_VUYX_32f_size);
                    const A_long dstLinePitch = srcLinePitch;

                    dispatch_convert_to_planar (localSrc, algoMemHandler, sizeX, sizeY, srcLinePitch, PixelFormat::VUYX_32f);
                    Algorithm_Main (algoMemHandler, sizeX, sizeY, algoControls);
                    dispatch_convert_to_interleaved (algoMemHandler, localSrc, localDst, sizeX, sizeY, srcLinePitch, dstLinePitch, PixelFormat::VUYX_32f);
                }
                break;

                case PrPixelFormat_RGB_444_10u:
                {
                    const PF_Pixel_RGB_10u* __restrict localSrc = reinterpret_cast<const PF_Pixel_RGB_10u* __restrict>(pfLayer->data);
                          PF_Pixel_RGB_10u* __restrict localDst = reinterpret_cast<      PF_Pixel_RGB_10u* __restrict>(output->data);
                    const A_long srcLinePitch = rowBytes / static_cast<A_long>(PF_Pixel_RGB_10u_size);
                    const A_long dstLinePitch = srcLinePitch;

                    dispatch_convert_to_planar (localSrc, algoMemHandler, sizeX, sizeY, srcLinePitch, PixelFormat::RGB_10u);
                    Algorithm_Main (algoMemHandler, sizeX, sizeY, algoControls);
                    dispatch_convert_to_interleaved (algoMemHandler, localSrc, localDst, sizeX, sizeY, srcLinePitch, dstLinePitch, PixelFormat::RGB_10u);
                }
                break;

                case PrPixelFormat_ARGB_4444_8u:
                {
                    const PF_Pixel_ARGB_8u* __restrict localSrc = reinterpret_cast<const PF_Pixel_ARGB_8u* __restrict>(pfLayer->data);
                          PF_Pixel_ARGB_8u* __restrict localDst = reinterpret_cast<      PF_Pixel_ARGB_8u* __restrict>(output->data);
                    const A_long srcLinePitch = rowBytes / static_cast<A_long>(PF_Pixel_ARGB_8u_size);
                    const A_long dstLinePitch = srcLinePitch;

                    dispatch_convert_to_planar (localSrc, algoMemHandler, sizeX, sizeY, srcLinePitch, PixelFormat::ARGB_8u);
                    Algorithm_Main (algoMemHandler, sizeX, sizeY, algoControls);
                    dispatch_convert_to_interleaved (algoMemHandler, localSrc, localDst, sizeX, sizeY, srcLinePitch, dstLinePitch, PixelFormat::ARGB_8u);
                }
                break;

                case PrPixelFormat_ARGB_4444_16u:
                {
                    const PF_Pixel_ARGB_16u* __restrict localSrc = reinterpret_cast<const PF_Pixel_ARGB_16u* __restrict>(pfLayer->data);
                          PF_Pixel_ARGB_16u* __restrict localDst = reinterpret_cast<      PF_Pixel_ARGB_16u* __restrict>(output->data);
                    const A_long srcLinePitch = rowBytes / static_cast<A_long>(PF_Pixel_ARGB_16u_size);
                    const A_long dstLinePitch = srcLinePitch;

                    dispatch_convert_to_planar (localSrc, algoMemHandler, sizeX, sizeY, srcLinePitch, PixelFormat::ARGB_16u);
                    Algorithm_Main (algoMemHandler, sizeX, sizeY, algoControls);
                    dispatch_convert_to_interleaved (algoMemHandler, localSrc, localDst, sizeX, sizeY, srcLinePitch, dstLinePitch, PixelFormat::ARGB_16u);
                }
                break;

                case PrPixelFormat_ARGB_4444_32f:
                {
                    const PF_Pixel_ARGB_32f* __restrict localSrc = reinterpret_cast<const PF_Pixel_ARGB_32f* __restrict>(pfLayer->data);
                          PF_Pixel_ARGB_32f* __restrict localDst = reinterpret_cast<      PF_Pixel_ARGB_32f* __restrict>(output->data);
                    const A_long srcLinePitch = rowBytes / static_cast<A_long>(PF_Pixel_ARGB_32f_size);
                    const A_long dstLinePitch = srcLinePitch;

                    dispatch_convert_to_planar (localSrc, algoMemHandler, sizeX, sizeY, srcLinePitch, PixelFormat::ARGB_32f);
                    Algorithm_Main (algoMemHandler, sizeX, sizeY, algoControls);
                    dispatch_convert_to_interleaved (algoMemHandler, localSrc, localDst, sizeX, sizeY, srcLinePitch, dstLinePitch, PixelFormat::ARGB_32f);
                }
                break;

                case PrPixelFormat_PRGB_4444_32f:
                {
                    const PF_Pixel_PRGB_32f* __restrict localSrc = reinterpret_cast<const PF_Pixel_PRGB_32f* __restrict>(pfLayer->data);
                          PF_Pixel_PRGB_32f* __restrict localDst = reinterpret_cast<      PF_Pixel_PRGB_32f* __restrict>(output->data);
                    const A_long srcLinePitch = rowBytes / static_cast<A_long>(PF_Pixel_PRGB_32f_size);
                    const A_long dstLinePitch = srcLinePitch;

                    dispatch_convert_to_planar (localSrc, algoMemHandler, sizeX, sizeY, srcLinePitch, PixelFormat::PRGB_32f);
                    Algorithm_Main (algoMemHandler, sizeX, sizeY, algoControls);
                    dispatch_convert_to_interleaved (algoMemHandler, localSrc, localDst, sizeX, sizeY, srcLinePitch, dstLinePitch, PixelFormat::PRGB_32f);
                }
                break;

                case PrPixelFormat_XRGB_4444_32f:
                {
                    const PF_Pixel_XRGB_32f* __restrict localSrc = reinterpret_cast<const PF_Pixel_XRGB_32f* __restrict>(pfLayer->data);
                          PF_Pixel_XRGB_32f* __restrict localDst = reinterpret_cast<      PF_Pixel_XRGB_32f* __restrict>(output->data);
                    const A_long srcLinePitch = rowBytes / static_cast<A_long>(PF_Pixel_XRGB_32f_size);
                    const A_long dstLinePitch = srcLinePitch;

                    dispatch_convert_to_planar (localSrc, algoMemHandler, sizeX, sizeY, srcLinePitch, PixelFormat::XRGB_32f);
                    Algorithm_Main (algoMemHandler, sizeX, sizeY, algoControls);
                    dispatch_convert_to_interleaved (algoMemHandler, localSrc, localDst, sizeX, sizeY, srcLinePitch, dstLinePitch, PixelFormat::XRGB_32f);
                }
                break;

                case PrPixelFormat_ARGB_4444_32f_Linear:
                {
                    const PF_Pixel_ARGB_32f* __restrict localSrc = reinterpret_cast<const PF_Pixel_ARGB_32f* __restrict>(pfLayer->data);
                          PF_Pixel_ARGB_32f* __restrict localDst = reinterpret_cast<      PF_Pixel_ARGB_32f* __restrict>(output->data);
                    const A_long srcLinePitch = rowBytes / static_cast<A_long>(PF_Pixel_ARGB_32f_size);
                    const A_long dstLinePitch = srcLinePitch;

                    dispatch_convert_to_planar (localSrc, algoMemHandler, sizeX, sizeY, srcLinePitch, PixelFormat::ARGB_32f_Linear);
                    Algorithm_Main (algoMemHandler, sizeX, sizeY, algoControls);
                    dispatch_convert_to_interleaved (algoMemHandler, localSrc, localDst, sizeX, sizeY, srcLinePitch, dstLinePitch, PixelFormat::ARGB_32f_Linear);
                }
                break;

                case PrPixelFormat_PRGB_4444_32f_Linear:
                {
                    const PF_Pixel_PRGB_32f* __restrict localSrc = reinterpret_cast<const PF_Pixel_PRGB_32f* __restrict>(pfLayer->data);
                          PF_Pixel_PRGB_32f* __restrict localDst = reinterpret_cast<      PF_Pixel_PRGB_32f* __restrict>(output->data);
                    const A_long srcLinePitch = rowBytes / static_cast<A_long>(PF_Pixel_PRGB_32f_size);
                    const A_long dstLinePitch = srcLinePitch;

                    dispatch_convert_to_planar (localSrc, algoMemHandler, sizeX, sizeY, srcLinePitch, PixelFormat::PRGB_32f_Linear);
                    Algorithm_Main (algoMemHandler, sizeX, sizeY, algoControls);
                    dispatch_convert_to_interleaved (algoMemHandler, localSrc, localDst, sizeX, sizeY, srcLinePitch, dstLinePitch, PixelFormat::PRGB_32f_Linear);
                }
                break;

                case PrPixelFormat_XRGB_4444_32f_Linear:
                {
                    const PF_Pixel_XRGB_32f* __restrict localSrc = reinterpret_cast<const PF_Pixel_XRGB_32f* __restrict>(pfLayer->data);
                          PF_Pixel_XRGB_32f* __restrict localDst = reinterpret_cast<      PF_Pixel_XRGB_32f* __restrict>(output->data);
                    const A_long srcLinePitch = rowBytes / static_cast<A_long>(PF_Pixel_XRGB_32f_size);
                    const A_long dstLinePitch = srcLinePitch;

                    dispatch_convert_to_planar (localSrc, algoMemHandler, sizeX, sizeY, srcLinePitch, PixelFormat::XRGB_32f_Linear);
                    Algorithm_Main (algoMemHandler, sizeX, sizeY, algoControls);
                    dispatch_convert_to_interleaved (algoMemHandler, localSrc, localDst, sizeX, sizeY, srcLinePitch, dstLinePitch, PixelFormat::XRGB_32f_Linear);
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