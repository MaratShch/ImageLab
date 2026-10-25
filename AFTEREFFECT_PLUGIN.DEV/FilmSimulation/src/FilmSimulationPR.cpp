#include "FilmSimulation.hpp"
#include "AlgoControlEnums.hpp"
#include "AlgoMemHandler.hpp"
#include "PrSDKAESupport.h"
#include "AlgoPrFormatIngest.hpp"
#include "AlgoPrFormatEgress.hpp"
#include "AlgorithmMain.hpp"
#include "AlgoAdobeControlEnums.hpp"

using namespace AlgoPrIngest;

PF_Err ProcessImgInPR
(
	PF_InData*    in_data,
	PF_OutData*   out_data,
	PF_ParamDef*  params[],
	PF_LayerDef*  output
)
{
	PF_Err err = PF_Err_NONE;

    // This plugin called from PR - check video fomat
    const PF_LayerDef* pfLayer = reinterpret_cast<const PF_LayerDef*>(&params[UnderlyingType(FilmSimulationCtrl::VIDEO_INPUT)]->u.ld);

    // The WHOLE frame is processed, from (0,0): the film model is spatial over
    // the full frame (blur, halation, flare mean, vignette, damage placement),
    // and the buffers below are addressed from data[0]. The extent_hint
    // rectangle is not an image of its own -- using its size with the origin at
    // (0,0) processed the wrong area whenever the hint did not start at (0,0).
    const A_long sizeX = (pfLayer->width  < output->width)  ? pfLayer->width  : output->width;
    const A_long sizeY = (pfLayer->height < output->height) ? pfLayer->height : output->height;
    const A_long rowBytes = pfLayer->rowbytes;

    if (sizeX <= 0 || sizeY <= 0)
        return PF_Err_NONE;

    MemHandler memHndl = alloc_memory_buffers (static_cast<int32_t>(sizeX), static_cast<int32_t>(sizeY));
    // Frees the arena on EVERY exit path, including an exception thrown by
    // AEFX_SuiteScoper below (caught in EffectMain).
    const MemHandlerGuard memGuard(memHndl);

    if (true == mem_handler_valid(memHndl))
    {
        PF_Err errFormat = PF_Err_INVALID_INDEX;
        PrPixelFormat destinationPixelFormat = PrPixelFormat_Invalid;

        // This plugin called frop PR - check video fomat
        auto const pixelFormatSuite{ AEFX_SuiteScoper<PF_PixelFormatSuite1>(in_data, kPFPixelFormatSuite, kPFPixelFormatSuiteVersion1, out_data) };

        if (PF_Err_NONE == (errFormat = pixelFormatSuite->GetPixelFormat(output, &destinationPixelFormat)))
        {
            const double fpsRate = image_lab_get_fps(in_data);
            const int32_t idx = film_sim_frame_index(in_data);
            const AlgoControls algoControls = getAlgoControls(params, fpsRate, idx);

            switch (destinationPixelFormat)
            {
                case PrPixelFormat_BGRA_4444_8u:
                {
                    const PF_Pixel_BGRA_8u* RESTRICT localSrc = reinterpret_cast<const PF_Pixel_BGRA_8u* RESTRICT>(pfLayer->data);
                          PF_Pixel_BGRA_8u* RESTRICT localDst = reinterpret_cast<      PF_Pixel_BGRA_8u* RESTRICT>(output->data);
                    const A_long srcLinePitch = rowBytes / static_cast<A_long>(PF_Pixel_BGRA_8u_size);
                    const A_long dstLinePitch = output->rowbytes / static_cast<A_long>(PF_Pixel_BGRA_8u_size);

                    // convert from Adobe buffer format to planar 
                    ingest_to_planar_f32 (localSrc, sizeX, sizeY, srcLinePitch, fmt_BGRA_4444_8u, memHndl.Src_R, memHndl.Src_G, memHndl.Src_B, memHndl.padW);

                    // perform film simulations
                    Algorithm_Main (memHndl, sizeX, sizeY, algoControls);

                    // back convert from planar format to Adobe buffer format
                    egress_from_planar_f32 (memHndl.Dst_R, memHndl.Dst_G, memHndl.Dst_B, memHndl.padW, sizeX, sizeY, localDst, dstLinePitch, fmt_BGRA_4444_8u, localSrc, sizeX, srcLinePitch);
                }
                break;

                case PrPixelFormat_BGRA_4444_16u:
                {
                    const PF_Pixel_BGRA_16u* RESTRICT localSrc = reinterpret_cast<const PF_Pixel_BGRA_16u* RESTRICT>(pfLayer->data);
                          PF_Pixel_BGRA_16u* RESTRICT localDst = reinterpret_cast<      PF_Pixel_BGRA_16u* RESTRICT>(output->data);
                    const A_long srcLinePitch = rowBytes / static_cast<A_long>(PF_Pixel_BGRA_16u_size);
                    const A_long dstLinePitch = output->rowbytes / static_cast<A_long>(PF_Pixel_BGRA_16u_size);

                    // convert from Adobe buffer format to planar 
                    ingest_to_planar_f32 (localSrc, sizeX, sizeY, srcLinePitch, fmt_BGRA_4444_16u, memHndl.Src_R, memHndl.Src_G, memHndl.Src_B, memHndl.padW);

                    // perform film simulations
                    Algorithm_Main (memHndl, sizeX, sizeY, algoControls);

                    // back convert from planar format to Adobe buffer format
                    egress_from_planar_f32 (memHndl.Dst_R, memHndl.Dst_G, memHndl.Dst_B, memHndl.padW, sizeX, sizeY, localDst, dstLinePitch, fmt_BGRA_4444_16u, localSrc, sizeX, srcLinePitch);
                }
                break;

                case PrPixelFormat_BGRA_4444_32f:
                {
                    const PF_Pixel_BGRA_32f* RESTRICT localSrc = reinterpret_cast<const PF_Pixel_BGRA_32f* RESTRICT>(pfLayer->data);
                          PF_Pixel_BGRA_32f* RESTRICT localDst = reinterpret_cast<      PF_Pixel_BGRA_32f* RESTRICT>(output->data);
                    const A_long srcLinePitch = rowBytes / static_cast<A_long>(PF_Pixel_BGRA_32f_size);
                    const A_long dstLinePitch = output->rowbytes / static_cast<A_long>(PF_Pixel_BGRA_32f_size);

                    // convert from Adobe buffer format to planar 
                    ingest_to_planar_f32 (localSrc, sizeX, sizeY, srcLinePitch, fmt_BGRA_4444_32f, memHndl.Src_R, memHndl.Src_G, memHndl.Src_B, memHndl.padW);

                    // perform film simulations
                    Algorithm_Main (memHndl, sizeX, sizeY, algoControls);

                    // back convert from planar format to Adobe buffer format
                    egress_from_planar_f32 (memHndl.Dst_R, memHndl.Dst_G, memHndl.Dst_B, memHndl.padW, sizeX, sizeY, localDst, dstLinePitch, fmt_BGRA_4444_32f, localSrc, sizeX, srcLinePitch);
                }
                break;

                case PrPixelFormat_BGRA_4444_32f_Linear:
                {
                    const PF_Pixel_BGRA_32f* RESTRICT localSrc = reinterpret_cast<const PF_Pixel_BGRA_32f* RESTRICT>(pfLayer->data);
                          PF_Pixel_BGRA_32f* RESTRICT localDst = reinterpret_cast<      PF_Pixel_BGRA_32f* RESTRICT>(output->data);
                    const A_long srcLinePitch = rowBytes / static_cast<A_long>(PF_Pixel_BGRA_32f_size);
                    const A_long dstLinePitch = output->rowbytes / static_cast<A_long>(PF_Pixel_BGRA_32f_size);

                    // convert from Adobe buffer format to planar 
                    ingest_to_planar_f32 (localSrc, sizeX, sizeY, srcLinePitch, fmt_BGRA_4444_32f_Linear, memHndl.Src_R, memHndl.Src_G, memHndl.Src_B, memHndl.padW);

                    // perform film simulations
                    Algorithm_Main (memHndl, sizeX, sizeY, algoControls);

                    // back convert from planar format to Adobe buffer format
                    egress_from_planar_f32 (memHndl.Dst_R, memHndl.Dst_G, memHndl.Dst_B, memHndl.padW, sizeX, sizeY, localDst, dstLinePitch, fmt_BGRA_4444_32f_Linear, localSrc, sizeX, srcLinePitch);
                }
                break;

                case PrPixelFormat_BGRP_4444_8u:
                {
                    const PF_Pixel_BGRP_8u* RESTRICT localSrc = reinterpret_cast<const PF_Pixel_BGRP_8u* RESTRICT>(pfLayer->data);
                          PF_Pixel_BGRP_8u* RESTRICT localDst = reinterpret_cast<      PF_Pixel_BGRP_8u* RESTRICT>(output->data);
                    const A_long srcLinePitch = rowBytes / static_cast<A_long>(PF_Pixel_BGRP_8u_size);
                    const A_long dstLinePitch = output->rowbytes / static_cast<A_long>(PF_Pixel_BGRP_8u_size);

                    // convert from Adobe buffer format to planar 
                    ingest_to_planar_f32 (localSrc, sizeX, sizeY, srcLinePitch, fmt_BGRP_4444_8u, memHndl.Src_R, memHndl.Src_G, memHndl.Src_B, memHndl.padW);

                    // perform film simulations
                    Algorithm_Main (memHndl, sizeX, sizeY, algoControls);

                    // back convert from planar format to Adobe buffer format
                    egress_from_planar_f32 (memHndl.Dst_R, memHndl.Dst_G, memHndl.Dst_B, memHndl.padW, sizeX, sizeY, localDst, dstLinePitch, fmt_BGRP_4444_8u, localSrc, sizeX, srcLinePitch);
                }
                break;

                case PrPixelFormat_BGRP_4444_16u:
                {
                    const PF_Pixel_BGRP_16u* RESTRICT localSrc = reinterpret_cast<const PF_Pixel_BGRP_16u* RESTRICT>(pfLayer->data);
                          PF_Pixel_BGRP_16u* RESTRICT localDst = reinterpret_cast<      PF_Pixel_BGRP_16u* RESTRICT>(output->data);
                    const A_long srcLinePitch = rowBytes / static_cast<A_long>(PF_Pixel_BGRP_16u_size);
                    const A_long dstLinePitch = output->rowbytes / static_cast<A_long>(PF_Pixel_BGRP_16u_size);

                    // convert from Adobe buffer format to planar 
                    ingest_to_planar_f32 (localSrc, sizeX, sizeY, srcLinePitch, fmt_BGRP_4444_16u, memHndl.Src_R, memHndl.Src_G, memHndl.Src_B, memHndl.padW);

                    // perform film simulations
                    Algorithm_Main (memHndl, sizeX, sizeY, algoControls);

                    // back convert from planar format to Adobe buffer format
                    egress_from_planar_f32 (memHndl.Dst_R, memHndl.Dst_G, memHndl.Dst_B, memHndl.padW, sizeX, sizeY, localDst, dstLinePitch, fmt_BGRP_4444_16u, localSrc, sizeX, srcLinePitch);
                }
                break;

                case PrPixelFormat_BGRP_4444_32f:
                {
                    const PF_Pixel_BGRP_32f* RESTRICT localSrc = reinterpret_cast<const PF_Pixel_BGRP_32f* RESTRICT>(pfLayer->data);
                          PF_Pixel_BGRP_32f* RESTRICT localDst = reinterpret_cast<      PF_Pixel_BGRP_32f* RESTRICT>(output->data);
                    const A_long srcLinePitch = rowBytes / static_cast<A_long>(PF_Pixel_BGRP_32f_size);
                    const A_long dstLinePitch = output->rowbytes / static_cast<A_long>(PF_Pixel_BGRP_32f_size);

                    // convert from Adobe buffer format to planar 
                    ingest_to_planar_f32 (localSrc, sizeX, sizeY, srcLinePitch, fmt_BGRP_4444_32f, memHndl.Src_R, memHndl.Src_G, memHndl.Src_B, memHndl.padW);

                    // perform film simulations
                    Algorithm_Main (memHndl, sizeX, sizeY, algoControls);

                    // back convert from planar format to Adobe buffer format
                    egress_from_planar_f32 (memHndl.Dst_R, memHndl.Dst_G, memHndl.Dst_B, memHndl.padW, sizeX, sizeY, localDst, dstLinePitch, fmt_BGRP_4444_32f, localSrc, sizeX, srcLinePitch);
                }
                break;

                case PrPixelFormat_BGRP_4444_32f_Linear:
                {
                    const PF_Pixel_BGRP_32f* RESTRICT localSrc = reinterpret_cast<const PF_Pixel_BGRP_32f* RESTRICT>(pfLayer->data);
                          PF_Pixel_BGRP_32f* RESTRICT localDst = reinterpret_cast<      PF_Pixel_BGRP_32f* RESTRICT>(output->data);
                    const A_long srcLinePitch = rowBytes / static_cast<A_long>(PF_Pixel_BGRP_32f_size);
                    const A_long dstLinePitch = output->rowbytes / static_cast<A_long>(PF_Pixel_BGRP_32f_size);

                    // convert from Adobe buffer format to planar 
                    ingest_to_planar_f32 (localSrc, sizeX, sizeY, srcLinePitch, fmt_BGRP_4444_32f_Linear, memHndl.Src_R, memHndl.Src_G, memHndl.Src_B, memHndl.padW);

                    // perform film simulations
                    Algorithm_Main (memHndl, sizeX, sizeY, algoControls);

                    // back convert from planar format to Adobe buffer format
                    egress_from_planar_f32 (memHndl.Dst_R, memHndl.Dst_G, memHndl.Dst_B, memHndl.padW, sizeX, sizeY, localDst, dstLinePitch, fmt_BGRP_4444_32f_Linear, localSrc, sizeX, srcLinePitch);
                }
                break;

                case PrPixelFormat_BGRX_4444_8u:
                {
                    const PF_Pixel_BGRX_8u* RESTRICT localSrc = reinterpret_cast<const PF_Pixel_BGRX_8u* RESTRICT>(pfLayer->data);
                          PF_Pixel_BGRX_8u* RESTRICT localDst = reinterpret_cast<      PF_Pixel_BGRX_8u* RESTRICT>(output->data);
                    const A_long srcLinePitch = rowBytes / static_cast<A_long>(PF_Pixel_BGRX_8u_size);
                    const A_long dstLinePitch = output->rowbytes / static_cast<A_long>(PF_Pixel_BGRX_8u_size);

                    // convert from Adobe buffer format to planar 
                    ingest_to_planar_f32 (localSrc, sizeX, sizeY, srcLinePitch, fmt_BGRX_4444_8u, memHndl.Src_R, memHndl.Src_G, memHndl.Src_B, memHndl.padW);

                    // perform film simulations
                    Algorithm_Main (memHndl, sizeX, sizeY, algoControls);

                    // back convert from planar format to Adobe buffer format
                    egress_from_planar_f32 (memHndl.Dst_R, memHndl.Dst_G, memHndl.Dst_B, memHndl.padW, sizeX, sizeY, localDst, dstLinePitch, fmt_BGRX_4444_8u, localSrc, sizeX, srcLinePitch);
                }
                break;

                case PrPixelFormat_BGRX_4444_16u:
                {
                    const PF_Pixel_BGRX_16u* RESTRICT localSrc = reinterpret_cast<const PF_Pixel_BGRX_16u* RESTRICT>(pfLayer->data);
                          PF_Pixel_BGRX_16u* RESTRICT localDst = reinterpret_cast<      PF_Pixel_BGRX_16u* RESTRICT>(output->data);
                    const A_long srcLinePitch = rowBytes / static_cast<A_long>(PF_Pixel_BGRX_16u_size);
                    const A_long dstLinePitch = output->rowbytes / static_cast<A_long>(PF_Pixel_BGRX_16u_size);

                    // convert from Adobe buffer format to planar 
                    ingest_to_planar_f32(localSrc, sizeX, sizeY, srcLinePitch, fmt_BGRX_4444_16u, memHndl.Src_R, memHndl.Src_G, memHndl.Src_B, memHndl.padW);

                    // perform film simulations
                    Algorithm_Main(memHndl, sizeX, sizeY, algoControls);

                    // back convert from planar format to Adobe buffer format
                    egress_from_planar_f32(memHndl.Dst_R, memHndl.Dst_G, memHndl.Dst_B, memHndl.padW, sizeX, sizeY, localDst, dstLinePitch, fmt_BGRX_4444_16u, localSrc, sizeX, srcLinePitch);
                }
                break;

                case PrPixelFormat_BGRX_4444_32f:
                {
                    const PF_Pixel_BGRX_32f* RESTRICT localSrc = reinterpret_cast<const PF_Pixel_BGRX_32f* RESTRICT>(pfLayer->data);
                          PF_Pixel_BGRX_32f* RESTRICT localDst = reinterpret_cast<      PF_Pixel_BGRX_32f* RESTRICT>(output->data);
                    const A_long srcLinePitch = rowBytes / static_cast<A_long>(PF_Pixel_BGRX_32f_size);
                    const A_long dstLinePitch = output->rowbytes / static_cast<A_long>(PF_Pixel_BGRX_32f_size);

                    // convert from Adobe buffer format to planar 
                    ingest_to_planar_f32 (localSrc, sizeX, sizeY, srcLinePitch, fmt_BGRX_4444_32f, memHndl.Src_R, memHndl.Src_G, memHndl.Src_B, memHndl.padW);

                    // perform film simulations
                    Algorithm_Main (memHndl, sizeX, sizeY, algoControls);

                    // back convert from planar format to Adobe buffer format
                    egress_from_planar_f32 (memHndl.Dst_R, memHndl.Dst_G, memHndl.Dst_B, memHndl.padW, sizeX, sizeY, localDst, dstLinePitch, fmt_BGRX_4444_32f, localSrc, sizeX, srcLinePitch);
                }
                break;

                case PrPixelFormat_BGRX_4444_32f_Linear:
                {
                    const PF_Pixel_BGRX_32f* RESTRICT localSrc = reinterpret_cast<const PF_Pixel_BGRX_32f* RESTRICT>(pfLayer->data);
                          PF_Pixel_BGRX_32f* RESTRICT localDst = reinterpret_cast<      PF_Pixel_BGRX_32f* RESTRICT>(output->data);
                    const A_long srcLinePitch = rowBytes / static_cast<A_long>(PF_Pixel_BGRX_32f_size);
                    const A_long dstLinePitch = output->rowbytes / static_cast<A_long>(PF_Pixel_BGRX_32f_size);

                    // convert from Adobe buffer format to planar 
                    ingest_to_planar_f32 (localSrc, sizeX, sizeY, srcLinePitch, fmt_BGRX_4444_32f_Linear, memHndl.Src_R, memHndl.Src_G, memHndl.Src_B, memHndl.padW);

                    // perform film simulations
                    Algorithm_Main (memHndl, sizeX, sizeY, algoControls);

                    // back convert from planar format to Adobe buffer format
                    egress_from_planar_f32 (memHndl.Dst_R, memHndl.Dst_G, memHndl.Dst_B, memHndl.padW, sizeX, sizeY, localDst, dstLinePitch, fmt_BGRX_4444_32f_Linear, localSrc, sizeX, srcLinePitch);
                }
                break;

                case PrPixelFormat_VUYA_4444_8u_709:
                {
                    const PF_Pixel_VUYA_8u* RESTRICT localSrc = reinterpret_cast<const PF_Pixel_VUYA_8u* RESTRICT>(pfLayer->data);
                          PF_Pixel_VUYA_8u* RESTRICT localDst = reinterpret_cast<      PF_Pixel_VUYA_8u* RESTRICT>(output->data);
                    const A_long srcLinePitch = rowBytes / static_cast<A_long>(PF_Pixel_VUYA_8u_size);
                    const A_long dstLinePitch = output->rowbytes / static_cast<A_long>(PF_Pixel_VUYA_8u_size);

                    // convert from Adobe buffer format to planar 
                    ingest_to_planar_f32 (localSrc, sizeX, sizeY, srcLinePitch, fmt_VUYA_4444_8u_709, memHndl.Src_R, memHndl.Src_G, memHndl.Src_B, memHndl.padW);

                    // perform film simulations
                    Algorithm_Main (memHndl, sizeX, sizeY, algoControls);

                    // back convert from planar format to Adobe buffer format
                    egress_from_planar_f32 (memHndl.Dst_R, memHndl.Dst_G, memHndl.Dst_B, memHndl.padW, sizeX, sizeY, localDst, dstLinePitch, fmt_VUYA_4444_8u_709, localSrc, sizeX, srcLinePitch);
                }
                break;

                case PrPixelFormat_VUYA_4444_8u:
                {
                    const PF_Pixel_VUYA_8u* RESTRICT localSrc = reinterpret_cast<const PF_Pixel_VUYA_8u* RESTRICT>(pfLayer->data);
                          PF_Pixel_VUYA_8u* RESTRICT localDst = reinterpret_cast<      PF_Pixel_VUYA_8u* RESTRICT>(output->data);
                    const A_long srcLinePitch = rowBytes / static_cast<A_long>(PF_Pixel_VUYA_8u_size);
                    const A_long dstLinePitch = output->rowbytes / static_cast<A_long>(PF_Pixel_VUYA_8u_size);

                    // convert from Adobe buffer format to planar 
                    ingest_to_planar_f32 (localSrc, sizeX, sizeY, srcLinePitch, fmt_VUYA_4444_8u, memHndl.Src_R, memHndl.Src_G, memHndl.Src_B, memHndl.padW);

                    // perform film simulations
                    Algorithm_Main (memHndl, sizeX, sizeY, algoControls);

                    // back convert from planar format to Adobe buffer format
                    egress_from_planar_f32 (memHndl.Dst_R, memHndl.Dst_G, memHndl.Dst_B, memHndl.padW, sizeX, sizeY, localDst, dstLinePitch, fmt_VUYA_4444_8u, localSrc, sizeX, srcLinePitch);
                }
                break;

                case PrPixelFormat_VUYA_4444_32f_709:
                {
                    const PF_Pixel_VUYA_32f* RESTRICT localSrc = reinterpret_cast<const PF_Pixel_VUYA_32f* RESTRICT>(pfLayer->data);
                          PF_Pixel_VUYA_32f* RESTRICT localDst = reinterpret_cast<      PF_Pixel_VUYA_32f* RESTRICT>(output->data);
                    const A_long srcLinePitch = rowBytes / static_cast<A_long>(PF_Pixel_VUYA_32f_size);
                    const A_long dstLinePitch = output->rowbytes / static_cast<A_long>(PF_Pixel_VUYA_32f_size);

                    // convert from Adobe buffer format to planar 
                    ingest_to_planar_f32 (localSrc, sizeX, sizeY, srcLinePitch, fmt_VUYA_4444_32f_709, memHndl.Src_R, memHndl.Src_G, memHndl.Src_B, memHndl.padW);

                    // perform film simulations
                    Algorithm_Main (memHndl, sizeX, sizeY, algoControls);

                    // back convert from planar format to Adobe buffer format
                    egress_from_planar_f32 (memHndl.Dst_R, memHndl.Dst_G, memHndl.Dst_B, memHndl.padW, sizeX, sizeY, localDst, dstLinePitch, fmt_VUYA_4444_32f_709, localSrc, sizeX, srcLinePitch);
                }
                break;

                case PrPixelFormat_VUYA_4444_32f:
                {
                    const PF_Pixel_VUYA_32f* RESTRICT localSrc = reinterpret_cast<const PF_Pixel_VUYA_32f* RESTRICT>(pfLayer->data);
                          PF_Pixel_VUYA_32f* RESTRICT localDst = reinterpret_cast<      PF_Pixel_VUYA_32f* RESTRICT>(output->data);
                    const A_long srcLinePitch = rowBytes / static_cast<A_long>(PF_Pixel_VUYA_32f_size);
                    const A_long dstLinePitch = output->rowbytes / static_cast<A_long>(PF_Pixel_VUYA_32f_size);

                    // convert from Adobe buffer format to planar 
                    ingest_to_planar_f32 (localSrc, sizeX, sizeY, srcLinePitch, fmt_VUYA_4444_32f, memHndl.Src_R, memHndl.Src_G, memHndl.Src_B, memHndl.padW);

                    // perform film simulations
                    Algorithm_Main (memHndl, sizeX, sizeY, algoControls);

                    // back convert from planar format to Adobe buffer format
                    egress_from_planar_f32 (memHndl.Dst_R, memHndl.Dst_G, memHndl.Dst_B, memHndl.padW, sizeX, sizeY, localDst, dstLinePitch, fmt_VUYA_4444_32f, localSrc, sizeX, srcLinePitch);
                }
                break;

                case PrPixelFormat_VUYP_4444_8u_709:
                {
                    const PF_Pixel_VUYP_8u* RESTRICT localSrc = reinterpret_cast<const PF_Pixel_VUYP_8u* RESTRICT>(pfLayer->data);
                          PF_Pixel_VUYP_8u* RESTRICT localDst = reinterpret_cast<      PF_Pixel_VUYP_8u* RESTRICT>(output->data);
                    const A_long srcLinePitch = rowBytes / static_cast<A_long>(PF_Pixel_VUYP_8u_size);
                    const A_long dstLinePitch = output->rowbytes / static_cast<A_long>(PF_Pixel_VUYP_8u_size);

                    // convert from Adobe buffer format to planar 
                    ingest_to_planar_f32 (localSrc, sizeX, sizeY, srcLinePitch, fmt_VUYP_4444_8u_709, memHndl.Src_R, memHndl.Src_G, memHndl.Src_B, memHndl.padW);

                    // perform film simulations
                    Algorithm_Main (memHndl, sizeX, sizeY, algoControls);

                    // back convert from planar format to Adobe buffer format
                    egress_from_planar_f32 (memHndl.Dst_R, memHndl.Dst_G, memHndl.Dst_B, memHndl.padW, sizeX, sizeY, localDst, dstLinePitch, fmt_VUYP_4444_8u_709, localSrc, sizeX, srcLinePitch);
                }
                break;

                case PrPixelFormat_VUYP_4444_8u:
                {
                    const PF_Pixel_VUYP_8u* RESTRICT localSrc = reinterpret_cast<const PF_Pixel_VUYP_8u* RESTRICT>(pfLayer->data);
                          PF_Pixel_VUYP_8u* RESTRICT localDst = reinterpret_cast<      PF_Pixel_VUYP_8u* RESTRICT>(output->data);
                    const A_long srcLinePitch = rowBytes / static_cast<A_long>(PF_Pixel_VUYP_8u_size);
                    const A_long dstLinePitch = output->rowbytes / static_cast<A_long>(PF_Pixel_VUYP_8u_size);

                    // convert from Adobe buffer format to planar 
                    ingest_to_planar_f32 (localSrc, sizeX, sizeY, srcLinePitch, fmt_VUYP_4444_8u, memHndl.Src_R, memHndl.Src_G, memHndl.Src_B, memHndl.padW);

                    // perform film simulations
                    Algorithm_Main (memHndl, sizeX, sizeY, algoControls);

                    // back convert from planar format to Adobe buffer format
                    egress_from_planar_f32 (memHndl.Dst_R, memHndl.Dst_G, memHndl.Dst_B, memHndl.padW, sizeX, sizeY, localDst, dstLinePitch, fmt_VUYP_4444_8u, localSrc, sizeX, srcLinePitch);
                }
                break;

                case PrPixelFormat_VUYP_4444_32f_709:
                {
                    const PF_Pixel_VUYP_32f* RESTRICT localSrc = reinterpret_cast<const PF_Pixel_VUYP_32f* RESTRICT>(pfLayer->data);
                          PF_Pixel_VUYP_32f* RESTRICT localDst = reinterpret_cast<      PF_Pixel_VUYP_32f* RESTRICT>(output->data);
                    const A_long srcLinePitch = rowBytes / static_cast<A_long>(PF_Pixel_VUYP_32f_size);
                    const A_long dstLinePitch = output->rowbytes / static_cast<A_long>(PF_Pixel_VUYP_32f_size);

                    // convert from Adobe buffer format to planar 
                    ingest_to_planar_f32 (localSrc, sizeX, sizeY, srcLinePitch, fmt_VUYP_4444_32f_709, memHndl.Src_R, memHndl.Src_G, memHndl.Src_B, memHndl.padW);

                    // perform film simulations
                    Algorithm_Main (memHndl, sizeX, sizeY, algoControls);

                    // back convert from planar format to Adobe buffer format
                    egress_from_planar_f32 (memHndl.Dst_R, memHndl.Dst_G, memHndl.Dst_B, memHndl.padW, sizeX, sizeY, localDst, dstLinePitch, fmt_VUYP_4444_32f_709, localSrc, sizeX, srcLinePitch);
                }
                break;

                case PrPixelFormat_VUYP_4444_32f:
                {
                    const PF_Pixel_VUYP_32f* RESTRICT localSrc = reinterpret_cast<const PF_Pixel_VUYP_32f* RESTRICT>(pfLayer->data);
                          PF_Pixel_VUYP_32f* RESTRICT localDst = reinterpret_cast<      PF_Pixel_VUYP_32f* RESTRICT>(output->data);
                    const A_long srcLinePitch = rowBytes / static_cast<A_long>(PF_Pixel_VUYP_32f_size);
                    const A_long dstLinePitch = output->rowbytes / static_cast<A_long>(PF_Pixel_VUYP_32f_size);

                    // convert from Adobe buffer format to planar 
                    ingest_to_planar_f32 (localSrc, sizeX, sizeY, srcLinePitch, fmt_VUYP_4444_32f, memHndl.Src_R, memHndl.Src_G, memHndl.Src_B, memHndl.padW);

                    // perform film simulations
                    Algorithm_Main (memHndl, sizeX, sizeY, algoControls);

                    // back convert from planar format to Adobe buffer format
                    egress_from_planar_f32 (memHndl.Dst_R, memHndl.Dst_G, memHndl.Dst_B, memHndl.padW, sizeX, sizeY, localDst, dstLinePitch, fmt_VUYP_4444_32f, localSrc, sizeX, srcLinePitch);
                }
                break;

                case PrPixelFormat_VUYX_4444_8u_709:
                {
                    const PF_Pixel_VUYX_8u* RESTRICT localSrc = reinterpret_cast<const PF_Pixel_VUYX_8u* RESTRICT>(pfLayer->data);
                          PF_Pixel_VUYX_8u* RESTRICT localDst = reinterpret_cast<      PF_Pixel_VUYX_8u* RESTRICT>(output->data);
                    const A_long srcLinePitch = rowBytes / static_cast<A_long>(PF_Pixel_VUYX_8u_size);
                    const A_long dstLinePitch = output->rowbytes / static_cast<A_long>(PF_Pixel_VUYX_8u_size);

                    // convert from Adobe buffer format to planar 
                    ingest_to_planar_f32 (localSrc, sizeX, sizeY, srcLinePitch, fmt_VUYX_4444_8u_709, memHndl.Src_R, memHndl.Src_G, memHndl.Src_B, memHndl.padW);

                    // perform film simulations
                    Algorithm_Main (memHndl, sizeX, sizeY, algoControls);

                    // back convert from planar format to Adobe buffer format
                    egress_from_planar_f32 (memHndl.Dst_R, memHndl.Dst_G, memHndl.Dst_B, memHndl.padW, sizeX, sizeY, localDst, dstLinePitch, fmt_VUYX_4444_8u_709, localSrc, sizeX, srcLinePitch);
                }
                break;

                case PrPixelFormat_VUYX_4444_8u:
                {
                    const PF_Pixel_VUYX_8u* RESTRICT localSrc = reinterpret_cast<const PF_Pixel_VUYX_8u* RESTRICT>(pfLayer->data);
                          PF_Pixel_VUYX_8u* RESTRICT localDst = reinterpret_cast<      PF_Pixel_VUYX_8u* RESTRICT>(output->data);
                    const A_long srcLinePitch = rowBytes / static_cast<A_long>(PF_Pixel_VUYX_8u_size);
                    const A_long dstLinePitch = output->rowbytes / static_cast<A_long>(PF_Pixel_VUYX_8u_size);

                    // convert from Adobe buffer format to planar 
                    ingest_to_planar_f32 (localSrc, sizeX, sizeY, srcLinePitch, fmt_VUYX_4444_8u, memHndl.Src_R, memHndl.Src_G, memHndl.Src_B, memHndl.padW);

                    // perform film simulations
                    Algorithm_Main (memHndl, sizeX, sizeY, algoControls);

                    // back convert from planar format to Adobe buffer format
                    egress_from_planar_f32 (memHndl.Dst_R, memHndl.Dst_G, memHndl.Dst_B, memHndl.padW, sizeX, sizeY, localDst, dstLinePitch, fmt_VUYX_4444_8u, localSrc, sizeX, srcLinePitch);
                }
                break;

                case PrPixelFormat_VUYX_4444_32f_709:
                {
                    const PF_Pixel_VUYX_32f* RESTRICT localSrc = reinterpret_cast<const PF_Pixel_VUYX_32f* RESTRICT>(pfLayer->data);
                          PF_Pixel_VUYX_32f* RESTRICT localDst = reinterpret_cast<      PF_Pixel_VUYX_32f* RESTRICT>(output->data);
                    const A_long srcLinePitch = rowBytes / static_cast<A_long>(PF_Pixel_VUYX_32f_size);
                    const A_long dstLinePitch = output->rowbytes / static_cast<A_long>(PF_Pixel_VUYX_32f_size);

                    // convert from Adobe buffer format to planar 
                    ingest_to_planar_f32 (localSrc, sizeX, sizeY, srcLinePitch, fmt_VUYX_4444_32f_709, memHndl.Src_R, memHndl.Src_G, memHndl.Src_B, memHndl.padW);

                    // perform film simulations
                    Algorithm_Main (memHndl, sizeX, sizeY, algoControls);

                    // back convert from planar format to Adobe buffer format
                    egress_from_planar_f32 (memHndl.Dst_R, memHndl.Dst_G, memHndl.Dst_B, memHndl.padW, sizeX, sizeY, localDst, dstLinePitch, fmt_VUYX_4444_32f_709, localSrc, sizeX, srcLinePitch);
                }
                break;

                case PrPixelFormat_VUYX_4444_32f:
                {
                    const PF_Pixel_VUYX_32f* RESTRICT localSrc = reinterpret_cast<const PF_Pixel_VUYX_32f* RESTRICT>(pfLayer->data);
                          PF_Pixel_VUYX_32f* RESTRICT localDst = reinterpret_cast<      PF_Pixel_VUYX_32f* RESTRICT>(output->data);
                    const A_long srcLinePitch = rowBytes / static_cast<A_long>(PF_Pixel_VUYX_32f_size);
                    const A_long dstLinePitch = output->rowbytes / static_cast<A_long>(PF_Pixel_VUYX_32f_size);

                    // convert from Adobe buffer format to planar 
                    ingest_to_planar_f32 (localSrc, sizeX, sizeY, srcLinePitch, fmt_VUYX_4444_32f, memHndl.Src_R, memHndl.Src_G, memHndl.Src_B, memHndl.padW);

                    // perform film simulations
                    Algorithm_Main (memHndl, sizeX, sizeY, algoControls);

                    // back convert from planar format to Adobe buffer format
                    egress_from_planar_f32 (memHndl.Dst_R, memHndl.Dst_G, memHndl.Dst_B, memHndl.padW, sizeX, sizeY, localDst, dstLinePitch, fmt_VUYX_4444_32f, localSrc, sizeX, srcLinePitch);
                }
                break;

                case PrPixelFormat_ARGB_4444_8u:
                {
                    const PF_Pixel_ARGB_8u* RESTRICT localSrc = reinterpret_cast<const PF_Pixel_ARGB_8u* RESTRICT>(pfLayer->data);
                          PF_Pixel_ARGB_8u* RESTRICT localDst = reinterpret_cast<      PF_Pixel_ARGB_8u* RESTRICT>(output->data);
                    const A_long srcLinePitch = rowBytes / static_cast<A_long>(PF_Pixel_ARGB_8u_size);
                    const A_long dstLinePitch = output->rowbytes / static_cast<A_long>(PF_Pixel_ARGB_8u_size);

                    // convert from Adobe buffer format to planar 
                    ingest_to_planar_f32 (localSrc, sizeX, sizeY, srcLinePitch, fmt_ARGB_4444_8u, memHndl.Src_R, memHndl.Src_G, memHndl.Src_B, memHndl.padW);

                    // perform film simulations
                    Algorithm_Main (memHndl, sizeX, sizeY, algoControls);

                    // back convert from planar format to Adobe buffer format
                    egress_from_planar_f32 (memHndl.Dst_R, memHndl.Dst_G, memHndl.Dst_B, memHndl.padW, sizeX, sizeY, localDst, dstLinePitch, fmt_ARGB_4444_8u, localSrc, sizeX, srcLinePitch);
                }
                break;

                case PrPixelFormat_PRGB_4444_8u:
                {
                    const PF_Pixel_PRGB_8u* RESTRICT localSrc = reinterpret_cast<const PF_Pixel_PRGB_8u* RESTRICT>(pfLayer->data);
                          PF_Pixel_PRGB_8u* RESTRICT localDst = reinterpret_cast<      PF_Pixel_PRGB_8u* RESTRICT>(output->data);
                    const A_long srcLinePitch = rowBytes / static_cast<A_long>(PF_Pixel_PRGB_8u_size);
                    const A_long dstLinePitch = output->rowbytes / static_cast<A_long>(PF_Pixel_PRGB_8u_size);

                    // convert from Adobe buffer format to planar 
                    ingest_to_planar_f32 (localSrc, sizeX, sizeY, srcLinePitch, fmt_PRGB_4444_8u, memHndl.Src_R, memHndl.Src_G, memHndl.Src_B, memHndl.padW);

                    // perform film simulations
                    Algorithm_Main (memHndl, sizeX, sizeY, algoControls);

                    // back convert from planar format to Adobe buffer format
                    egress_from_planar_f32 (memHndl.Dst_R, memHndl.Dst_G, memHndl.Dst_B, memHndl.padW, sizeX, sizeY, localDst, dstLinePitch, fmt_PRGB_4444_8u, localSrc, sizeX, srcLinePitch);
                }
                break;

                case PrPixelFormat_XRGB_4444_8u:
                {
                    const PF_Pixel_XRGB_8u* RESTRICT localSrc = reinterpret_cast<const PF_Pixel_XRGB_8u* RESTRICT>(pfLayer->data);
                          PF_Pixel_XRGB_8u* RESTRICT localDst = reinterpret_cast<      PF_Pixel_XRGB_8u* RESTRICT>(output->data);
                    const A_long srcLinePitch = rowBytes / static_cast<A_long>(PF_Pixel_XRGB_8u_size);
                    const A_long dstLinePitch = output->rowbytes / static_cast<A_long>(PF_Pixel_XRGB_8u_size);

                    // convert from Adobe buffer format to planar 
                    ingest_to_planar_f32 (localSrc, sizeX, sizeY, srcLinePitch, fmt_XRGB_4444_8u, memHndl.Src_R, memHndl.Src_G, memHndl.Src_B, memHndl.padW);

                    // perform film simulations
                    Algorithm_Main (memHndl, sizeX, sizeY, algoControls);

                    // back convert from planar format to Adobe buffer format
                    egress_from_planar_f32 (memHndl.Dst_R, memHndl.Dst_G, memHndl.Dst_B, memHndl.padW, sizeX, sizeY, localDst, dstLinePitch, fmt_XRGB_4444_8u, localSrc, sizeX, srcLinePitch);
                }
                break;

                case PrPixelFormat_ARGB_4444_16u:
                {
                    const PF_Pixel_ARGB_16u* RESTRICT localSrc = reinterpret_cast<const PF_Pixel_ARGB_16u* RESTRICT>(pfLayer->data);
                          PF_Pixel_ARGB_16u* RESTRICT localDst = reinterpret_cast<      PF_Pixel_ARGB_16u* RESTRICT>(output->data);
                    const A_long srcLinePitch = rowBytes / static_cast<A_long>(PF_Pixel_ARGB_16u_size);
                    const A_long dstLinePitch = output->rowbytes / static_cast<A_long>(PF_Pixel_ARGB_16u_size);

                    // convert from Adobe buffer format to planar 
                    ingest_to_planar_f32 (localSrc, sizeX, sizeY, srcLinePitch, fmt_ARGB_4444_16u, memHndl.Src_R, memHndl.Src_G, memHndl.Src_B, memHndl.padW);

                    // perform film simulations
                    Algorithm_Main (memHndl, sizeX, sizeY, algoControls);

                    // back convert from planar format to Adobe buffer format
                    egress_from_planar_f32 (memHndl.Dst_R, memHndl.Dst_G, memHndl.Dst_B, memHndl.padW, sizeX, sizeY, localDst, dstLinePitch, fmt_ARGB_4444_16u, localSrc, sizeX, srcLinePitch);
                }
                break;

                case PrPixelFormat_PRGB_4444_16u:
                {
                    const PF_Pixel_PRGB_16u* RESTRICT localSrc = reinterpret_cast<const PF_Pixel_PRGB_16u* RESTRICT>(pfLayer->data);
                          PF_Pixel_PRGB_16u* RESTRICT localDst = reinterpret_cast<      PF_Pixel_PRGB_16u* RESTRICT>(output->data);
                    const A_long srcLinePitch = rowBytes / static_cast<A_long>(PF_Pixel_PRGB_16u_size);
                    const A_long dstLinePitch = output->rowbytes / static_cast<A_long>(PF_Pixel_PRGB_16u_size);

                    // convert from Adobe buffer format to planar 
                    ingest_to_planar_f32 (localSrc, sizeX, sizeY, srcLinePitch, fmt_PRGB_4444_16u, memHndl.Src_R, memHndl.Src_G, memHndl.Src_B, memHndl.padW);

                    // perform film simulations
                    Algorithm_Main (memHndl, sizeX, sizeY, algoControls);

                    // back convert from planar format to Adobe buffer format
                    egress_from_planar_f32 (memHndl.Dst_R, memHndl.Dst_G, memHndl.Dst_B, memHndl.padW, sizeX, sizeY, localDst, dstLinePitch, fmt_PRGB_4444_16u, localSrc, sizeX, srcLinePitch);
                }
                break;

                case PrPixelFormat_XRGB_4444_16u:
                {
                    const PF_Pixel_XRGB_16u* RESTRICT localSrc = reinterpret_cast<const PF_Pixel_XRGB_16u* RESTRICT>(pfLayer->data);
                          PF_Pixel_XRGB_16u* RESTRICT localDst = reinterpret_cast<      PF_Pixel_XRGB_16u* RESTRICT>(output->data);
                    const A_long srcLinePitch = rowBytes / static_cast<A_long>(PF_Pixel_XRGB_16u_size);
                    const A_long dstLinePitch = output->rowbytes / static_cast<A_long>(PF_Pixel_XRGB_16u_size);

                    // convert from Adobe buffer format to planar 
                    ingest_to_planar_f32 (localSrc, sizeX, sizeY, srcLinePitch, fmt_XRGB_4444_16u, memHndl.Src_R, memHndl.Src_G, memHndl.Src_B, memHndl.padW);

                    // perform film simulations
                    Algorithm_Main (memHndl, sizeX, sizeY, algoControls);

                    // back convert from planar format to Adobe buffer format
                    egress_from_planar_f32 (memHndl.Dst_R, memHndl.Dst_G, memHndl.Dst_B, memHndl.padW, sizeX, sizeY, localDst, dstLinePitch, fmt_XRGB_4444_16u, localSrc, sizeX, srcLinePitch);
                }
                break;

                case PrPixelFormat_ARGB_4444_32f:
                {
                    const PF_Pixel_ARGB_32f* RESTRICT localSrc = reinterpret_cast<const PF_Pixel_ARGB_32f* RESTRICT>(pfLayer->data);
                          PF_Pixel_ARGB_32f* RESTRICT localDst = reinterpret_cast<      PF_Pixel_ARGB_32f* RESTRICT>(output->data);
                    const A_long srcLinePitch = rowBytes / static_cast<A_long>(PF_Pixel_ARGB_32f_size);
                    const A_long dstLinePitch = output->rowbytes / static_cast<A_long>(PF_Pixel_ARGB_32f_size);

                    // convert from Adobe buffer format to planar 
                    ingest_to_planar_f32 (localSrc, sizeX, sizeY, srcLinePitch, fmt_ARGB_4444_32f, memHndl.Src_R, memHndl.Src_G, memHndl.Src_B, memHndl.padW);

                    // perform film simulations
                    Algorithm_Main (memHndl, sizeX, sizeY, algoControls);

                    // back convert from planar format to Adobe buffer format
                    egress_from_planar_f32 (memHndl.Dst_R, memHndl.Dst_G, memHndl.Dst_B, memHndl.padW, sizeX, sizeY, localDst, dstLinePitch, fmt_ARGB_4444_32f, localSrc, sizeX, srcLinePitch);
                }
                break;

                case PrPixelFormat_PRGB_4444_32f:
                {
                    const PF_Pixel_PRGB_32f* RESTRICT localSrc = reinterpret_cast<const PF_Pixel_PRGB_32f* RESTRICT>(pfLayer->data);
                          PF_Pixel_PRGB_32f* RESTRICT localDst = reinterpret_cast<      PF_Pixel_PRGB_32f* RESTRICT>(output->data);
                    const A_long srcLinePitch = rowBytes / static_cast<A_long>(PF_Pixel_PRGB_32f_size);
                    const A_long dstLinePitch = output->rowbytes / static_cast<A_long>(PF_Pixel_PRGB_32f_size);

                    // convert from Adobe buffer format to planar 
                    ingest_to_planar_f32 (localSrc, sizeX, sizeY, srcLinePitch, fmt_PRGB_4444_32f, memHndl.Src_R, memHndl.Src_G, memHndl.Src_B, memHndl.padW);

                    // perform film simulations
                    Algorithm_Main (memHndl, sizeX, sizeY, algoControls);

                    // back convert from planar format to Adobe buffer format
                    egress_from_planar_f32 (memHndl.Dst_R, memHndl.Dst_G, memHndl.Dst_B, memHndl.padW, sizeX, sizeY, localDst, dstLinePitch, fmt_PRGB_4444_32f, localSrc, sizeX, srcLinePitch);
                }
                break;

                case PrPixelFormat_XRGB_4444_32f:
                {
                    const PF_Pixel_XRGB_32f* RESTRICT localSrc = reinterpret_cast<const PF_Pixel_XRGB_32f* RESTRICT>(pfLayer->data);
                          PF_Pixel_XRGB_32f* RESTRICT localDst = reinterpret_cast<      PF_Pixel_XRGB_32f* RESTRICT>(output->data);
                    const A_long srcLinePitch = rowBytes / static_cast<A_long>(PF_Pixel_XRGB_32f_size);
                    const A_long dstLinePitch = output->rowbytes / static_cast<A_long>(PF_Pixel_XRGB_32f_size);

                    // convert from Adobe buffer format to planar 
                    ingest_to_planar_f32 (localSrc, sizeX, sizeY, srcLinePitch, fmt_XRGB_4444_32f, memHndl.Src_R, memHndl.Src_G, memHndl.Src_B, memHndl.padW);

                    // perform film simulations
                    Algorithm_Main (memHndl, sizeX, sizeY, algoControls);

                    // back convert from planar format to Adobe buffer format
                    egress_from_planar_f32 (memHndl.Dst_R, memHndl.Dst_G, memHndl.Dst_B, memHndl.padW, sizeX, sizeY, localDst, dstLinePitch, fmt_XRGB_4444_32f, localSrc, sizeX, srcLinePitch);
                }
                break;

                case PrPixelFormat_ARGB_4444_32f_Linear:
                {
                    const PF_Pixel_ARGB_32f* RESTRICT localSrc = reinterpret_cast<const PF_Pixel_ARGB_32f* RESTRICT>(pfLayer->data);
                          PF_Pixel_ARGB_32f* RESTRICT localDst = reinterpret_cast<      PF_Pixel_ARGB_32f* RESTRICT>(output->data);
                    const A_long srcLinePitch = rowBytes / static_cast<A_long>(PF_Pixel_ARGB_32f_size);
                    const A_long dstLinePitch = output->rowbytes / static_cast<A_long>(PF_Pixel_ARGB_32f_size);

                    // convert from Adobe buffer format to planar 
                    ingest_to_planar_f32 (localSrc, sizeX, sizeY, srcLinePitch, fmt_ARGB_4444_32f_Linear, memHndl.Src_R, memHndl.Src_G, memHndl.Src_B, memHndl.padW);

                    // perform film simulations
                    Algorithm_Main (memHndl, sizeX, sizeY, algoControls);

                    // back convert from planar format to Adobe buffer format
                    egress_from_planar_f32 (memHndl.Dst_R, memHndl.Dst_G, memHndl.Dst_B, memHndl.padW, sizeX, sizeY, localDst, dstLinePitch, fmt_ARGB_4444_32f_Linear, localSrc, sizeX, srcLinePitch);
                }
                break;

                case PrPixelFormat_PRGB_4444_32f_Linear:
                {
                    const PF_Pixel_PRGB_32f* RESTRICT localSrc = reinterpret_cast<const PF_Pixel_PRGB_32f* RESTRICT>(pfLayer->data);
                          PF_Pixel_PRGB_32f* RESTRICT localDst = reinterpret_cast<      PF_Pixel_PRGB_32f* RESTRICT>(output->data);
                    const A_long srcLinePitch = rowBytes / static_cast<A_long>(PF_Pixel_PRGB_32f_size);
                    const A_long dstLinePitch = output->rowbytes / static_cast<A_long>(PF_Pixel_PRGB_32f_size);

                    // convert from Adobe buffer format to planar 
                    ingest_to_planar_f32(localSrc, sizeX, sizeY, srcLinePitch, fmt_PRGB_4444_32f_Linear, memHndl.Src_R, memHndl.Src_G, memHndl.Src_B, memHndl.padW);

                    // perform film simulations
                    Algorithm_Main(memHndl, sizeX, sizeY, algoControls);

                    // back convert from planar format to Adobe buffer format
                    egress_from_planar_f32(memHndl.Dst_R, memHndl.Dst_G, memHndl.Dst_B, memHndl.padW, sizeX, sizeY, localDst, dstLinePitch, fmt_PRGB_4444_32f_Linear, localSrc, sizeX, srcLinePitch);
                }
                break;

                case PrPixelFormat_XRGB_4444_32f_Linear:
                {
                    const PF_Pixel_XRGB_32f* RESTRICT localSrc = reinterpret_cast<const PF_Pixel_XRGB_32f* RESTRICT>(pfLayer->data);
                          PF_Pixel_XRGB_32f* RESTRICT localDst = reinterpret_cast<      PF_Pixel_XRGB_32f* RESTRICT>(output->data);
                    const A_long srcLinePitch = rowBytes / static_cast<A_long>(PF_Pixel_XRGB_32f_size);
                    const A_long dstLinePitch = output->rowbytes / static_cast<A_long>(PF_Pixel_XRGB_32f_size);

                    // convert from Adobe buffer format to planar 
                    ingest_to_planar_f32(localSrc, sizeX, sizeY, srcLinePitch, fmt_XRGB_4444_32f_Linear, memHndl.Src_R, memHndl.Src_G, memHndl.Src_B, memHndl.padW);

                    // perform film simulations
                    Algorithm_Main(memHndl, sizeX, sizeY, algoControls);

                    // back convert from planar format to Adobe buffer format
                    egress_from_planar_f32(memHndl.Dst_R, memHndl.Dst_G, memHndl.Dst_B, memHndl.padW, sizeX, sizeY, localDst, dstLinePitch, fmt_XRGB_4444_32f_Linear, localSrc, sizeX, srcLinePitch);
                }
                break;

                case PrPixelFormat_RGB_444_10u:
                {
                    const PF_Pixel_RGB_10u* RESTRICT localSrc = reinterpret_cast<const PF_Pixel_RGB_10u* RESTRICT>(pfLayer->data);
                          PF_Pixel_RGB_10u* RESTRICT localDst = reinterpret_cast<      PF_Pixel_RGB_10u* RESTRICT>(output->data);
                    const A_long srcLinePitch = rowBytes / static_cast<A_long>(PF_Pixel_RGB_10u_size);
                    const A_long dstLinePitch = output->rowbytes / static_cast<A_long>(PF_Pixel_RGB_10u_size);

                    // convert from Adobe buffer format to planar 
                    ingest_to_planar_f32 (localSrc, sizeX, sizeY, srcLinePitch, fmt_RGB_444_10u, memHndl.Src_R, memHndl.Src_G, memHndl.Src_B, memHndl.padW);

                    // perform film simulations
                    Algorithm_Main (memHndl, sizeX, sizeY, algoControls);

                    // back convert from planar format to Adobe buffer format
                    egress_from_planar_f32 (memHndl.Dst_R, memHndl.Dst_G, memHndl.Dst_B, memHndl.padW, sizeX, sizeY, localDst, dstLinePitch, fmt_RGB_444_10u, localSrc, sizeX, srcLinePitch);
                }
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

    } // if (true == mem_handler_valid(memHndl)) -- memGuard frees the arena
    else
        err = PF_Err_OUT_OF_MEMORY;

	return err;
}
