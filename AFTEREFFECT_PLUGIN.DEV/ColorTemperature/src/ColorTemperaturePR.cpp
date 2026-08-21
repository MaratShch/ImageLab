#include "ColorTemperature.hpp"
#include "ColorTemperatureEnums.hpp"
#include "ColorTemperatureControlsPresets.hpp"
#include "CompileTimeUtils.hpp"
#include "CommonAuxPixFormat.hpp"
#include "PrSDKAESupport.h"
#include "ColorLocus.hpp"
#include "AlgoSuperPixel.hpp"
#include "LinearLut/LinearLut.hpp"
#include "AlgoPrFormatIngest.hpp"
#include "AlgoPrFormatEgress.hpp"


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
    const A_long rowBytes = pfLayer->rowbytes;

    static const auto& lut8  = LinLut_srgb_8bit_double ::LINEARIZE_LUT_SRGB_8BIT_F64;
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
            const auto& locusGate = getLocusGate(obs_CIE_1931_2deg == algoCtrl.observer);

            SuperPixel<double> super{};     // computed in double, max accuracy
            CctDuv<double> cct_duv{};       // computed in double, max accuracy

            switch (destinationPixelFormat)
            {
                case PrPixelFormat_BGRA_4444_8u:
                {
                    const PF_Pixel_BGRA_8u* RESTRICT localSrc = reinterpret_cast<const PF_Pixel_BGRA_8u* RESTRICT>(pfLayer->data);
                          PF_Pixel_BGRA_8u* RESTRICT localDst = reinterpret_cast<      PF_Pixel_BGRA_8u* RESTRICT>(output->data);
                    const A_long srcLinePitch = rowBytes / static_cast<A_long>(PF_Pixel_BGRA_8u_size);
                    const A_long dstLinePitch = srcLinePitch;

                    AlgoPrIngest::ingest_and_superpixel
                    (
                        localSrc, 
                        sizeX, 
                        sizeY, 
                        srcLinePitch, 
                        AlgoPrIngest::fmt_BGRA_4444_8u, 
                        lut8, lut16, lut10,
                        locusGate,
                        algoMemHandler.input_f32_interleaved, 
                        super, 
                        algoCtrl.confidenceMap
                    );

                    Algorithm_Main (getCctHndl(), super, algoMemHandler, sizeX, sizeY, algoCtrl, cct_duv);

                    AlgoPrIngest::egress_from_linear_f32
                    (
                        algoMemHandler.output_f32_interleaved,
                        sizeX,
                        sizeY,
                        localDst,
                        dstLinePitch,
                        AlgoPrIngest::fmt_BGRA_4444_8u,
                        lut8, lut16, lut10, 
                        localSrc, 
                        sizeX,
                        srcLinePitch
                    );
                }
                break;

                case PrPixelFormat_BGRA_4444_16u:
                {
                    const PF_Pixel_BGRA_16u* RESTRICT localSrc = reinterpret_cast<const PF_Pixel_BGRA_16u* RESTRICT>(pfLayer->data);
                          PF_Pixel_BGRA_16u* RESTRICT localDst = reinterpret_cast<      PF_Pixel_BGRA_16u* RESTRICT>(output->data);
                    const A_long srcLinePitch = rowBytes / static_cast<A_long>(PF_Pixel_BGRA_16u_size);
                    const A_long dstLinePitch = srcLinePitch;

                    AlgoPrIngest::ingest_and_superpixel
                    (
                        localSrc, 
                        sizeX, 
                        sizeY, 
                        srcLinePitch, 
                        AlgoPrIngest::fmt_BGRA_4444_16u, 
                        lut8, lut16, lut10,
                        locusGate,
                        algoMemHandler.input_f32_interleaved,
                        super, 
                        algoCtrl.confidenceMap
                    );

                    Algorithm_Main (getCctHndl(), super, algoMemHandler, sizeX, sizeY, algoCtrl, cct_duv);

                    AlgoPrIngest::egress_from_linear_f32
                    (
                        algoMemHandler.output_f32_interleaved,
                        sizeX,
                        sizeY,
                        localDst,
                        dstLinePitch,
                        AlgoPrIngest::fmt_BGRA_4444_16u,
                        lut8, lut16, lut10,
                        localSrc,
                        sizeX,
                        srcLinePitch
                    );
                }
                break;

                case PrPixelFormat_BGRA_4444_32f:
                {
                    const PF_Pixel_BGRA_32f* RESTRICT localSrc = reinterpret_cast<const PF_Pixel_BGRA_32f* RESTRICT>(pfLayer->data);
                          PF_Pixel_BGRA_32f* RESTRICT localDst = reinterpret_cast<      PF_Pixel_BGRA_32f* RESTRICT>(output->data);
                    const A_long srcLinePitch = rowBytes / static_cast<A_long>(PF_Pixel_BGRA_32f_size);
                    const A_long dstLinePitch = srcLinePitch;

                    AlgoPrIngest::ingest_and_superpixel
                    (
                        localSrc,
                        sizeX, 
                        sizeY, 
                        srcLinePitch, 
                        AlgoPrIngest::fmt_BGRA_4444_32f, 
                        lut8, lut16, lut10,
                        locusGate,
                        algoMemHandler.input_f32_interleaved,
                        super,
                        algoCtrl.confidenceMap
                    );

                    Algorithm_Main (getCctHndl(), super, algoMemHandler, sizeX, sizeY, algoCtrl, cct_duv);

                    AlgoPrIngest::egress_from_linear_f32
                    (
                        algoMemHandler.output_f32_interleaved,
                        sizeX,
                        sizeY,
                        localDst,
                        dstLinePitch,
                        AlgoPrIngest::fmt_BGRA_4444_32f,
                        lut8, lut16, lut10,
                        localSrc,
                        sizeX,
                        srcLinePitch
                    );
                }
                break;

                case PrPixelFormat_BGRA_4444_32f_Linear:
                {
                    const PF_Pixel_BGRA_32f* RESTRICT localSrc = reinterpret_cast<const PF_Pixel_BGRA_32f* RESTRICT>(pfLayer->data);
                          PF_Pixel_BGRA_32f* RESTRICT localDst = reinterpret_cast<      PF_Pixel_BGRA_32f* RESTRICT>(output->data);
                    const A_long srcLinePitch = rowBytes / static_cast<A_long>(PF_Pixel_BGRA_32f_size);
                    const A_long dstLinePitch = srcLinePitch;

                    AlgoPrIngest::ingest_and_superpixel
                    (
                        localSrc, 
                        sizeX, 
                        sizeY, 
                        srcLinePitch, 
                        AlgoPrIngest::fmt_BGRA_4444_32f_Linear, 
                        lut8, lut16, lut10,
                        locusGate,
                        algoMemHandler.input_f32_interleaved, 
                        super, 
                        algoCtrl.confidenceMap
                    );

                    Algorithm_Main (getCctHndl(), super, algoMemHandler, sizeX, sizeY, algoCtrl, cct_duv);

                    AlgoPrIngest::egress_from_linear_f32
                    (
                        algoMemHandler.output_f32_interleaved,
                        sizeX,
                        sizeY,
                        localDst,
                        dstLinePitch,
                        AlgoPrIngest::fmt_BGRA_4444_32f_Linear,
                        lut8, lut16, lut10,
                        localSrc,
                        sizeX,
                        srcLinePitch
                    );
                }
                break;

                case PrPixelFormat_BGRP_4444_8u:
                {
                    const PF_Pixel_BGRP_8u* RESTRICT localSrc = reinterpret_cast<const PF_Pixel_BGRP_8u* RESTRICT>(pfLayer->data);
                          PF_Pixel_BGRP_8u* RESTRICT localDst = reinterpret_cast<      PF_Pixel_BGRP_8u* RESTRICT>(output->data);
                    const A_long srcLinePitch = rowBytes / static_cast<A_long>(PF_Pixel_BGRP_8u_size);
                    const A_long dstLinePitch = srcLinePitch;

                    AlgoPrIngest::ingest_and_superpixel
                    (
                        localSrc, 
                        sizeX, 
                        sizeY, 
                        srcLinePitch, 
                        AlgoPrIngest::fmt_BGRP_4444_8u, 
                        lut8, lut16, lut10,
                        locusGate,
                        algoMemHandler.input_f32_interleaved,
                        super,
                        algoCtrl.confidenceMap
                    );

                    Algorithm_Main (getCctHndl(), super, algoMemHandler, sizeX, sizeY, algoCtrl, cct_duv);

                    AlgoPrIngest::egress_from_linear_f32
                    (
                        algoMemHandler.output_f32_interleaved,
                        sizeX,
                        sizeY,
                        localDst,
                        dstLinePitch,
                        AlgoPrIngest::fmt_BGRP_4444_8u,
                        lut8, lut16, lut10,
                        localSrc,
                        sizeX,
                        srcLinePitch
                    );
                }
                break;

                case PrPixelFormat_BGRP_4444_16u:
                {
                    const PF_Pixel_BGRP_16u* RESTRICT localSrc = reinterpret_cast<const PF_Pixel_BGRP_16u* RESTRICT>(pfLayer->data);
                          PF_Pixel_BGRP_16u* RESTRICT localDst = reinterpret_cast<      PF_Pixel_BGRP_16u* RESTRICT>(output->data);
                    const A_long srcLinePitch = rowBytes / static_cast<A_long>(PF_Pixel_BGRP_16u_size);
                    const A_long dstLinePitch = srcLinePitch;

                    AlgoPrIngest::ingest_and_superpixel
                    (
                        localSrc,
                        sizeX,
                        sizeY, 
                        srcLinePitch, 
                        AlgoPrIngest::fmt_BGRP_4444_16u, 
                        lut8, lut16, lut10,
                        locusGate,
                        algoMemHandler.input_f32_interleaved,
                        super,
                        algoCtrl.confidenceMap
                    );

                    Algorithm_Main (getCctHndl(), super, algoMemHandler, sizeX, sizeY, algoCtrl, cct_duv);

                    AlgoPrIngest::egress_from_linear_f32
                    (
                        algoMemHandler.output_f32_interleaved,
                        sizeX,
                        sizeY,
                        localDst,
                        dstLinePitch,
                        AlgoPrIngest::fmt_BGRP_4444_16u,
                        lut8, lut16, lut10,
                        localSrc,
                        sizeX,
                        srcLinePitch
                    );
                }
                break;

                case PrPixelFormat_BGRP_4444_32f:
                {
                    const PF_Pixel_BGRP_32f* RESTRICT localSrc = reinterpret_cast<const PF_Pixel_BGRP_32f* RESTRICT>(pfLayer->data);
                          PF_Pixel_BGRP_32f* RESTRICT localDst = reinterpret_cast<      PF_Pixel_BGRP_32f* RESTRICT>(output->data);
                    const A_long srcLinePitch = rowBytes / static_cast<A_long>(PF_Pixel_BGRP_32f_size);
                    const A_long dstLinePitch = srcLinePitch;

                    AlgoPrIngest::ingest_and_superpixel
                    (
                        localSrc,
                        sizeX,
                        sizeY,
                        srcLinePitch,
                        AlgoPrIngest::fmt_BGRP_4444_32f,
                        lut8, lut16, lut10,
                        locusGate,
                        algoMemHandler.input_f32_interleaved,
                        super,
                        algoCtrl.confidenceMap
                    );

                    Algorithm_Main (getCctHndl(), super, algoMemHandler, sizeX, sizeY, algoCtrl, cct_duv);

                    AlgoPrIngest::egress_from_linear_f32
                    (
                        algoMemHandler.output_f32_interleaved,
                        sizeX,
                        sizeY,
                        localDst,
                        dstLinePitch,
                        AlgoPrIngest::fmt_BGRP_4444_32f,
                        lut8, lut16, lut10,
                        localSrc,
                        sizeX,
                        srcLinePitch
                    );
                }
                break;

                case PrPixelFormat_BGRP_4444_32f_Linear:
                {
                    const PF_Pixel_BGRP_32f* RESTRICT localSrc = reinterpret_cast<const PF_Pixel_BGRP_32f* RESTRICT>(pfLayer->data);
                          PF_Pixel_BGRP_32f* RESTRICT localDst = reinterpret_cast<      PF_Pixel_BGRP_32f* RESTRICT>(output->data);
                    const A_long srcLinePitch = rowBytes / static_cast<A_long>(PF_Pixel_BGRP_32f_size);
                    const A_long dstLinePitch = srcLinePitch;

                    AlgoPrIngest::ingest_and_superpixel
                    (
                        localSrc,
                        sizeX,
                        sizeY,
                        srcLinePitch,
                        AlgoPrIngest::fmt_BGRP_4444_32f_Linear,
                        lut8, lut16, lut10,
                        locusGate,
                        algoMemHandler.input_f32_interleaved,
                        super,
                        algoCtrl.confidenceMap
                    );

                    Algorithm_Main (getCctHndl(), super, algoMemHandler, sizeX, sizeY, algoCtrl, cct_duv);

                    AlgoPrIngest::egress_from_linear_f32
                    (
                        algoMemHandler.output_f32_interleaved,
                        sizeX,
                        sizeY,
                        localDst,
                        dstLinePitch,
                        AlgoPrIngest::fmt_BGRP_4444_32f_Linear,
                        lut8, lut16, lut10,
                        localSrc,
                        sizeX,
                        srcLinePitch
                    );
                }
                break;

                case PrPixelFormat_BGRX_4444_8u:
                {
                    const PF_Pixel_BGRX_8u* RESTRICT localSrc = reinterpret_cast<const PF_Pixel_BGRX_8u* RESTRICT>(pfLayer->data);
                          PF_Pixel_BGRX_8u* RESTRICT localDst = reinterpret_cast<      PF_Pixel_BGRX_8u* RESTRICT>(output->data);
                    const A_long srcLinePitch = rowBytes / static_cast<A_long>(PF_Pixel_BGRX_8u_size);
                    const A_long dstLinePitch = srcLinePitch;

                    AlgoPrIngest::ingest_and_superpixel
                    (
                        localSrc,
                        sizeX,
                        sizeY,
                        srcLinePitch,
                        AlgoPrIngest::fmt_BGRX_4444_8u,
                        lut8, lut16, lut10,
                        locusGate,
                        algoMemHandler.input_f32_interleaved,
                        super,
                        algoCtrl.confidenceMap
                    );

                    Algorithm_Main (getCctHndl(), super, algoMemHandler, sizeX, sizeY, algoCtrl, cct_duv);

                    AlgoPrIngest::egress_from_linear_f32
                    (
                        algoMemHandler.output_f32_interleaved,
                        sizeX,
                        sizeY,
                        localDst,
                        dstLinePitch,
                        AlgoPrIngest::fmt_BGRX_4444_8u,
                        lut8, lut16, lut10,
                        localSrc,
                        sizeX,
                        srcLinePitch
                    );
                }
                break;

                case PrPixelFormat_BGRX_4444_16u:
                {
                    const PF_Pixel_BGRX_16u* RESTRICT localSrc = reinterpret_cast<const PF_Pixel_BGRX_16u* RESTRICT>(pfLayer->data);
                          PF_Pixel_BGRX_16u* RESTRICT localDst = reinterpret_cast<      PF_Pixel_BGRX_16u* RESTRICT>(output->data);
                    const A_long srcLinePitch = rowBytes / static_cast<A_long>(PF_Pixel_BGRX_16u_size);
                    const A_long dstLinePitch = srcLinePitch;

                    AlgoPrIngest::ingest_and_superpixel
                    (
                        localSrc,
                        sizeX,
                        sizeY,
                        srcLinePitch,
                        AlgoPrIngest::fmt_BGRX_4444_16u,
                        lut8, lut16, lut10,
                        locusGate,
                        algoMemHandler.input_f32_interleaved,
                        super,
                        algoCtrl.confidenceMap
                    );

                    Algorithm_Main (getCctHndl(), super, algoMemHandler, sizeX, sizeY, algoCtrl, cct_duv);

                    AlgoPrIngest::egress_from_linear_f32
                    (
                        algoMemHandler.output_f32_interleaved,
                        sizeX,
                        sizeY,
                        localDst,
                        dstLinePitch,
                        AlgoPrIngest::fmt_BGRX_4444_16u,
                        lut8, lut16, lut10,
                        localSrc,
                        sizeX,
                        srcLinePitch
                    );
                }
                break;

                case PrPixelFormat_BGRX_4444_32f:
                {
                    const PF_Pixel_BGRX_32f* RESTRICT localSrc = reinterpret_cast<const PF_Pixel_BGRX_32f* RESTRICT>(pfLayer->data);
                          PF_Pixel_BGRX_32f* RESTRICT localDst = reinterpret_cast<      PF_Pixel_BGRX_32f* RESTRICT>(output->data);
                    const A_long srcLinePitch = rowBytes / static_cast<A_long>(PF_Pixel_BGRX_32f_size);
                    const A_long dstLinePitch = srcLinePitch;

                    AlgoPrIngest::ingest_and_superpixel
                    (
                        localSrc,
                        sizeX,
                        sizeY,
                        srcLinePitch,
                        AlgoPrIngest::fmt_BGRX_4444_32f,
                        lut8, lut16, lut10,
                        locusGate,
                        algoMemHandler.input_f32_interleaved,
                        super,
                        algoCtrl.confidenceMap
                    );

                    Algorithm_Main (getCctHndl(), super, algoMemHandler, sizeX, sizeY, algoCtrl, cct_duv);

                    AlgoPrIngest::egress_from_linear_f32
                    (
                        algoMemHandler.output_f32_interleaved,
                        sizeX,
                        sizeY,
                        localDst,
                        dstLinePitch,
                        AlgoPrIngest::fmt_BGRX_4444_32f,
                        lut8, lut16, lut10,
                        localSrc,
                        sizeX,
                        srcLinePitch
                    );
                }
                break;

                case PrPixelFormat_BGRX_4444_32f_Linear:
                {
                    const PF_Pixel_BGRX_32f* RESTRICT localSrc = reinterpret_cast<const PF_Pixel_BGRX_32f* RESTRICT>(pfLayer->data);
                          PF_Pixel_BGRX_32f* RESTRICT localDst = reinterpret_cast<      PF_Pixel_BGRX_32f* RESTRICT>(output->data);
                    const A_long srcLinePitch = rowBytes / static_cast<A_long>(PF_Pixel_BGRX_32f_size);
                    const A_long dstLinePitch = srcLinePitch;

                    AlgoPrIngest::ingest_and_superpixel
                    (
                        localSrc,
                        sizeX,
                        sizeY,
                        srcLinePitch,
                        AlgoPrIngest::fmt_BGRX_4444_32f_Linear,
                        lut8, lut16, lut10,
                        locusGate,
                        algoMemHandler.input_f32_interleaved,
                        super,
                        algoCtrl.confidenceMap
                    );

                    Algorithm_Main (getCctHndl(), super, algoMemHandler, sizeX, sizeY, algoCtrl, cct_duv);

                    AlgoPrIngest::egress_from_linear_f32
                    (
                        algoMemHandler.output_f32_interleaved,
                        sizeX,
                        sizeY,
                        localDst,
                        dstLinePitch,
                        AlgoPrIngest::fmt_BGRX_4444_32f_Linear,
                        lut8, lut16, lut10,
                        localSrc,
                        sizeX,
                        srcLinePitch
                    );
                }
                break;

                case PrPixelFormat_VUYA_4444_8u_709:
                case PrPixelFormat_VUYA_4444_8u:
                {
                    const PF_Pixel_VUYA_8u* RESTRICT localSrc = reinterpret_cast<const PF_Pixel_VUYA_8u* RESTRICT>(pfLayer->data);
                          PF_Pixel_VUYA_8u* RESTRICT localDst = reinterpret_cast<      PF_Pixel_VUYA_8u* RESTRICT>(output->data);
                    const A_long srcLinePitch = rowBytes / static_cast<A_long>(PF_Pixel_VUYA_8u_size);
                    const A_long dstLinePitch = srcLinePitch;

                    AlgoPrIngest::ingest_and_superpixel
                    (
                        localSrc,
                        sizeX,
                        sizeY,
                        srcLinePitch,
                        destinationPixelFormat == PrPixelFormat_VUYA_4444_8u_709 ? AlgoPrIngest::fmt_VUYA_4444_8u_709 : AlgoPrIngest::fmt_VUYA_4444_8u,
                        lut8, lut16, lut10,
                        locusGate,
                        algoMemHandler.input_f32_interleaved,
                        super,
                        algoCtrl.confidenceMap
                    );

                    Algorithm_Main (getCctHndl(), super, algoMemHandler, sizeX, sizeY, algoCtrl, cct_duv);

                    AlgoPrIngest::egress_from_linear_f32
                    (
                        algoMemHandler.output_f32_interleaved,
                        sizeX,
                        sizeY,
                        localDst,
                        dstLinePitch,
                        destinationPixelFormat == PrPixelFormat_VUYA_4444_8u_709 ? AlgoPrIngest::fmt_VUYA_4444_8u_709 : AlgoPrIngest::fmt_VUYA_4444_8u,
                        lut8, lut16, lut10,
                        localSrc,
                        sizeX,
                        srcLinePitch
                    );
                }
                break;

                case PrPixelFormat_VUYA_4444_32f_709:
                case PrPixelFormat_VUYA_4444_32f:
                {
                    const PF_Pixel_VUYA_32f* RESTRICT localSrc = reinterpret_cast<const PF_Pixel_VUYA_32f* RESTRICT>(pfLayer->data);
                          PF_Pixel_VUYA_32f* RESTRICT localDst = reinterpret_cast<      PF_Pixel_VUYA_32f* RESTRICT>(output->data);
                    const A_long srcLinePitch = rowBytes / static_cast<A_long>(PF_Pixel_VUYA_32f_size);
                    const A_long dstLinePitch = srcLinePitch;

                    AlgoPrIngest::ingest_and_superpixel
                    (
                        localSrc,
                        sizeX,
                        sizeY,
                        srcLinePitch,
                        destinationPixelFormat == PrPixelFormat_VUYA_4444_32f_709 ? AlgoPrIngest::fmt_VUYA_4444_32f_709 : AlgoPrIngest::fmt_VUYA_4444_32f,
                        lut8, lut16, lut10,
                        locusGate,
                        algoMemHandler.input_f32_interleaved,
                        super,
                        algoCtrl.confidenceMap
                    );

                    Algorithm_Main (getCctHndl(), super, algoMemHandler, sizeX, sizeY, algoCtrl, cct_duv);

                    AlgoPrIngest::egress_from_linear_f32
                    (
                        algoMemHandler.output_f32_interleaved,
                        sizeX,
                        sizeY,
                        localDst,
                        dstLinePitch,
                        destinationPixelFormat == PrPixelFormat_VUYA_4444_32f_709 ? AlgoPrIngest::fmt_VUYA_4444_32f_709 : AlgoPrIngest::fmt_VUYA_4444_32f,
                        lut8, lut16, lut10,
                        localSrc,
                        sizeX,
                        srcLinePitch
                    );
                }
                break;

                case PrPixelFormat_VUYP_4444_8u_709:
                case PrPixelFormat_VUYP_4444_8u:
                {
                    const PF_Pixel_VUYP_8u* RESTRICT localSrc = reinterpret_cast<const PF_Pixel_VUYP_8u* RESTRICT>(pfLayer->data);
                          PF_Pixel_VUYP_8u* RESTRICT localDst = reinterpret_cast<      PF_Pixel_VUYP_8u* RESTRICT>(output->data);
                    const A_long srcLinePitch = rowBytes / static_cast<A_long>(PF_Pixel_VUYP_8u_size);
                    const A_long dstLinePitch = srcLinePitch;

                    AlgoPrIngest::ingest_and_superpixel
                    (
                        localSrc,
                        sizeX,
                        sizeY,
                        srcLinePitch,
                        destinationPixelFormat == PrPixelFormat_VUYP_4444_8u_709 ? AlgoPrIngest::fmt_VUYP_4444_8u_709 : AlgoPrIngest::fmt_VUYP_4444_8u,
                        lut8, lut16, lut10,
                        locusGate,
                        algoMemHandler.input_f32_interleaved,
                        super,
                        algoCtrl.confidenceMap
                    );

                    Algorithm_Main (getCctHndl(), super, algoMemHandler, sizeX, sizeY, algoCtrl, cct_duv);

                    AlgoPrIngest::egress_from_linear_f32
                    (
                        algoMemHandler.output_f32_interleaved,
                        sizeX,
                        sizeY,
                        localDst,
                        dstLinePitch,
                        destinationPixelFormat == PrPixelFormat_VUYP_4444_8u_709 ? AlgoPrIngest::fmt_VUYP_4444_8u_709 : AlgoPrIngest::fmt_VUYP_4444_8u,
                        lut8, lut16, lut10,
                        localSrc,
                        sizeX,
                        srcLinePitch
                    );
                }
                break;

                case PrPixelFormat_VUYP_4444_32f_709:
                case PrPixelFormat_VUYP_4444_32f:
                {
                    const PF_Pixel_VUYP_32f* RESTRICT localSrc = reinterpret_cast<const PF_Pixel_VUYP_32f* RESTRICT>(pfLayer->data);
                          PF_Pixel_VUYP_32f* RESTRICT localDst = reinterpret_cast<      PF_Pixel_VUYP_32f* RESTRICT>(output->data);
                    const A_long srcLinePitch = rowBytes / static_cast<A_long>(PF_Pixel_VUYP_32f_size);
                    const A_long dstLinePitch = srcLinePitch;

                    AlgoPrIngest::ingest_and_superpixel
                    (
                        localSrc,
                        sizeX,
                        sizeY,
                        srcLinePitch,
                        destinationPixelFormat == PrPixelFormat_VUYP_4444_32f_709 ? AlgoPrIngest::fmt_VUYP_4444_32f_709 : AlgoPrIngest::fmt_VUYP_4444_32f,
                        lut8, lut16, lut10,
                        locusGate,
                        algoMemHandler.input_f32_interleaved,
                        super,
                        algoCtrl.confidenceMap
                    );

                    Algorithm_Main (getCctHndl(), super, algoMemHandler, sizeX, sizeY, algoCtrl, cct_duv);

                    AlgoPrIngest::egress_from_linear_f32
                    (
                        algoMemHandler.output_f32_interleaved,
                        sizeX,
                        sizeY,
                        localDst,
                        dstLinePitch,
                        destinationPixelFormat == PrPixelFormat_VUYP_4444_32f_709 ? AlgoPrIngest::fmt_VUYP_4444_32f_709 : AlgoPrIngest::fmt_VUYP_4444_32f,
                        lut8, lut16, lut10,
                        localSrc,
                        sizeX,
                        srcLinePitch
                    );
                }
                break;

                case PrPixelFormat_VUYX_4444_8u_709:
                case PrPixelFormat_VUYX_4444_8u:
                {
                    const PF_Pixel_VUYX_8u* RESTRICT localSrc = reinterpret_cast<const PF_Pixel_VUYX_8u* RESTRICT>(pfLayer->data);
                          PF_Pixel_VUYX_8u* RESTRICT localDst = reinterpret_cast<      PF_Pixel_VUYX_8u* RESTRICT>(output->data);
                    const A_long srcLinePitch = rowBytes / static_cast<A_long>(PF_Pixel_VUYX_8u_size);
                    const A_long dstLinePitch = srcLinePitch;

                    AlgoPrIngest::ingest_and_superpixel
                    (
                        localSrc,
                        sizeX,
                        sizeY,
                        srcLinePitch,
                        destinationPixelFormat == PrPixelFormat_VUYX_4444_8u_709 ? AlgoPrIngest::fmt_VUYX_4444_8u_709 : AlgoPrIngest::fmt_VUYX_4444_8u,
                        lut8, lut16, lut10,
                        locusGate,
                        algoMemHandler.input_f32_interleaved,
                        super,
                        algoCtrl.confidenceMap
                    );

                    Algorithm_Main(getCctHndl(), super, algoMemHandler, sizeX, sizeY, algoCtrl, cct_duv);

                    AlgoPrIngest::egress_from_linear_f32
                    (
                        algoMemHandler.output_f32_interleaved,
                        sizeX,
                        sizeY,
                        localDst,
                        dstLinePitch,
                        destinationPixelFormat == PrPixelFormat_VUYX_4444_8u_709 ? AlgoPrIngest::fmt_VUYX_4444_8u_709 : AlgoPrIngest::fmt_VUYX_4444_8u,
                        lut8, lut16, lut10,
                        localSrc,
                        sizeX,
                        srcLinePitch
                    );
                }
                break;

                case PrPixelFormat_VUYX_4444_32f_709:
                case PrPixelFormat_VUYX_4444_32f:
                {
                    const PF_Pixel_VUYX_32f* RESTRICT localSrc = reinterpret_cast<const PF_Pixel_VUYX_32f* RESTRICT>(pfLayer->data);
                    PF_Pixel_VUYX_32f* RESTRICT localDst = reinterpret_cast<      PF_Pixel_VUYX_32f* RESTRICT>(output->data);
                    const A_long srcLinePitch = rowBytes / static_cast<A_long>(PF_Pixel_VUYX_32f_size);
                    const A_long dstLinePitch = srcLinePitch;

                    AlgoPrIngest::ingest_and_superpixel
                    (
                        localSrc,
                        sizeX,
                        sizeY,
                        srcLinePitch,
                        destinationPixelFormat == PrPixelFormat_VUYX_4444_32f_709 ? AlgoPrIngest::fmt_VUYX_4444_32f_709 : AlgoPrIngest::fmt_VUYX_4444_32f,
                        lut8, lut16, lut10,
                        locusGate,
                        algoMemHandler.input_f32_interleaved,
                        super,
                        algoCtrl.confidenceMap
                    );

                    Algorithm_Main(getCctHndl(), super, algoMemHandler, sizeX, sizeY, algoCtrl, cct_duv);

                    AlgoPrIngest::egress_from_linear_f32
                    (
                        algoMemHandler.output_f32_interleaved,
                        sizeX,
                        sizeY,
                        localDst,
                        dstLinePitch,
                        destinationPixelFormat == PrPixelFormat_VUYX_4444_32f_709 ? AlgoPrIngest::fmt_VUYX_4444_32f_709 : AlgoPrIngest::fmt_VUYX_4444_32f,
                        lut8, lut16, lut10,
                        localSrc,
                        sizeX,
                        srcLinePitch
                    );
                }
                break;

                case PrPixelFormat_ARGB_4444_8u:
                {
                    const PF_Pixel_ARGB_8u* RESTRICT localSrc = reinterpret_cast<const PF_Pixel_ARGB_8u* RESTRICT>(pfLayer->data);
                          PF_Pixel_ARGB_8u* RESTRICT localDst = reinterpret_cast<      PF_Pixel_ARGB_8u* RESTRICT>(output->data);
                    const A_long srcLinePitch = rowBytes / static_cast<A_long>(PF_Pixel_ARGB_8u_size);
                    const A_long dstLinePitch = srcLinePitch;

                    AlgoPrIngest::ingest_and_superpixel
                    (
                        localSrc,
                        sizeX,
                        sizeY,
                        srcLinePitch,
                        AlgoPrIngest::fmt_ARGB_4444_8u,
                        lut8, lut16, lut10,
                        locusGate,
                        algoMemHandler.input_f32_interleaved,
                        super,
                        algoCtrl.confidenceMap
                    );

                    Algorithm_Main(getCctHndl(), super, algoMemHandler, sizeX, sizeY, algoCtrl, cct_duv);

                    AlgoPrIngest::egress_from_linear_f32
                    (
                        algoMemHandler.output_f32_interleaved,
                        sizeX,
                        sizeY,
                        localDst,
                        dstLinePitch,
                        AlgoPrIngest::fmt_ARGB_4444_8u,
                        lut8, lut16, lut10,
                        localSrc,
                        sizeX,
                        srcLinePitch
                    );
                }
                break;

                case PrPixelFormat_PRGB_4444_8u:
                {
                    const PF_Pixel_PRGB_8u* RESTRICT localSrc = reinterpret_cast<const PF_Pixel_PRGB_8u* RESTRICT>(pfLayer->data);
                          PF_Pixel_PRGB_8u* RESTRICT localDst = reinterpret_cast<      PF_Pixel_PRGB_8u* RESTRICT>(output->data);
                    const A_long srcLinePitch = rowBytes / static_cast<A_long>(PF_Pixel_PRGB_8u_size);
                    const A_long dstLinePitch = srcLinePitch;

                    AlgoPrIngest::ingest_and_superpixel
                    (
                        localSrc,
                        sizeX,
                        sizeY,
                        srcLinePitch,
                        AlgoPrIngest::fmt_PRGB_4444_8u,
                        lut8, lut16, lut10,
                        locusGate,
                        algoMemHandler.input_f32_interleaved,
                        super,
                        algoCtrl.confidenceMap
                    );

                    Algorithm_Main(getCctHndl(), super, algoMemHandler, sizeX, sizeY, algoCtrl, cct_duv);

                    AlgoPrIngest::egress_from_linear_f32
                    (
                        algoMemHandler.output_f32_interleaved,
                        sizeX,
                        sizeY,
                        localDst,
                        dstLinePitch,
                        AlgoPrIngest::fmt_PRGB_4444_8u,
                        lut8, lut16, lut10,
                        localSrc,
                        sizeX,
                        srcLinePitch
                    );
                }
                break;

                case PrPixelFormat_XRGB_4444_8u:
                {
                    const PF_Pixel_XRGB_8u* RESTRICT localSrc = reinterpret_cast<const PF_Pixel_XRGB_8u* RESTRICT>(pfLayer->data);
                          PF_Pixel_XRGB_8u* RESTRICT localDst = reinterpret_cast<      PF_Pixel_XRGB_8u* RESTRICT>(output->data);
                    const A_long srcLinePitch = rowBytes / static_cast<A_long>(PF_Pixel_XRGB_8u_size);
                    const A_long dstLinePitch = srcLinePitch;

                    AlgoPrIngest::ingest_and_superpixel
                    (
                        localSrc,
                        sizeX,
                        sizeY,
                        srcLinePitch,
                        AlgoPrIngest::fmt_XRGB_4444_8u,
                        lut8, lut16, lut10,
                        locusGate,
                        algoMemHandler.input_f32_interleaved,
                        super,
                        algoCtrl.confidenceMap
                    );

                    Algorithm_Main(getCctHndl(), super, algoMemHandler, sizeX, sizeY, algoCtrl, cct_duv);

                    AlgoPrIngest::egress_from_linear_f32
                    (
                        algoMemHandler.output_f32_interleaved,
                        sizeX,
                        sizeY,
                        localDst,
                        dstLinePitch,
                        AlgoPrIngest::fmt_XRGB_4444_8u,
                        lut8, lut16, lut10,
                        localSrc,
                        sizeX,
                        srcLinePitch
                    );
                }
                break;

                case PrPixelFormat_ARGB_4444_16u:
                {
                    const PF_Pixel_ARGB_16u* RESTRICT localSrc = reinterpret_cast<const PF_Pixel_ARGB_16u* RESTRICT>(pfLayer->data);
                          PF_Pixel_ARGB_16u* RESTRICT localDst = reinterpret_cast<      PF_Pixel_ARGB_16u* RESTRICT>(output->data);
                    const A_long srcLinePitch = rowBytes / static_cast<A_long>(PF_Pixel_ARGB_16u_size);
                    const A_long dstLinePitch = srcLinePitch;

                    AlgoPrIngest::ingest_and_superpixel
                    (
                        localSrc,
                        sizeX,
                        sizeY,
                        srcLinePitch,
                        AlgoPrIngest::fmt_ARGB_4444_16u,
                        lut8, lut16, lut10,
                        locusGate,
                        algoMemHandler.input_f32_interleaved,
                        super,
                        algoCtrl.confidenceMap
                    );

                    Algorithm_Main (getCctHndl(), super, algoMemHandler, sizeX, sizeY, algoCtrl, cct_duv);

                    AlgoPrIngest::egress_from_linear_f32
                    (
                        algoMemHandler.output_f32_interleaved,
                        sizeX,
                        sizeY,
                        localDst,
                        dstLinePitch,
                        AlgoPrIngest::fmt_ARGB_4444_16u,
                        lut8, lut16, lut10,
                        localSrc,
                        sizeX,
                        srcLinePitch
                    );
                }
                break;

                case PrPixelFormat_PRGB_4444_16u:
                {
                    const PF_Pixel_PRGB_16u* RESTRICT localSrc = reinterpret_cast<const PF_Pixel_PRGB_16u* RESTRICT>(pfLayer->data);
                          PF_Pixel_PRGB_16u* RESTRICT localDst = reinterpret_cast<      PF_Pixel_PRGB_16u* RESTRICT>(output->data);
                    const A_long srcLinePitch = rowBytes / static_cast<A_long>(PF_Pixel_PRGB_16u_size);
                    const A_long dstLinePitch = srcLinePitch;

                    AlgoPrIngest::ingest_and_superpixel
                    (
                        localSrc,
                        sizeX,
                        sizeY,
                        srcLinePitch,
                        AlgoPrIngest::fmt_PRGB_4444_16u,
                        lut8, lut16, lut10,
                        locusGate,
                        algoMemHandler.input_f32_interleaved,
                        super,
                        algoCtrl.confidenceMap
                    );

                    Algorithm_Main (getCctHndl(), super, algoMemHandler, sizeX, sizeY, algoCtrl, cct_duv);

                    AlgoPrIngest::egress_from_linear_f32
                    (
                        algoMemHandler.output_f32_interleaved,
                        sizeX,
                        sizeY,
                        localDst,
                        dstLinePitch,
                        AlgoPrIngest::fmt_PRGB_4444_16u,
                        lut8, lut16, lut10,
                        localSrc,
                        sizeX,
                        srcLinePitch
                    );
                }
                break;

                case PrPixelFormat_XRGB_4444_16u:
                {
                    const PF_Pixel_XRGB_16u* RESTRICT localSrc = reinterpret_cast<const PF_Pixel_XRGB_16u* RESTRICT>(pfLayer->data);
                          PF_Pixel_XRGB_16u* RESTRICT localDst = reinterpret_cast<      PF_Pixel_XRGB_16u* RESTRICT>(output->data);
                    const A_long srcLinePitch = rowBytes / static_cast<A_long>(PF_Pixel_XRGB_16u_size);
                    const A_long dstLinePitch = srcLinePitch;

                    AlgoPrIngest::ingest_and_superpixel
                    (
                        localSrc,
                        sizeX,
                        sizeY,
                        srcLinePitch,
                        AlgoPrIngest::fmt_XRGB_4444_16u,
                        lut8, lut16, lut10,
                        locusGate,
                        algoMemHandler.input_f32_interleaved,
                        super,
                        algoCtrl.confidenceMap
                    );

                    Algorithm_Main(getCctHndl(), super, algoMemHandler, sizeX, sizeY, algoCtrl, cct_duv);

                    AlgoPrIngest::egress_from_linear_f32
                    (
                        algoMemHandler.output_f32_interleaved,
                        sizeX,
                        sizeY,
                        localDst,
                        dstLinePitch,
                        AlgoPrIngest::fmt_XRGB_4444_16u,
                        lut8, lut16, lut10,
                        localSrc,
                        sizeX,
                        srcLinePitch
                    );
                }
                break;

                case PrPixelFormat_ARGB_4444_32f:
                {
                    const PF_Pixel_ARGB_32f* RESTRICT localSrc = reinterpret_cast<const PF_Pixel_ARGB_32f* RESTRICT>(pfLayer->data);
                          PF_Pixel_ARGB_32f* RESTRICT localDst = reinterpret_cast<      PF_Pixel_ARGB_32f* RESTRICT>(output->data);
                    const A_long srcLinePitch = rowBytes / static_cast<A_long>(PF_Pixel_ARGB_32f_size);
                    const A_long dstLinePitch = srcLinePitch;

                    AlgoPrIngest::ingest_and_superpixel
                    (
                        localSrc,
                        sizeX,
                        sizeY,
                        srcLinePitch,
                        AlgoPrIngest::fmt_ARGB_4444_32f,
                        lut8, lut16, lut10,
                        locusGate,
                        algoMemHandler.input_f32_interleaved,
                        super,
                        algoCtrl.confidenceMap
                    );

                    Algorithm_Main(getCctHndl(), super, algoMemHandler, sizeX, sizeY, algoCtrl, cct_duv);

                    AlgoPrIngest::egress_from_linear_f32
                    (
                        algoMemHandler.output_f32_interleaved,
                        sizeX,
                        sizeY,
                        localDst,
                        dstLinePitch,
                        AlgoPrIngest::fmt_ARGB_4444_32f,
                        lut8, lut16, lut10,
                        localSrc,
                        sizeX,
                        srcLinePitch
                    );
                }
                break;

                case PrPixelFormat_PRGB_4444_32f:
                {
                    const PF_Pixel_PRGB_32f* RESTRICT localSrc = reinterpret_cast<const PF_Pixel_PRGB_32f* RESTRICT>(pfLayer->data);
                          PF_Pixel_PRGB_32f* RESTRICT localDst = reinterpret_cast<      PF_Pixel_PRGB_32f* RESTRICT>(output->data);
                    const A_long srcLinePitch = rowBytes / static_cast<A_long>(PF_Pixel_PRGB_32f_size);
                    const A_long dstLinePitch = srcLinePitch;

                    AlgoPrIngest::ingest_and_superpixel
                    (
                        localSrc,
                        sizeX,
                        sizeY,
                        srcLinePitch,
                        AlgoPrIngest::fmt_PRGB_4444_32f,
                        lut8, lut16, lut10,
                        locusGate,
                        algoMemHandler.input_f32_interleaved,
                        super,
                        algoCtrl.confidenceMap
                    );

                    Algorithm_Main (getCctHndl(), super, algoMemHandler, sizeX, sizeY, algoCtrl, cct_duv);

                    AlgoPrIngest::egress_from_linear_f32
                    (
                        algoMemHandler.output_f32_interleaved,
                        sizeX,
                        sizeY,
                        localDst,
                        dstLinePitch,
                        AlgoPrIngest::fmt_PRGB_4444_32f,
                        lut8, lut16, lut10,
                        localSrc,
                        sizeX,
                        srcLinePitch
                    );
                }
                break;

                case PrPixelFormat_XRGB_4444_32f:
                {
                    const PF_Pixel_XRGB_32f* RESTRICT localSrc = reinterpret_cast<const PF_Pixel_XRGB_32f* RESTRICT>(pfLayer->data);
                          PF_Pixel_XRGB_32f* RESTRICT localDst = reinterpret_cast<      PF_Pixel_XRGB_32f* RESTRICT>(output->data);
                    const A_long srcLinePitch = rowBytes / static_cast<A_long>(PF_Pixel_XRGB_32f_size);
                    const A_long dstLinePitch = srcLinePitch;

                    AlgoPrIngest::ingest_and_superpixel
                    (
                        localSrc,
                        sizeX,
                        sizeY,
                        srcLinePitch,
                        AlgoPrIngest::fmt_XRGB_4444_32f,
                        lut8, lut16, lut10,
                        locusGate,
                        algoMemHandler.input_f32_interleaved,
                        super,
                        algoCtrl.confidenceMap
                    );

                    Algorithm_Main(getCctHndl(), super, algoMemHandler, sizeX, sizeY, algoCtrl, cct_duv);

                    AlgoPrIngest::egress_from_linear_f32
                    (
                        algoMemHandler.output_f32_interleaved,
                        sizeX,
                        sizeY,
                        localDst,
                        dstLinePitch,
                        AlgoPrIngest::fmt_XRGB_4444_32f,
                        lut8, lut16, lut10,
                        localSrc,
                        sizeX,
                        srcLinePitch
                    );
                }
                break;

                case PrPixelFormat_ARGB_4444_32f_Linear:
                {
                    const PF_Pixel_ARGB_32f* RESTRICT localSrc = reinterpret_cast<const PF_Pixel_ARGB_32f* RESTRICT>(pfLayer->data);
                          PF_Pixel_ARGB_32f* RESTRICT localDst = reinterpret_cast<      PF_Pixel_ARGB_32f* RESTRICT>(output->data);
                    const A_long srcLinePitch = rowBytes / static_cast<A_long>(PF_Pixel_ARGB_32f_size);
                    const A_long dstLinePitch = srcLinePitch;

                    AlgoPrIngest::ingest_and_superpixel
                    (
                        localSrc,
                        sizeX,
                        sizeY,
                        srcLinePitch,
                        AlgoPrIngest::fmt_ARGB_4444_32f_Linear,
                        lut8, lut16, lut10,
                        locusGate,
                        algoMemHandler.input_f32_interleaved,
                        super,
                        algoCtrl.confidenceMap
                    );

                    Algorithm_Main(getCctHndl(), super, algoMemHandler, sizeX, sizeY, algoCtrl, cct_duv);

                    AlgoPrIngest::egress_from_linear_f32
                    (
                        algoMemHandler.output_f32_interleaved,
                        sizeX,
                        sizeY,
                        localDst,
                        dstLinePitch,
                        AlgoPrIngest::fmt_ARGB_4444_32f_Linear,
                        lut8, lut16, lut10,
                        localSrc,
                        sizeX,
                        srcLinePitch
                    );
                }
                break;

                case PrPixelFormat_PRGB_4444_32f_Linear:
                {
                    const PF_Pixel_PRGB_32f* RESTRICT localSrc = reinterpret_cast<const PF_Pixel_PRGB_32f* RESTRICT>(pfLayer->data);
                          PF_Pixel_PRGB_32f* RESTRICT localDst = reinterpret_cast<      PF_Pixel_PRGB_32f* RESTRICT>(output->data);
                    const A_long srcLinePitch = rowBytes / static_cast<A_long>(PF_Pixel_PRGB_32f_size);
                    const A_long dstLinePitch = srcLinePitch;

                    AlgoPrIngest::ingest_and_superpixel
                    (
                        localSrc,
                        sizeX,
                        sizeY,
                        srcLinePitch,
                        AlgoPrIngest::fmt_PRGB_4444_32f_Linear,
                        lut8, lut16, lut10,
                        locusGate,
                        algoMemHandler.input_f32_interleaved,
                        super,
                        algoCtrl.confidenceMap
                    );

                    Algorithm_Main (getCctHndl(), super, algoMemHandler, sizeX, sizeY, algoCtrl, cct_duv);

                    AlgoPrIngest::egress_from_linear_f32
                    (
                        algoMemHandler.output_f32_interleaved,
                        sizeX,
                        sizeY,
                        localDst,
                        dstLinePitch,
                        AlgoPrIngest::fmt_PRGB_4444_32f_Linear,
                        lut8, lut16, lut10,
                        localSrc,
                        sizeX,
                        srcLinePitch
                    );
                }
                break;

                case PrPixelFormat_XRGB_4444_32f_Linear:
                {
                    const PF_Pixel_XRGB_32f* RESTRICT localSrc = reinterpret_cast<const PF_Pixel_XRGB_32f* RESTRICT>(pfLayer->data);
                          PF_Pixel_XRGB_32f* RESTRICT localDst = reinterpret_cast<      PF_Pixel_XRGB_32f* RESTRICT>(output->data);
                    const A_long srcLinePitch = rowBytes / static_cast<A_long>(PF_Pixel_XRGB_32f_size);
                    const A_long dstLinePitch = srcLinePitch;

                    AlgoPrIngest::ingest_and_superpixel
                    (
                        localSrc,
                        sizeX,
                        sizeY,
                        srcLinePitch,
                        AlgoPrIngest::fmt_XRGB_4444_32f_Linear,
                        lut8, lut16, lut10,
                        locusGate,
                        algoMemHandler.input_f32_interleaved,
                        super,
                        algoCtrl.confidenceMap
                    );

                    Algorithm_Main (getCctHndl(), super, algoMemHandler, sizeX, sizeY, algoCtrl, cct_duv);

                    AlgoPrIngest::egress_from_linear_f32
                    (
                        algoMemHandler.output_f32_interleaved,
                        sizeX,
                        sizeY,
                        localDst,
                        dstLinePitch,
                        AlgoPrIngest::fmt_XRGB_4444_32f_Linear,
                        lut8, lut16, lut10,
                        localSrc,
                        sizeX,
                        srcLinePitch
                    );
                }
                break;

                case PrPixelFormat_RGB_444_10u:
                {
                    const PF_Pixel_RGB_10u* RESTRICT localSrc = reinterpret_cast<const PF_Pixel_RGB_10u* RESTRICT>(pfLayer->data);
                          PF_Pixel_RGB_10u* RESTRICT localDst = reinterpret_cast<      PF_Pixel_RGB_10u* RESTRICT>(output->data);
                    const A_long srcLinePitch = rowBytes / static_cast<A_long>(PF_Pixel_RGB_10u_size);
                    const A_long dstLinePitch = srcLinePitch;

                    AlgoPrIngest::ingest_and_superpixel
                    (
                        localSrc,
                        sizeX,
                        sizeY,
                        srcLinePitch,
                        AlgoPrIngest::fmt_RGB_444_10u,
                        lut8, lut16, lut10,
                        locusGate,
                        algoMemHandler.input_f32_interleaved,
                        super,
                        algoCtrl.confidenceMap
                    );

                    Algorithm_Main (getCctHndl(), super, algoMemHandler, sizeX, sizeY, algoCtrl, cct_duv);

                    AlgoPrIngest::egress_from_linear_f32
                    (
                        algoMemHandler.output_f32_interleaved,
                        sizeX,
                        sizeY,
                        localDst,
                        dstLinePitch,
                        AlgoPrIngest::fmt_RGB_444_10u,
                        lut8, lut16, lut10,
                        localSrc,
                        sizeX,
                        srcLinePitch
                    );
                }
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
