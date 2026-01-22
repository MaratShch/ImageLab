#include "ArtPointillism.hpp"
#include "ArtPointillismAlgo.hpp"
#include "ArtPointillismEnums.hpp"
#include "ArtPointillismControl.hpp"
#include "Avx2ColorConverts.hpp"
#include "CommonSmartRender.hpp"
#include "ImageLabMemInterface.hpp"


using PPontillismControls = PontillismControls*;

PF_Err
ArtPointilism_PreRender
(
    PF_InData			*in_data,
    PF_OutData			*out_data,
    PF_PreRenderExtra	*extra
)
{
    PontillismControls renderParams{};
    PF_Err err = PF_Err_NONE;

    AEFX_SuiteScoper<PF_HandleSuite1> handleSuite = AEFX_SuiteScoper<PF_HandleSuite1>(in_data, kPFHandleSuite, kPFHandleSuiteVersion1, out_data);
    PF_Handle paramsHandler = handleSuite->host_new_handle(PontillismControlsSize);

    if (nullptr != paramsHandler)
    {
        PPontillismControls paramsStrP = reinterpret_cast<PPontillismControls>(handleSuite->host_lock_handle(paramsHandler));
        if (nullptr != paramsStrP)
        {
            CACHE_ALIGN PF_ParamDef paramVal{};

            extra->output->pre_render_data = paramsHandler;

            PF_CHECKOUT_PARAM (in_data, UnderlyingType(ArtPointillismControls::ART_POINTILLISM_PAINTER_STYLE), in_data->current_time, in_data->time_step, in_data->time_scale, &paramVal);
                renderParams.PainterStyle = static_cast<ArtPointillismPainter>(paramVal.u.pd.value - 1);
            PF_CHECKOUT_PARAM(in_data, UnderlyingType(ArtPointillismControls::ART_POINTILLISM_SLIDER_DOT_DENCITY), in_data->current_time, in_data->time_step, in_data->time_scale, &paramVal);
                renderParams.DotDencity = static_cast<int32_t>(paramVal.u.sd.value);
            PF_CHECKOUT_PARAM(in_data, UnderlyingType(ArtPointillismControls::ART_POINTILLISM_SLIDER_DOT_SIZE), in_data->current_time, in_data->time_step, in_data->time_scale, &paramVal);
                renderParams.DotSize = static_cast<int32_t>(paramVal.u.sd.value);
            PF_CHECKOUT_PARAM(in_data, UnderlyingType(ArtPointillismControls::ART_POINTILLISM_SLIDER_EDGE_SENSITIVITY), in_data->current_time, in_data->time_step, in_data->time_scale, &paramVal);
                renderParams.EdgeSensitivity = static_cast<int32_t>(paramVal.u.sd.value);
            PF_CHECKOUT_PARAM(in_data, UnderlyingType(ArtPointillismControls::ART_POINTILLISM_SLIDER_COLOR_VIBRANCE), in_data->current_time, in_data->time_step, in_data->time_scale, &paramVal);
                renderParams.Vibrancy = static_cast<int32_t>(paramVal.u.sd.value);
            PF_CHECKOUT_PARAM(in_data, UnderlyingType(ArtPointillismControls::ART_POINTILLISM_BACKGROUND_ART), in_data->current_time, in_data->time_step, in_data->time_scale, &paramVal);
                renderParams.Background = static_cast<BackgroundArt>(paramVal.u.pd.value - 1);
            PF_CHECKOUT_PARAM(in_data, UnderlyingType(ArtPointillismControls::ART_POINTILLISM_OPACITY), in_data->current_time, in_data->time_step, in_data->time_scale, &paramVal);
                renderParams.Opacity = static_cast<int32_t>(paramVal.u.sd.value);
            PF_CHECKOUT_PARAM(in_data, UnderlyingType(ArtPointillismControls::ART_POINTILLISM_RANDOM_SEED), in_data->current_time, in_data->time_step, in_data->time_scale, &paramVal);
                renderParams.RandomSeed = static_cast<int32_t>(paramVal.u.sd.value);

           PF_RenderRequest req = extra->input->output_request;
           PF_CheckoutResult in_result{};

            ERR(extra->cb->checkout_layer
                (in_data->effect_ref, 
                    UnderlyingType(ArtPointillismControls::ART_POINTILLISM_INPUT), 
                    UnderlyingType(ArtPointillismControls::ART_POINTILLISM_INPUT), 
                    &req, 
                    in_data->current_time, 
                    in_data->time_step, 
                    in_data->time_scale, 
                    &in_result)
            );

            UnionLRect(&in_result.result_rect, &extra->output->result_rect);
            UnionLRect(&in_result.max_result_rect, &extra->output->max_result_rect);

            handleSuite->host_unlock_handle(paramsHandler);

        } // if (nullptr != paramsStrP)
        else
            err = PF_Err_INTERNAL_STRUCT_DAMAGED;
    } // if (nullptr != paramsHandler)
    else
        err = PF_Err_OUT_OF_MEMORY;

    return err;
}


PF_Err
ArtPointilism_SmartRender
(
    PF_InData				*in_data,
    PF_OutData				*out_data,
    PF_SmartRenderExtra		*extraP
)
{
    PF_EffectWorld* input_worldP = nullptr;
    PF_EffectWorld* output_worldP = nullptr;
    PF_Err	err = PF_Err_NONE;

    AEFX_SuiteScoper<PF_HandleSuite1> handleSuite = AEFX_SuiteScoper<PF_HandleSuite1>(in_data, kPFHandleSuite, kPFHandleSuiteVersion1, out_data);
    const PontillismControls* pFilterStrParams = reinterpret_cast<const PontillismControls*>(handleSuite->host_lock_handle(reinterpret_cast<PF_Handle>(extraP->input->pre_render_data)));

    if (nullptr != pFilterStrParams)
    {
        ERR((extraP->cb->checkout_layer_pixels(in_data->effect_ref, UnderlyingType(ArtPointillismControls::ART_POINTILLISM_INPUT), &input_worldP)));
        ERR(extraP->cb->checkout_output(in_data->effect_ref, &output_worldP));

        if (nullptr != input_worldP && nullptr != output_worldP)
        {
            const A_long sizeX = input_worldP->width;
            const A_long sizeY = input_worldP->height;
            const A_long srcRowBytes = input_worldP->rowbytes;  // Get input buffer pitch in bytes
            const A_long dstRowBytes = output_worldP->rowbytes; // Get output buffer pitch in bytes

            PF_PixelFormat format = PF_PixelFormat_INVALID;
            AEFX_SuiteScoper<PF_WorldSuite2> wsP = AEFX_SuiteScoper<PF_WorldSuite2>(in_data, kPFWorldSuite, kPFWorldSuiteVersion2, out_data);

            MemHandler algoMemHandler = alloc_memory_buffers(sizeX, sizeY);
            if (true == mem_handler_valid(algoMemHandler))
            {
                float* __restrict srcL  = algoMemHandler.L;
                float* __restrict srcAB = algoMemHandler.ab;
                float* __restrict dstL  = algoMemHandler.dst_L;
                float* __restrict dstAB = algoMemHandler.dst_ab;

                if (PF_Err_NONE == wsP->PF_GetPixelFormat(input_worldP, &format))
                {
                    switch (format)
                    {
                        case PF_PixelFormat_ARGB128:
                        {
                            const A_long srcPitch = srcRowBytes / static_cast<A_long>(PF_Pixel_ARGB_32f_size);
                            const A_long dstPitch = dstRowBytes / static_cast<A_long>(PF_Pixel_ARGB_32f_size);

                            const PF_Pixel_ARGB_32f* __restrict input_pixels  = reinterpret_cast<const PF_Pixel_ARGB_32f* __restrict>(input_worldP->data);
                                  PF_Pixel_ARGB_32f* __restrict output_pixels = reinterpret_cast<      PF_Pixel_ARGB_32f* __restrict>(output_worldP->data);

                            // convert to semi-planar CieLAB color space
                            AVX2_ConvertRgbToCIELab_SemiPlanar (input_pixels, srcL, srcAB, sizeX, sizeY, srcPitch, sizeX);

                            // execute algorithm
                            ArtPointillismAlgorithmExec (algoMemHandler, *pFilterStrParams, sizeX, sizeY);

                            // back convert to native buffer format after processing complete
                            AVX2_ConvertCIELab_SemiPlanar_ToRgb (input_pixels, dstL, dstAB, output_pixels, sizeX, sizeY, srcPitch, dstPitch);
                        }
                        break;

                        case PF_PixelFormat_ARGB64:
                        {
                            const A_long srcPitch = srcRowBytes / static_cast<A_long>(PF_Pixel_ARGB_16u_size);
                            const A_long dstPitch = dstRowBytes / static_cast<A_long>(PF_Pixel_ARGB_16u_size);

                            const PF_Pixel_ARGB_16u* __restrict input_pixels  = reinterpret_cast<const PF_Pixel_ARGB_16u* __restrict>(input_worldP->data);
                                  PF_Pixel_ARGB_16u* __restrict output_pixels = reinterpret_cast<      PF_Pixel_ARGB_16u* __restrict>(output_worldP->data);

                            // convert to semi-planar CieLAB color space
                            AVX2_ConvertRgbToCIELab_SemiPlanar (input_pixels, srcL, srcAB, sizeX, sizeY, srcPitch, sizeX);

                            // execute algorithm
                            ArtPointillismAlgorithmExec (algoMemHandler, *pFilterStrParams, sizeX, sizeY);

                            // back convert to native buffer format after processing complete
                            AVX2_ConvertCIELab_SemiPlanar_ToRgb (input_pixels, dstL, dstAB, output_pixels, sizeX, sizeY, srcPitch, dstPitch);
                        }
                        break;

                        case PF_PixelFormat_ARGB32:
                        {
                            const A_long srcPitch = srcRowBytes / static_cast<A_long>(PF_Pixel_ARGB_8u_size);
                            const A_long dstPitch = dstRowBytes / static_cast<A_long>(PF_Pixel_ARGB_8u_size);

                            const PF_Pixel_ARGB_8u* __restrict input_pixels  = reinterpret_cast<const PF_Pixel_ARGB_8u* __restrict>(input_worldP->data);
                                  PF_Pixel_ARGB_8u* __restrict output_pixels = reinterpret_cast<      PF_Pixel_ARGB_8u* __restrict>(output_worldP->data);

                            // convert to semi-planar CieLAB color space
                            AVX2_ConvertRgbToCIELab_SemiPlanar(input_pixels, srcL, srcAB, sizeX, sizeY, srcPitch, sizeX);

                            // execute algorithm
                            ArtPointillismAlgorithmExec(algoMemHandler, *pFilterStrParams, sizeX, sizeY);

                            // back convert to native buffer format after processing complete
                            AVX2_ConvertCIELab_SemiPlanar_ToRgb(input_pixels, dstL, dstAB, output_pixels, sizeX, sizeY, srcPitch, dstPitch);
                        }
                        break;

                        default:
                            err = PF_Err_BAD_CALLBACK_PARAM;
                        break;
                    } // switch (format)

                } // if (PF_Err_NONE == wsP->PF_GetPixelFormat(input_worldP, &format))

            } //  if (true == mem_handler_valid(algoMemHandler))

        } // if (nullptr != input_worldP && nullptr != output_worldP)

        handleSuite->host_unlock_handle(reinterpret_cast<PF_Handle>(extraP->input->pre_render_data));

    } // if (nullptr != pFilterStrParams)

    return PF_Err_NONE;
}