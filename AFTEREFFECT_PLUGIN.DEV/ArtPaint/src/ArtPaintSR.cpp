#include "ArtPaint.hpp"
#include "ArtPaintEnums.hpp"
#include "PaintMemHandler.hpp"
#include "PaintColorDispatcher.hpp"
#include "PaintColorDispatcherOut.hpp"
#include "PaintAlgoMain.hpp"
#include "CommonSmartRender.hpp"
#include "ImageLabMemInterface.hpp"


PF_Err
ArtPaint_PreRender
(
    PF_InData			*in_data,
    PF_OutData			*out_data,
    PF_PreRenderExtra	*extra
)
{
    PF_Err err = PF_Err_NONE;

    AEFX_SuiteScoper<PF_HandleSuite1> handleSuite = AEFX_SuiteScoper<PF_HandleSuite1>(in_data, kPFHandleSuite, kPFHandleSuiteVersion1, out_data);
    PF_Handle paramsHandler = handleSuite->host_new_handle(AlgoControlsSize);

    if (nullptr != paramsHandler)
    {
        PAlgoControls paramsStrP = reinterpret_cast<PAlgoControls>(handleSuite->host_lock_handle(paramsHandler));
        if (nullptr != paramsStrP)
        {
            CACHE_ALIGN PF_ParamDef paramVal[6]{};

            const A_long current_time = in_data->current_time;
            const A_long time_step    = in_data->time_step;
            const A_long time_scale   = in_data->time_scale;

            PF_CHECKOUT_PARAM(in_data, UnderlyingType(ArtPaintControls::ART_PAINT_RENDER_QUALITY),   current_time, time_step, time_scale, &paramVal[0]);
            PF_CHECKOUT_PARAM(in_data, UnderlyingType(ArtPaintControls::ART_PAINT_STYLE),            current_time, time_step, time_scale, &paramVal[1]);
            PF_CHECKOUT_PARAM(in_data, UnderlyingType(ArtPaintControls::ART_PAINT_BRUSH_WIDTH),      current_time, time_step, time_scale, &paramVal[2]);
            PF_CHECKOUT_PARAM(in_data, UnderlyingType(ArtPaintControls::ART_PAINT_BRUSH_LENGTH),     current_time, time_step, time_scale, &paramVal[3]);
            PF_CHECKOUT_PARAM(in_data, UnderlyingType(ArtPaintControls::ART_PAINT_STROKE_CURVATIVE), current_time, time_step, time_scale, &paramVal[4]);
            PF_CHECKOUT_PARAM(in_data, UnderlyingType(ArtPaintControls::ART_PAINT_STROKE_SPREADING), current_time, time_step, time_scale, &paramVal[5]);

            extra->output->pre_render_data = paramsHandler;

            paramsStrP->quality = static_cast<RenderQuality>(paramVal[0].u.pd.value - 1);
            paramsStrP->bias    = static_cast<StrokeBias>(paramVal[1].u.pd.value - 1);
            paramsStrP->sigma   = static_cast<float>(paramVal[2].u.fs_d.value);
            paramsStrP->angular = static_cast<float>(paramVal[3].u.fs_d.value);
            paramsStrP->angle   = static_cast<float>(paramVal[4].u.fs_d.value);
            paramsStrP->iter    = static_cast<int32_t>(paramVal[5].u.sd.value);

            PF_RenderRequest req = extra->input->output_request;
            PF_CheckoutResult in_result{};

            ERR(extra->cb->checkout_layer
            (in_data->effect_ref,
                UnderlyingType(ArtPaintControls::ART_PAINT_INPUT),
                UnderlyingType(ArtPaintControls::ART_PAINT_INPUT),
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
ArtPaint_SmartRender
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
    const PAlgoControls pFilterStrParams = reinterpret_cast<const PAlgoControls>(handleSuite->host_lock_handle(reinterpret_cast<PF_Handle>(extraP->input->pre_render_data)));

    if (nullptr != pFilterStrParams)
    {
        ERR((extraP->cb->checkout_layer_pixels(in_data->effect_ref, UnderlyingType(ArtPaintControls::ART_PAINT_INPUT), &input_worldP)));
        ERR(extraP->cb->checkout_output(in_data->effect_ref, &output_worldP));

        if (nullptr != input_worldP && nullptr != output_worldP)
        {
            const A_long sizeX = input_worldP->width;
            const A_long sizeY = input_worldP->height;
            const A_long srcRowBytes = input_worldP->rowbytes;  // Get input buffer pitch in bytes
            const A_long dstRowBytes = output_worldP->rowbytes; // Get output buffer pitch in bytes

            PF_PixelFormat format = PF_PixelFormat_INVALID;
            AEFX_SuiteScoper<PF_WorldSuite2> wsP = AEFX_SuiteScoper<PF_WorldSuite2>(in_data, kPFWorldSuite, kPFWorldSuiteVersion2, out_data);

            MemHandler algoMemHandler = alloc_memory_buffers(sizeX, sizeY, pFilterStrParams->quality);
            if (true == check_memory_buffers(algoMemHandler))
            {
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

                            dispatch_convert_to_planar (input_pixels, algoMemHandler, sizeX, sizeY, srcPitch, PixelFormat::ARGB_32f, pFilterStrParams->quality);
                            PaintAlgorithmMain (algoMemHandler, *pFilterStrParams);
                            dispatch_convert_to_interleaved (algoMemHandler, input_pixels, output_pixels, sizeX, sizeY, srcPitch, dstPitch, PixelFormat::ARGB_32f, pFilterStrParams->quality);
                        }
                        break;

                        case PF_PixelFormat_ARGB64:
                        {
                            const A_long srcPitch = srcRowBytes / static_cast<A_long>(PF_Pixel_ARGB_16u_size);
                            const A_long dstPitch = dstRowBytes / static_cast<A_long>(PF_Pixel_ARGB_16u_size);

                            const PF_Pixel_ARGB_16u* __restrict input_pixels  = reinterpret_cast<const PF_Pixel_ARGB_16u* __restrict>(input_worldP->data);
                                  PF_Pixel_ARGB_16u* __restrict output_pixels = reinterpret_cast<      PF_Pixel_ARGB_16u* __restrict>(output_worldP->data);

                            dispatch_convert_to_planar (input_pixels, algoMemHandler, sizeX, sizeY, srcPitch, PixelFormat::ARGB_16u, pFilterStrParams->quality);
                            PaintAlgorithmMain (algoMemHandler, *pFilterStrParams);
                            dispatch_convert_to_interleaved (algoMemHandler, input_pixels, output_pixels, sizeX, sizeY, srcPitch, dstPitch, PixelFormat::ARGB_16u, pFilterStrParams->quality);
                        }
                        break;

                        case PF_PixelFormat_ARGB32:
                        {
                            const A_long srcPitch = srcRowBytes / static_cast<A_long>(PF_Pixel_ARGB_8u_size);
                            const A_long dstPitch = dstRowBytes / static_cast<A_long>(PF_Pixel_ARGB_8u_size);

                            const PF_Pixel_ARGB_8u* __restrict input_pixels  = reinterpret_cast<const PF_Pixel_ARGB_8u* __restrict>(input_worldP->data);
                                  PF_Pixel_ARGB_8u* __restrict output_pixels = reinterpret_cast<      PF_Pixel_ARGB_8u* __restrict>(output_worldP->data);

                            dispatch_convert_to_planar (input_pixels, algoMemHandler, sizeX, sizeY, srcPitch, PixelFormat::ARGB_8u, pFilterStrParams->quality);
                            PaintAlgorithmMain (algoMemHandler, *pFilterStrParams);
                            dispatch_convert_to_interleaved (algoMemHandler, input_pixels, output_pixels, sizeX, sizeY, srcPitch, dstPitch, PixelFormat::ARGB_8u, pFilterStrParams->quality);
                        }
                        break;

                        default:
                            err = PF_Err_BAD_CALLBACK_PARAM;
                        break;
                    } // switch (format)

                } // if (PF_Err_NONE == wsP->PF_GetPixelFormat(input_worldP, &format))

                  // free (return back to memory manager) already non used buffer
                free_memory_buffers(algoMemHandler);

            } //  if (true == check_memory_buffers(algoMemHandler))

        } // if (nullptr != input_worldP && nullptr != output_worldP)

        handleSuite->host_unlock_handle(reinterpret_cast<PF_Handle>(extraP->input->pre_render_data));

    } // if (nullptr != pFilterStrParams)

    return err;
}