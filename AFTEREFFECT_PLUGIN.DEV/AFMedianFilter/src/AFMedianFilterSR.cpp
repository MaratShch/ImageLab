#include "AFMedianFilter.hpp"
#include "AFMedianFilterEnum.hpp"
#include "CommonSmartRender.hpp"
#include "ImageLabMemInterface.hpp"
#include "AlgorithmMain.hpp"
#include "AlgoControls.hpp"
#include "ColorConvert.hpp"

PF_Err
AFMedianFilter_PreRender
(
    PF_InData			*in_data,
    PF_OutData			*out_data,
    PF_PreRenderExtra	*extra
) 
{
    PF_Err err = PF_Err_NONE;

    AEFX_SuiteScoper<PF_HandleSuite1> handleSuite = AEFX_SuiteScoper<PF_HandleSuite1>(in_data, kPFHandleSuite, kPFHandleSuiteVersion1, out_data);
    PF_Handle paramsHandler = handleSuite->host_new_handle(sizeof(AlgoControls));

    if (nullptr != paramsHandler)
    {
        AlgoControls* paramsStrP = reinterpret_cast<AlgoControls*>(handleSuite->host_lock_handle(paramsHandler));
        if (nullptr != paramsStrP)
        {
            CACHE_ALIGN PF_ParamDef paramVal[4]{};

            const A_long current_time = in_data->current_time;
            const A_long time_step  = in_data->time_step;
            const A_long time_scale = in_data->time_scale;

            PF_CHECKOUT_PARAM(in_data, UnderlyingType(AFMF::eIMAGE_AFMEDIAN_OUTPUT_TYPE), current_time, time_step, time_scale, &paramVal[0]);
            PF_CHECKOUT_PARAM(in_data, UnderlyingType(AFMF::eIMAGE_AFMEDIAN_PARAM_RADIUS), current_time, time_step, time_scale, &paramVal[1]);
            PF_CHECKOUT_PARAM(in_data, UnderlyingType(AFMF::eIMAGE_AFMEDIAN_PARAM_TOLERANCE), current_time, time_step, time_scale, &paramVal[2]);
            PF_CHECKOUT_PARAM(in_data, UnderlyingType(AFMF::eIMAGE_AFMEDIAN_PARAM_ITERATIONS), current_time, time_step, time_scale, &paramVal[3]);

            extra->output->pre_render_data = paramsHandler;

            paramsStrP->outputType = static_cast<AFMF_Output>(paramVal[0].u.pd.value - 1);
            paramsStrP->radius     = popup2value(static_cast<int32_t>(paramVal[1].u.pd.value - 1));
            paramsStrP->tolerance  = static_cast<float>(paramVal[2].u.fs_d.value / 10.0);
            paramsStrP->iterations = popup2value(static_cast<int32_t>(paramVal[3].u.pd.value - 1));

            PF_RenderRequest req = extra->input->output_request;
            PF_CheckoutResult in_result{};

            ERR(extra->cb->checkout_layer
            (in_data->effect_ref,
                UnderlyingType(AFMF::eIMAGE_AFMEDIAN_INPUT),
                UnderlyingType(AFMF::eIMAGE_AFMEDIAN_INPUT),
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
AFMedianFilter_SmartRender
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
    const AlgoControls* pAlgoStrParams = reinterpret_cast<const AlgoControls*>(handleSuite->host_lock_handle(reinterpret_cast<PF_Handle>(extraP->input->pre_render_data)));

    if (nullptr != pAlgoStrParams)
    {
        ERR((extraP->cb->checkout_layer_pixels(in_data->effect_ref, UnderlyingType(AFMF::eIMAGE_AFMEDIAN_INPUT), &input_worldP)));
        ERR(extraP->cb->checkout_output(in_data->effect_ref, &output_worldP));

        if (nullptr != input_worldP && nullptr != output_worldP)
        {
            const A_long sizeX = input_worldP->width;
            const A_long sizeY = input_worldP->height;
            const A_long srcRowBytes = input_worldP->rowbytes;  // Get input buffer pitch in bytes
            const A_long dstRowBytes = output_worldP->rowbytes; // Get output buffer pitch in bytes

            MemHandler algoMemHandler = alloc_memory_buffers(sizeX, sizeY);
            if (true == mem_handler_valid(algoMemHandler))
            {
                PF_PixelFormat format = PF_PixelFormat_INVALID;
                AEFX_SuiteScoper<PF_WorldSuite2> wsP = AEFX_SuiteScoper<PF_WorldSuite2>(in_data, kPFWorldSuite, kPFWorldSuiteVersion2, out_data);
                if (PF_Err_NONE == wsP->PF_GetPixelFormat(input_worldP, &format))
                {
                    switch (format)
                    {
                        case PF_PixelFormat_ARGB128:
                        {
                            const A_long srcPitch = srcRowBytes / static_cast<A_long>(PF_Pixel_ARGB_32f_size);
                            const A_long dstPitch = dstRowBytes / static_cast<A_long>(PF_Pixel_ARGB_32f_size);

                            const PF_Pixel_ARGB_32f* __restrict input_pixels  = reinterpret_cast<PF_Pixel_ARGB_32f* __restrict>(input_worldP->data);
                                  PF_Pixel_ARGB_32f* __restrict output_pixels = reinterpret_cast<PF_Pixel_ARGB_32f* __restrict>(output_worldP->data);

                            dispatch_convert_to_planar (input_pixels, algoMemHandler, sizeX, sizeY, srcPitch, PixelFormat::ARGB_32f);
                            Algorithm_Main (algoMemHandler, sizeX, sizeY, *pAlgoStrParams);
                            dispatch_convert_to_interleaved (algoMemHandler, input_pixels, output_pixels, sizeX, sizeY, srcPitch, dstPitch, PixelFormat::ARGB_32f);
                        }
                        break;

                        case PF_PixelFormat_ARGB64:
                        {
                            const A_long srcPitch = srcRowBytes / static_cast<A_long>(PF_Pixel_ARGB_16u_size);
                            const A_long dstPitch = dstRowBytes / static_cast<A_long>(PF_Pixel_ARGB_16u_size);

                            const PF_Pixel_ARGB_16u* __restrict input_pixels  = reinterpret_cast<PF_Pixel_ARGB_16u* __restrict>(input_worldP->data);
                                  PF_Pixel_ARGB_16u* __restrict output_pixels = reinterpret_cast<PF_Pixel_ARGB_16u* __restrict>(output_worldP->data);

                            dispatch_convert_to_planar (input_pixels, algoMemHandler, sizeX, sizeY, srcPitch, PixelFormat::ARGB_16u);
                            Algorithm_Main (algoMemHandler, sizeX, sizeY, *pAlgoStrParams);
                            dispatch_convert_to_interleaved (algoMemHandler, input_pixels, output_pixels, sizeX, sizeY, srcPitch, dstPitch, PixelFormat::ARGB_16u);
                        }
                        break;

                        case PF_PixelFormat_ARGB32:
                        {
                            const A_long srcPitch = srcRowBytes / static_cast<A_long>(PF_Pixel_ARGB_8u_size);
                            const A_long dstPitch = dstRowBytes / static_cast<A_long>(PF_Pixel_ARGB_8u_size);

                            const PF_Pixel_ARGB_8u* __restrict input_pixels  = reinterpret_cast<PF_Pixel_ARGB_8u* __restrict>(input_worldP->data);
                                  PF_Pixel_ARGB_8u* __restrict output_pixels = reinterpret_cast<PF_Pixel_ARGB_8u* __restrict>(output_worldP->data);

                            dispatch_convert_to_planar (input_pixels, algoMemHandler, sizeX, sizeY, srcPitch, PixelFormat::ARGB_8u);
                            Algorithm_Main (algoMemHandler, sizeX, sizeY, *pAlgoStrParams);
                            dispatch_convert_to_interleaved (algoMemHandler, input_pixels, output_pixels, sizeX, sizeY, srcPitch, dstPitch, PixelFormat::ARGB_8u);
                        }
                        break;

                        default:
                            err = PF_Err_BAD_CALLBACK_PARAM;
                        break;
                    } // switch (format)

                } // if (PF_Err_NONE == wsP->PF_GetPixelFormat(input_worldP, &format))
                else
                    err = PF_Err_INTERNAL_STRUCT_DAMAGED;

                free_memory_buffers(algoMemHandler);
            }
            else
                err = PF_Err_OUT_OF_MEMORY;

            ERR(extraP->cb->checkin_layer_pixels(in_data->effect_ref, UnderlyingType(AFMF::eIMAGE_AFMEDIAN_INPUT)));

        } // if (nullptr != input_worldP && nullptr != output_worldP)

        handleSuite->host_unlock_handle(reinterpret_cast<PF_Handle>(extraP->input->pre_render_data));

    } // if (nullptr != pFilterStrParams)
    else
        err = PF_Err_OUT_OF_MEMORY;

    return err;
}