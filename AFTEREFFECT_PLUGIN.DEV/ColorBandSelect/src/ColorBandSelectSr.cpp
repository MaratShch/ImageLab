#include "CommonSmartRender.hpp"
#include "ColorBandSelectEnums.hpp"
#include "ColorBandSelect.hpp"
#include "ColorBandSelectProc.hpp"


PF_Err
ColorBandSelect_PreRender
(
    PF_InData			*in_data,
    PF_OutData			*out_data,
    PF_PreRenderExtra	*extra
)
{
    PF_Err err = PF_Err_NONE;

    AEFX_SuiteScoper<PF_HandleSuite1> handleSuite = AEFX_SuiteScoper<PF_HandleSuite1>(in_data, kPFHandleSuite, kPFHandleSuiteVersion1, out_data);
    PF_Handle paramsHandler = handleSuite->host_new_handle(sizeof(int32_t));

    if (nullptr != paramsHandler)
    {
        int32_t* paramsStrP = reinterpret_cast<int32_t*>(handleSuite->host_lock_handle(paramsHandler));
        if (nullptr != paramsStrP)
        {
            CACHE_ALIGN PF_ParamDef paramVal[3]{};

            const A_long current_time = in_data->current_time;
            const A_long time_step  = in_data->time_step;
            const A_long time_scale = in_data->time_scale;

            PF_CHECKOUT_PARAM(in_data, COLOR_BAND_CHANNEL_RED,   current_time, time_step, time_scale, &paramVal[0]);
            PF_CHECKOUT_PARAM(in_data, COLOR_BAND_CHANNEL_GREEN, current_time, time_step, time_scale, &paramVal[1]);
            PF_CHECKOUT_PARAM(in_data, COLOR_BAND_CHANNEL_BLUE,  current_time, time_step, time_scale, &paramVal[2]);

            extra->output->pre_render_data = paramsHandler;

                                        // RED                            // GREEN                        // BLUE
            const int32_t colorMask = (paramVal[0].u.bd.value << 0) | (paramVal[1].u.bd.value << 1) | ((paramVal[2].u.bd.value << 2));
            *paramsStrP = colorMask;

            PF_RenderRequest req = extra->input->output_request;
            PF_CheckoutResult in_result{};

            ERR(extra->cb->checkout_layer
            (in_data->effect_ref,
                COLOR_BAND_FILTER_INPUT,
                COLOR_BAND_FILTER_INPUT,
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
ColorBandSelect_SmartRender
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
    const int32_t* pAlgoStrParams = reinterpret_cast<const int32_t*>(handleSuite->host_lock_handle(reinterpret_cast<PF_Handle>(extraP->input->pre_render_data)));

    if (nullptr != pAlgoStrParams)
    {
        ERR((extraP->cb->checkout_layer_pixels(in_data->effect_ref, COLOR_BAND_FILTER_INPUT, &input_worldP)));
        ERR(extraP->cb->checkout_output(in_data->effect_ref, &output_worldP));

        if (nullptr != input_worldP && nullptr != output_worldP)
        {
            const A_long sizeX = input_worldP->width;
            const A_long sizeY = input_worldP->height;
            const A_long srcRowBytes = input_worldP->rowbytes;  // Get input buffer pitch in bytes
            const A_long dstRowBytes = output_worldP->rowbytes; // Get output buffer pitch in bytes

            PF_PixelFormat format = PF_PixelFormat_INVALID;
            AEFX_SuiteScoper<PF_WorldSuite2> wsP = AEFX_SuiteScoper<PF_WorldSuite2>(in_data, kPFWorldSuite, kPFWorldSuiteVersion2, out_data);
            
            if (PF_Err_NONE == wsP->PF_GetPixelFormat(input_worldP, &format))
            {
                const int32_t ColorMask = *pAlgoStrParams & 0x07;
                const A_long ChannelR = ColorMask & 0x01 ? 1 : 0;
                const A_long ChannelG = ColorMask & 0x02 ? 1 : 0;
                const A_long ChannelB = ColorMask & 0x04 ? 1 : 0;

                if (0 != ChannelR && 0 != ChannelG && 0 != ChannelB)
                {
                    auto const& worldTransformSuite{ AEFX_SuiteScoper<PF_WorldTransformSuite1>(in_data, kPFWorldTransformSuite, kPFWorldTransformSuiteVersion1, out_data) };
                    err = worldTransformSuite->copy(in_data->effect_ref, input_worldP, output_worldP, NULL, NULL);
                }
                else
                {
                    switch (format)
                    {
                        case PF_PixelFormat_ARGB128:
                        {
                            const A_long srcPitch = srcRowBytes / static_cast<A_long>(PF_Pixel_ARGB_32f_size);
                            const A_long dstPitch = dstRowBytes / static_cast<A_long>(PF_Pixel_ARGB_32f_size);

                            const PF_Pixel_ARGB_32f* __restrict input_pixels  = reinterpret_cast<PF_Pixel_ARGB_32f* __restrict>(input_worldP->data);
                                  PF_Pixel_ARGB_32f* __restrict output_pixels = reinterpret_cast<PF_Pixel_ARGB_32f* __restrict>(output_worldP->data);

                            ImgCopyByChannelMask (input_pixels, output_pixels, srcPitch, dstPitch, sizeX, sizeY, ChannelR, ChannelG, ChannelB);
                        }
                        break;

                        case PF_PixelFormat_ARGB64:
                        {
                            const A_long srcPitch = srcRowBytes / static_cast<A_long>(PF_Pixel_ARGB_16u_size);
                            const A_long dstPitch = dstRowBytes / static_cast<A_long>(PF_Pixel_ARGB_16u_size);

                            const PF_Pixel_ARGB_16u* __restrict input_pixels  = reinterpret_cast<PF_Pixel_ARGB_16u* __restrict>(input_worldP->data);
                                  PF_Pixel_ARGB_16u* __restrict output_pixels = reinterpret_cast<PF_Pixel_ARGB_16u* __restrict>(output_worldP->data);

                            ImgCopyByChannelMask (input_pixels, output_pixels, srcPitch, dstPitch, sizeX, sizeY, ChannelR, ChannelG, ChannelB);
                        }
                        break;

                        case PF_PixelFormat_ARGB32:
                        {
                            const A_long srcPitch = srcRowBytes / static_cast<A_long>(PF_Pixel_ARGB_8u_size);
                            const A_long dstPitch = dstRowBytes / static_cast<A_long>(PF_Pixel_ARGB_8u_size);

                            const PF_Pixel_ARGB_8u* __restrict input_pixels  = reinterpret_cast<PF_Pixel_ARGB_8u* __restrict>(input_worldP->data);
                                  PF_Pixel_ARGB_8u* __restrict output_pixels = reinterpret_cast<PF_Pixel_ARGB_8u* __restrict>(output_worldP->data);

                            ImgCopyByChannelMask (input_pixels, output_pixels, srcPitch, dstPitch, sizeX, sizeY, ChannelR, ChannelG, ChannelB);
                        }
                        break;

                        default:
                            err = PF_Err_BAD_CALLBACK_PARAM;
                        break;
                    } // switch (format)
                }

            } // if (PF_Err_NONE == wsP->PF_GetPixelFormat(input_worldP, &format))
            else
                err = PF_Err_INTERNAL_STRUCT_DAMAGED;

            ERR(extraP->cb->checkin_layer_pixels(in_data->effect_ref, COLOR_BAND_FILTER_INPUT));

        } // if (nullptr != input_worldP && nullptr != output_worldP)

        handleSuite->host_unlock_handle(reinterpret_cast<PF_Handle>(extraP->input->pre_render_data));

    } // if (nullptr != pFilterStrParams)
    else
        err = PF_Err_OUT_OF_MEMORY;

    return err;
}