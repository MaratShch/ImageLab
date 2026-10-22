#include "GuidedFilter.hpp"
#include "GuidedFilterEnum.hpp"
#include "CommonSmartRender.hpp"
#include "AlgoControls.hpp"
#include "AlgoMemHandler.hpp"
#include "AlgorithmMain.hpp"
#include "AlgoColorDispatcher.hpp"
#include "AlgoColorDispatcherOut.hpp"

using PAlgoControls = AlgoControls*;

PF_Err
GuidedFilter_PreRender
(
    PF_InData			*in_data,
    PF_OutData			*out_data,
    PF_PreRenderExtra	*extra
)
{
    PF_Err err = PF_Err_NONE;

    AEFX_SuiteScoper<PF_HandleSuite1> handleSuite = AEFX_SuiteScoper<PF_HandleSuite1>(in_data, kPFHandleSuite, kPFHandleSuiteVersion1, out_data);
    PF_Handle paramsHandler = handleSuite->host_new_handle(static_cast<A_HandleSize>(gAlgoControlsStrSize));

    if (nullptr != paramsHandler)
    {
        PAlgoControls paramsStrP = reinterpret_cast<PAlgoControls>(handleSuite->host_lock_handle(paramsHandler));
        if (nullptr != paramsStrP)
        {
            CACHE_ALIGN PF_ParamDef paramVal[6]{};
            extra->output->pre_render_data = paramsHandler;

            const A_long   current_time = in_data->current_time;
            const A_long   time_step    = in_data->time_step;
            const A_u_long time_scale   = in_data->time_scale;

            PF_CHECKOUT_PARAM(in_data, UnderlyingType(CtrlItems::gFILTER_MODE),     current_time, time_step, time_scale, &paramVal[0]);
            PF_CHECKOUT_PARAM(in_data, UnderlyingType(CtrlItems::gFILTER_RADIUS),   current_time, time_step, time_scale, &paramVal[1]);
            PF_CHECKOUT_PARAM(in_data, UnderlyingType(CtrlItems::gFILTER_EPSILON),  current_time, time_step, time_scale, &paramVal[2]);
            PF_CHECKOUT_PARAM(in_data, UnderlyingType(CtrlItems::gFILTER_SHOW_MAP), current_time, time_step, time_scale, &paramVal[3]);
            PF_CHECKOUT_PARAM(in_data, UnderlyingType(CtrlItems::gFILTER_SKIN_TOLERANCE), current_time, time_step, time_scale, &paramVal[4]);
            PF_CHECKOUT_PARAM(in_data, UnderlyingType(CtrlItems::gFILTER_SKIN_SOFTNESS),  current_time, time_step, time_scale, &paramVal[5]);

            paramsStrP->filterMode = static_cast<FilterMode>(paramVal[0].u.pd.value - 1);
            paramsStrP->radius = paramVal[1].u.sd.value;
            paramsStrP->epsilon = static_cast<float>(paramVal[2].u.fs_d.value);
            paramsStrP->showConfidenceMap = paramVal[3].u.bd.value;
            paramsStrP->skinTolerance = static_cast<float>(paramVal[4].u.fs_d.value);
            paramsStrP->skinSoftness  = static_cast<float>(paramVal[5].u.fs_d.value);

            PF_RenderRequest req = extra->input->output_request;
            PF_CheckoutResult in_result{};

            ERR(extra->cb->checkout_layer
                (in_data->effect_ref,
                    UnderlyingType(CtrlItems::gFILTER_INPUT),
                    UnderlyingType(CtrlItems::gFILTER_INPUT),
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
GuidedFilter_SmartRender
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
        ERR((extraP->cb->checkout_layer_pixels(in_data->effect_ref, UnderlyingType(CtrlItems::gFILTER_INPUT), &input_worldP)));
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
                if (PF_Err_NONE == wsP->PF_GetPixelFormat(input_worldP, &format))
                {
                    switch (format)
                    {
                        case PF_PixelFormat_ARGB128:
                        {
                            const A_long srcPitch = srcRowBytes / static_cast<A_long>(PF_Pixel_ARGB_32f_size);
                            const A_long dstPitch = dstRowBytes / static_cast<A_long>(PF_Pixel_ARGB_32f_size);

                            const PF_Pixel_ARGB_32f* RESTRICT localSrc = reinterpret_cast<const PF_Pixel_ARGB_32f* RESTRICT>(input_worldP->data);
                                  PF_Pixel_ARGB_32f* RESTRICT localDst = reinterpret_cast<      PF_Pixel_ARGB_32f* RESTRICT>(output_worldP->data);

                            dispatch_convert_to_planar (localSrc, algoMemHandler, sizeX, sizeY, srcPitch, PixelFormat::ARGB_32f);
                            Algorithm_Main (algoMemHandler, sizeX, sizeY, *pFilterStrParams);
                            dispatch_convert_to_interleaved (algoMemHandler, localSrc, localDst, sizeX, sizeY, srcPitch, dstPitch, PixelFormat::ARGB_32f);
                        }
                        break;

                        case PF_PixelFormat_ARGB64:
                        {
                            const A_long srcPitch = srcRowBytes / static_cast<A_long>(PF_Pixel_ARGB_16u_size);
                            const A_long dstPitch = dstRowBytes / static_cast<A_long>(PF_Pixel_ARGB_16u_size);

                            const PF_Pixel_ARGB_16u* RESTRICT localSrc = reinterpret_cast<const PF_Pixel_ARGB_16u* RESTRICT>(input_worldP->data);
                                  PF_Pixel_ARGB_16u* RESTRICT localDst = reinterpret_cast<      PF_Pixel_ARGB_16u* RESTRICT>(output_worldP->data);

                            dispatch_convert_to_planar (localSrc, algoMemHandler, sizeX, sizeY, srcPitch, PixelFormat::ARGB_16u);
                            Algorithm_Main (algoMemHandler, sizeX, sizeY, *pFilterStrParams);
                            dispatch_convert_to_interleaved (algoMemHandler, localSrc, localDst, sizeX, sizeY, srcPitch, dstPitch, PixelFormat::ARGB_16u);
                        }
                        break;

                        case PF_PixelFormat_ARGB32:
                        {
                            const A_long srcPitch = srcRowBytes / static_cast<A_long>(PF_Pixel_ARGB_8u_size);
                            const A_long dstPitch = dstRowBytes / static_cast<A_long>(PF_Pixel_ARGB_8u_size);

                            const PF_Pixel_ARGB_8u* RESTRICT localSrc = reinterpret_cast<const PF_Pixel_ARGB_8u* RESTRICT>(input_worldP->data);
                                  PF_Pixel_ARGB_8u* RESTRICT localDst = reinterpret_cast<      PF_Pixel_ARGB_8u* RESTRICT>(output_worldP->data);

                            dispatch_convert_to_planar (localSrc, algoMemHandler, sizeX, sizeY, srcPitch, PixelFormat::ARGB_8u);
                            Algorithm_Main (algoMemHandler, sizeX, sizeY, *pFilterStrParams);
                            dispatch_convert_to_interleaved (algoMemHandler, localSrc, localDst, sizeX, sizeY, srcPitch, dstPitch, PixelFormat::ARGB_8u);
                        }
                        break;

                        default:
                            err = PF_Err_BAD_CALLBACK_PARAM;
                        break;
                    } // switch (format)

                } // if (PF_Err_NONE == wsP->PF_GetPixelFormat(input_worldP, &format))

                  // free (return back to memory manager) already non used buffer
                free_memory_buffers(algoMemHandler);

            } //  if (true == mem_handler_valid(algoMemHandler))

        } // if (nullptr != input_worldP && nullptr != output_worldP)

        handleSuite->host_unlock_handle(reinterpret_cast<PF_Handle>(extraP->input->pre_render_data));

    } // if (nullptr != pFilterStrParams)

    return err;
}