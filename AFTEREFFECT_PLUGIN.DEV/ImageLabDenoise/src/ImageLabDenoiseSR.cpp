#include "ImageLabDenoise.hpp"
#include "ImageLabDenoiseEnum.hpp"
#include "CommonSmartRender.hpp"
#include "AlgoMemHandler.hpp"
#include "AlgoControls.hpp"
#include "CommonSmartRender.hpp"
#include "ColorConvert.hpp"
#include "AlgorithmMain.hpp"


using PAlgoControls = AlgoControls*;

PF_Err
ImageLabDenoise_PreRender
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
        PAlgoControls paramsStrP = reinterpret_cast<PAlgoControls>(handleSuite->host_lock_handle(paramsHandler));
        if (nullptr != paramsStrP)
        {
            CACHE_ALIGN PF_ParamDef paramVal{};

            extra->output->pre_render_data = paramsHandler;

            // ============= Acquire algorithm control parameters ==================== //
            PF_CHECKOUT_PARAM (in_data, UnderlyingType(eDenoiseControl::eIMAGE_LAB_DENOISE_ACC_SANDARD), in_data->current_time, in_data->time_step, in_data->time_scale, &paramVal);
                paramsStrP->accuracy = static_cast<ProcAccuracy>(paramVal.u.pd.value - 1);

            PF_CHECKOUT_PARAM(in_data, UnderlyingType(eDenoiseControl::eIMAGE_LAB_DENOISE_AMOUNT), in_data->current_time, in_data->time_step, in_data->time_scale, &paramVal);
                paramsStrP->master_denoise_amount = static_cast<float>(paramVal.u.fs_d.value);

            PF_CHECKOUT_PARAM(in_data, UnderlyingType(eDenoiseControl::eIMAGE_LAB_DENOISE_LUMA_STRENGTH), in_data->current_time, in_data->time_step, in_data->time_scale, &paramVal);
                paramsStrP->luma_strength = static_cast<float>(paramVal.u.fs_d.value);

            PF_CHECKOUT_PARAM(in_data, UnderlyingType(eDenoiseControl::eIMAGE_LAB_DENOISE_CHROMA_STRENGTH), in_data->current_time, in_data->time_step, in_data->time_scale, &paramVal);
                paramsStrP->chroma_strength = static_cast<float>(paramVal.u.fs_d.value);

            PF_CHECKOUT_PARAM(in_data, UnderlyingType(eDenoiseControl::eIMAGE_LAB_DENOISE_DETAILS_PRESERVATION), in_data->current_time, in_data->time_step, in_data->time_scale, &paramVal);
                paramsStrP->fine_detail_preservation = static_cast<float>(paramVal.u.fs_d.value);

            PF_CHECKOUT_PARAM(in_data, UnderlyingType(eDenoiseControl::eIMAGE_LAB_DENOISE_COARSE_NOISE), in_data->current_time, in_data->time_step, in_data->time_scale, &paramVal);
                paramsStrP->coarse_noise_reduction = static_cast<float>(paramVal.u.fs_d.value);

            PF_RenderRequest req = extra->input->output_request;
            PF_CheckoutResult in_result{};

            ERR(extra->cb->checkout_layer
            (in_data->effect_ref,
                UnderlyingType(eDenoiseControl::eIMAGE_LAB_DENOISE_INPUT),
                UnderlyingType(eDenoiseControl::eIMAGE_LAB_DENOISE_INPUT),
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
ImageLabDenoise_SmartRender
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
    const AlgoControls* pFilterStrParams = reinterpret_cast<const AlgoControls*>(handleSuite->host_lock_handle(reinterpret_cast<PF_Handle>(extraP->input->pre_render_data)));

    if (nullptr != pFilterStrParams)
    {
        ERR((extraP->cb->checkout_layer_pixels(in_data->effect_ref, UnderlyingType(eDenoiseControl::eIMAGE_LAB_DENOISE_INPUT), &input_worldP)));
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

                        const PF_Pixel_ARGB_32f* __restrict input_pixels  = reinterpret_cast<const PF_Pixel_ARGB_32f* __restrict>(input_worldP->data);
                              PF_Pixel_ARGB_32f* __restrict output_pixels = reinterpret_cast<      PF_Pixel_ARGB_32f* __restrict>(output_worldP->data);

                        // convert to planar YUV Ortonormal color space
                        AVX2_Convert_ARGB_32f_YUV (input_pixels, algoMemHandler.Y_planar, algoMemHandler.U_planar, algoMemHandler.V_planar, sizeX, sizeY, srcPitch);

                        // execute algorithm
                        Algorithm_Main (algoMemHandler, sizeX, sizeY, *pFilterStrParams);

                        // back convert to native buffer format after processing complete
                        AVX2_Convert_YUV_to_ARGB_32f(algoMemHandler.Accum_Y, algoMemHandler.Accum_U, algoMemHandler.Accum_V, input_pixels, output_pixels, sizeX, sizeY, srcPitch, dstPitch);
                    }
                    break;

                    case PF_PixelFormat_ARGB64:
                    {
                        const A_long srcPitch = srcRowBytes / static_cast<A_long>(PF_Pixel_ARGB_16u_size);
                        const A_long dstPitch = dstRowBytes / static_cast<A_long>(PF_Pixel_ARGB_16u_size);

                        const PF_Pixel_ARGB_16u* __restrict input_pixels  = reinterpret_cast<const PF_Pixel_ARGB_16u* __restrict>(input_worldP->data);
                              PF_Pixel_ARGB_16u* __restrict output_pixels = reinterpret_cast<      PF_Pixel_ARGB_16u* __restrict>(output_worldP->data);

                        // convert to planar YUV Ortonormal color space
                        AVX2_Convert_ARGB_16u_YUV (input_pixels, algoMemHandler.Y_planar, algoMemHandler.U_planar, algoMemHandler.V_planar, sizeX, sizeY, srcPitch);

                        // execute algorithm
                        Algorithm_Main (algoMemHandler, sizeX, sizeY, *pFilterStrParams);

                        // back convert to native buffer format after processing complete
                        AVX2_Convert_YUV_to_ARGB_16u (algoMemHandler.Accum_Y, algoMemHandler.Accum_U, algoMemHandler.Accum_V, input_pixels, output_pixels, sizeX, sizeY, srcPitch, dstPitch);
                    }
                    break;

                    case PF_PixelFormat_ARGB32:
                    {
                        const A_long srcPitch = srcRowBytes / static_cast<A_long>(PF_Pixel_ARGB_8u_size);
                        const A_long dstPitch = dstRowBytes / static_cast<A_long>(PF_Pixel_ARGB_8u_size);

                        const PF_Pixel_ARGB_8u* __restrict input_pixels  = reinterpret_cast<const PF_Pixel_ARGB_8u* __restrict>(input_worldP->data);
                              PF_Pixel_ARGB_8u* __restrict output_pixels = reinterpret_cast<      PF_Pixel_ARGB_8u* __restrict>(output_worldP->data);

                        // convert to planar YUV Ortonormal color space
                        AVX2_Convert_ARGB_8u_YUV (input_pixels, algoMemHandler.Y_planar, algoMemHandler.U_planar, algoMemHandler.V_planar, sizeX, sizeY, srcPitch);

                        // execute algorithm
                        Algorithm_Main (algoMemHandler, sizeX, sizeY, *pFilterStrParams);

                        // back convert to native buffer format after processing complete
                        AVX2_Convert_YUV_to_ARGB_8u (algoMemHandler.Accum_Y, algoMemHandler.Accum_U, algoMemHandler.Accum_V, input_pixels, output_pixels, sizeX, sizeY, srcPitch, dstPitch);
                    }
                    break;

                    default:
                        err = PF_Err_BAD_CALLBACK_PARAM;
                        break;
                    } // switch (format)

                } // if (PF_Err_NONE == wsP->PF_GetPixelFormat(input_worldP, &format))

                // free (return back to memory manager) already non used buffer
                free_memory_buffers (algoMemHandler);

            } //  if (true == mem_handler_valid(algoMemHandler))

        } // if (nullptr != input_worldP && nullptr != output_worldP)

        handleSuite->host_unlock_handle(reinterpret_cast<PF_Handle>(extraP->input->pre_render_data));

    } // if (nullptr != pFilterStrParams)

    return PF_Err_NONE;
}
