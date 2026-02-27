#include "ImageLabDenoise.hpp"
#include "ImageLabDenoiseEnum.hpp"
#include "CommonSmartRender.hpp"
#include "ImageLabMemInterface.hpp"
#include "AlgoControls.hpp"
#include "CommonSmartRender.hpp"

using PAlgoControls = AlgoControls*;

PF_Err
ImageLabDenoise_PreRender
(
    PF_InData			*in_data,
    PF_OutData			*out_data,
    PF_PreRenderExtra	*extra
)
{
    CACHE_ALIGN AlgoControls sRenderParam{};
    PF_Err err = PF_Err_NONE;

    AEFX_SuiteScoper<PF_HandleSuite1> handleSuite = AEFX_SuiteScoper<PF_HandleSuite1>(in_data, kPFHandleSuite, kPFHandleSuiteVersion1, out_data);
    PF_Handle paramsHandler = handleSuite->host_new_handle(sizeof(sRenderParam));

    if (nullptr != paramsHandler)
    {
        PAlgoControls paramsStrP = reinterpret_cast<PAlgoControls>(handleSuite->host_lock_handle(paramsHandler));
        if (nullptr != paramsStrP)
        {
            CACHE_ALIGN PF_ParamDef paramVal{};

            extra->output->pre_render_data = paramsHandler;

            // ============= Acquire algorithm control parameters ==================== //
            PF_CHECKOUT_PARAM (in_data, UnderlyingType(eDenoiseControl::eIMAGE_LAB_DENOISE_ACC_SANDARD), in_data->current_time, in_data->time_step, in_data->time_scale, &paramVal);
            sRenderParam.accuracy = static_cast<ProcAccuracy>(paramVal.u.pd.value - 1);

            PF_CHECKOUT_PARAM(in_data, UnderlyingType(eDenoiseControl::eIMAGE_LAB_DENOISE_AMOUNT), in_data->current_time, in_data->time_step, in_data->time_scale, &paramVal);
            sRenderParam.master_denoise_amount = static_cast<float>(paramVal.u.fs_d.value);

            PF_CHECKOUT_PARAM(in_data, UnderlyingType(eDenoiseControl::eIMAGE_LAB_DENOISE_LUMA_STRENGTH), in_data->current_time, in_data->time_step, in_data->time_scale, &paramVal);
            sRenderParam.luma_strength = static_cast<float>(paramVal.u.fs_d.value);

            PF_CHECKOUT_PARAM(in_data, UnderlyingType(eDenoiseControl::eIMAGE_LAB_DENOISE_CHROMA_STRENGTH), in_data->current_time, in_data->time_step, in_data->time_scale, &paramVal);
            sRenderParam.chroma_strength = static_cast<float>(paramVal.u.fs_d.value);

            PF_CHECKOUT_PARAM(in_data, UnderlyingType(eDenoiseControl::eIMAGE_LAB_DENOISE_DETAILS_PRESERVATION), in_data->current_time, in_data->time_step, in_data->time_scale, &paramVal);
            sRenderParam.fine_detail_preservation = static_cast<float>(paramVal.u.fs_d.value);

            PF_CHECKOUT_PARAM(in_data, UnderlyingType(eDenoiseControl::eIMAGE_LAB_DENOISE_COARSE_NOISE), in_data->current_time, in_data->time_step, in_data->time_scale, &paramVal);
            sRenderParam.coarse_noise_reduction = static_cast<float>(paramVal.u.fs_d.value);

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
    PF_Err err = PF_Err_NONE;
    return err;
}
