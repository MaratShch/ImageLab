#include "AuthomaticWhiteBalance.hpp"
#include "FastAriphmetics.hpp"
#include "ImageLabMemInterface.hpp"
#include "CompileTimeUtils.hpp"
#include "AlgCommonFunctions.hpp"
#include "AlgCorrectionMatrix.hpp"
#include "CommonSmartRender.hpp"
#include "AE_Effect.h"


PF_Err
AuthomaticWhiteBalance_PreRender
(
    PF_InData			*in_data,
    PF_OutData			*out_data,
    PF_PreRenderExtra	*extra
) 
{
    AWB_SmartRenderParams renderParams{};
    PF_Err err = PF_Err_NONE;

    AEFX_SuiteScoper<PF_HandleSuite1> handleSuite = AEFX_SuiteScoper<PF_HandleSuite1>(in_data, kPFHandleSuite, kPFHandleSuiteVersion1, out_data);
    PF_Handle paramsHandler = handleSuite->host_new_handle(AWB_SmartRenderParamsSize);
    if (nullptr != paramsHandler)
    {
        PAWB_SmartRenderParams paramsStrP = reinterpret_cast<PAWB_SmartRenderParams>(handleSuite->host_lock_handle(paramsHandler));
        if (nullptr != paramsStrP)
        {
            extra->output->pre_render_data = paramsHandler;

            PF_ParamDef param_Illuminant{};
            PF_ParamDef	param_Chromatic{};
            PF_ParamDef param_ColorSpace{};
            PF_ParamDef param_GrayThreshold{};
            PF_ParamDef param_IterationsNumber{};

            const PF_Err errParam1 = PF_CHECKOUT_PARAM(in_data, AWB_ILLUMINATE_POPUP,  in_data->current_time, in_data->time_step, in_data->time_scale, &param_Illuminant);
            const PF_Err errParam2 = PF_CHECKOUT_PARAM(in_data, AWB_CHROMATIC_POPUP,   in_data->current_time, in_data->time_step, in_data->time_scale, &param_Chromatic);
            const PF_Err errParam3 = PF_CHECKOUT_PARAM(in_data, AWB_COLOR_SPACE_POPUP, in_data->current_time, in_data->time_step, in_data->time_scale, &param_ColorSpace);
            const PF_Err errParam4 = PF_CHECKOUT_PARAM(in_data, AWB_THRESHOLD_SLIDER,  in_data->current_time, in_data->time_step, in_data->time_scale, &param_GrayThreshold);
            const PF_Err errParam5 = PF_CHECKOUT_PARAM(in_data, AWB_ITERATIONS_SLIDER, in_data->current_time, in_data->time_step, in_data->time_scale, &param_IterationsNumber);

            if (PF_Err_NONE == errParam1 && PF_Err_NONE == errParam2 && PF_Err_NONE == errParam3 && PF_Err_NONE == errParam4 && PF_Err_NONE == errParam5)
            {
                paramsStrP->srParam_Illuminant        =  CLAMP_VALUE(static_cast<eILLUMINATE>(param_Illuminant.u.pd.value), DAYLIGHT, COOL_WHITE_FLUORESCENT);
                paramsStrP->srParam_ChromaticAdapt    =  CLAMP_VALUE(static_cast<eChromaticAdaptation>(param_Chromatic.u.pd.value), CHROMATIC_CAT02, CHROMATIC_CMCCAT2000);
                paramsStrP->srParam_ColorSpace        =  CLAMP_VALUE(static_cast<eCOLOR_SPACE>(param_ColorSpace.u.pd.value), BT601, SMPTE240M);
                paramsStrP->srParam_GrayThreshold     = static_cast<float>(CLAMP_VALUE(static_cast<int32_t>(param_GrayThreshold.u.sd.value), gMinGrayThreshold, gMaxGrayThreshold)) / 100.f;
                paramsStrP->srParam_ItrerationsNumber =  CLAMP_VALUE(static_cast<int32_t>(param_IterationsNumber.u.sd.value), iterMinCnt, iterMaxCnt);
            } // if (PF_Err_NONE == errParam1 && PF_Err_NONE == errParam2 ... )
            else
            {
                paramsStrP->srParam_Illuminant        = DAYLIGHT;
                paramsStrP->srParam_ChromaticAdapt    = CHROMATIC_CAT02;
                paramsStrP->srParam_ColorSpace        = BT709;
                paramsStrP->srParam_GrayThreshold     = static_cast<float>(gDefGrayThreshold) / 100.f;
                paramsStrP->srParam_ItrerationsNumber = iterDefCnt;
            }

            PF_RenderRequest req = extra->input->output_request;
            PF_CheckoutResult in_result{};

            ERR(extra->cb->checkout_layer
            (in_data->effect_ref, AWB_INPUT, AWB_INPUT, &req, in_data->current_time, in_data->time_step, in_data->time_scale, &in_result));

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
AuthomaticWhiteBalance_SmartRender
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
    const PAWB_SmartRenderParams pAWBStrParams = reinterpret_cast<const PAWB_SmartRenderParams>(handleSuite->host_lock_handle(reinterpret_cast<PF_Handle>(extraP->input->pre_render_data)));

    if (nullptr != pAWBStrParams)
    {
        ERR((extraP->cb->checkout_layer_pixels(in_data->effect_ref, AWB_INPUT, &input_worldP)));
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
                switch (format)
                {
                    case PF_PixelFormat_ARGB128:
                    {
                        const A_long srcPitch = srcRowBytes / static_cast<A_long>(PF_Pixel_ARGB_32f_size);
                        const A_long dstPitch = dstRowBytes / static_cast<A_long>(PF_Pixel_ARGB_32f_size);

                        PF_Pixel_ARGB_32f* __restrict input_pixels  = reinterpret_cast<PF_Pixel_ARGB_32f* __restrict>(input_worldP->data);
                        PF_Pixel_ARGB_32f* __restrict output_pixels = reinterpret_cast<PF_Pixel_ARGB_32f* __restrict>(output_worldP->data);

                    }
                    break;

                    case PF_PixelFormat_ARGB64:
                    {
                        const A_long srcPitch = srcRowBytes / static_cast<A_long>(PF_Pixel_ARGB_16u_size);
                        const A_long dstPitch = dstRowBytes / static_cast<A_long>(PF_Pixel_ARGB_16u_size);

                        PF_Pixel_ARGB_16u* __restrict input_pixels  = reinterpret_cast<PF_Pixel_ARGB_16u* __restrict>(input_worldP->data);
                        PF_Pixel_ARGB_16u* __restrict output_pixels = reinterpret_cast<PF_Pixel_ARGB_16u* __restrict>(output_worldP->data);

                    }
                    break;

                    case PF_PixelFormat_ARGB32:
                    {
                        const A_long srcPitch = srcRowBytes / static_cast<A_long>(PF_Pixel_ARGB_8u_size);
                        const A_long dstPitch = dstRowBytes / static_cast<A_long>(PF_Pixel_ARGB_8u_size);

                        PF_Pixel_ARGB_8u* __restrict input_pixels  = reinterpret_cast<PF_Pixel_ARGB_8u* __restrict>(input_worldP->data);
                        PF_Pixel_ARGB_8u* __restrict output_pixels = reinterpret_cast<PF_Pixel_ARGB_8u* __restrict>(output_worldP->data);

                    }
                    break;

                    default:
                        err = PF_Err_BAD_CALLBACK_PARAM;
                    break;
                } // switch (format)

            } // if (PF_Err_NONE == wsP->PF_GetPixelFormat(input_worldP, &format))

            ERR(extraP->cb->checkin_layer_pixels(in_data->effect_ref, AWB_INPUT));

        } // if (nullptr != input_worldP && nullptr != output_worldP)

        handleSuite->host_unlock_handle(reinterpret_cast<PF_Handle>(extraP->input->pre_render_data));

    } // if (nullptr != pFilterStrParams)
    else
        err = PF_Err_OUT_OF_MEMORY;

    return err;
}