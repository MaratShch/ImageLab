#include <vector>
#include "FilmSimulation.hpp"
#include "CommonSmartRender.hpp"
#include "AlgoAdobeControlEnums.hpp"

namespace
{
    // Checkout id of the input layer (the plugin's own number, not a param index).
    constexpr A_long kInputCheckoutId = 1;

    constexpr A_long kInputIndex = static_cast<A_long>(UnderlyingType(FilmSimulationCtrl::VIDEO_INPUT));
    constexpr A_long kTotalParams = static_cast<A_long>(UnderlyingType(FilmSimulationCtrl::TOTAL_PARAMS));
}


// ---------------------------------------------------------------------------
// The film model needs the WHOLE frame (blurs and halation reach across it, the
// veiling flare uses the frame mean, vignette and damage are placed by
// position), so the input is always requested in full and the output is
// declared to be the same rectangle (RETURNS_EXTRA_PIXELS: AE crops it).
// ---------------------------------------------------------------------------
PF_Err
FilmSimulation_PreRender
(
    PF_InData			*in_data,
    PF_OutData			*out_data,
    PF_PreRenderExtra	*extra
)
{
    PF_Err err = PF_Err_NONE;

    PF_RenderRequest  req = extra->input->output_request;
    PF_CheckoutResult inResult;
    AEFX_CLR_STRUCT(inResult);

    // The full layer, united with what AE asked for (an upstream effect may
    // have grown the layer's rectangle). in_data->width / height are the
    // layer's full-resolution size, never smaller than its downsampled size,
    // and AE clips a request to the pixels that exist (max_result_rect), so
    // the request always covers the whole frame at any resolution.
    PF_LRect fullRect;
    fullRect.left   = 0;
    fullRect.top    = 0;
    fullRect.right  = in_data->width;
    fullRect.bottom = in_data->height;
    UnionLRect(&fullRect, &req.rect);

    ERR(extra->cb->checkout_layer(in_data->effect_ref, kInputIndex, kInputCheckoutId, &req,
                                  in_data->current_time, in_data->time_step, in_data->time_scale, &inResult));

    if (PF_Err_NONE == err)
    {
        extra->output->result_rect     = inResult.result_rect;
        extra->output->max_result_rect = inResult.max_result_rect;
        extra->output->solid           = FALSE;
        extra->output->flags          |= PF_RenderOutputFlag_RETURNS_EXTRA_PIXELS;
    }

    return err;
}


PF_Err
FilmSimulation_SmartRender
(
    PF_InData				*in_data,
    PF_OutData				*out_data,
    PF_SmartRenderExtra		*extraP
)
{
    PF_Err err  = PF_Err_NONE;
    PF_Err err2 = PF_Err_NONE;

    PF_EffectWorld* input  = nullptr;
    PF_EffectWorld* output = nullptr;

    ERR(extraP->cb->checkout_layer_pixels(in_data->effect_ref, kInputCheckoutId, &input));
    ERR(extraP->cb->checkout_output(in_data->effect_ref, &output));

    if (PF_Err_NONE == err && nullptr != input && nullptr != output)
    {
        // Smart Render receives no params[]: check every parameter out at the
        // current time and hand getAlgoControls() the same array layout as
        // PF_Cmd_RENDER (index 0, the layer, is not read by getAlgoControls).
        std::vector<PF_ParamDef>  paramStore(static_cast<size_t>(kTotalParams));
        std::vector<PF_ParamDef*> params(static_cast<size_t>(kTotalParams), nullptr);
        std::vector<bool>         checkedOut(static_cast<size_t>(kTotalParams), false);

        for (A_long i = 0; i < kTotalParams; i++)
        {
            AEFX_CLR_STRUCT(paramStore[i]);
            params[i] = &paramStore[i];
        }

        for (A_long i = kInputIndex + 1; i < kTotalParams && PF_Err_NONE == err; i++)
        {
            err = PF_CHECKOUT_PARAM(in_data, i, in_data->current_time, in_data->time_step, in_data->time_scale, &paramStore[i]);
            if (PF_Err_NONE == err)
                checkedOut[i] = true;
        }

        if (PF_Err_NONE == err)
        {
            try
            {
                err = ProcessImgAEWorld(in_data, out_data, params.data(), input, output);
            }
            catch (...)
            {
                // check the params back in below, then report
                err = PF_Err_INTERNAL_STRUCT_DAMAGED;
            }
        }

        for (A_long i = kInputIndex + 1; i < kTotalParams; i++)
        {
            if (true == checkedOut[i])
                ERR2(PF_CHECKIN_PARAM(in_data, &paramStore[i]));
        }
    }

    ERR2(extraP->cb->checkin_layer_pixels(in_data->effect_ref, kInputCheckoutId));

    return err;
}
