#include <atomic>
#include "RetroVision.hpp"
#include "RetroVisionEnum.hpp"
#include "RetroVisionGui.hpp"

static PF_Err
RetroVision_UpdateControls_UI
(
    PF_InData	*in_data,
    PF_OutData	*out_data,
    PF_ParamDef	*params[],
    bool bAlgoActive,
    const AEFX_SuiteScoper<PF_ParamUtilsSuite3>& paramUtilsSuite
)
{
    if (true == bAlgoActive)
    {
        // Algorithm is activates
        SetBitmapIdx(1);
        params[UnderlyingType(RetroVision::eRETRO_VISION_DISPLAY)           ]->ui_flags &= ~PF_PUI_DISABLED;
        params[UnderlyingType(RetroVision::eRETRO_VISION_CGA_PALETTE)       ]->ui_flags &= ~PF_PUI_DISABLED;
        params[UnderlyingType(RetroVision::eRETRO_VISION_CGA_INTTENCITY_BIT)]->ui_flags &= ~PF_PUI_DISABLED;

        paramUtilsSuite->PF_UpdateParamUI(in_data->effect_ref, UnderlyingType(RetroVision::eRETRO_VISION_DISPLAY), params[UnderlyingType(RetroVision::eRETRO_VISION_DISPLAY)]);
        paramUtilsSuite->PF_UpdateParamUI(in_data->effect_ref, UnderlyingType(RetroVision::eRETRO_VISION_CGA_PALETTE), params[UnderlyingType(RetroVision::eRETRO_VISION_CGA_PALETTE)]);
        paramUtilsSuite->PF_UpdateParamUI(in_data->effect_ref, UnderlyingType(RetroVision::eRETRO_VISION_CGA_INTTENCITY_BIT), params[UnderlyingType(RetroVision::eRETRO_VISION_CGA_INTTENCITY_BIT)]);
    }
    else
    {
        // Algorithm is deactivated
        SetBitmapIdx(0);
        params[UnderlyingType(RetroVision::eRETRO_VISION_DISPLAY)           ]->ui_flags |= PF_PUI_DISABLED;
        params[UnderlyingType(RetroVision::eRETRO_VISION_CGA_PALETTE)       ]->ui_flags |= PF_PUI_DISABLED;
        params[UnderlyingType(RetroVision::eRETRO_VISION_CGA_INTTENCITY_BIT)]->ui_flags |= PF_PUI_DISABLED;
        params[UnderlyingType(RetroVision::eRETRO_VISION_EGA_PALETTE)       ]->ui_flags |= PF_PUI_DISABLED;
        params[UnderlyingType(RetroVision::eRETRO_VISION_VGA_PALETTE)       ]->ui_flags |= PF_PUI_DISABLED;

        paramUtilsSuite->PF_UpdateParamUI(in_data->effect_ref, UnderlyingType(RetroVision::eRETRO_VISION_DISPLAY), params[UnderlyingType(RetroVision::eRETRO_VISION_DISPLAY)]);
        paramUtilsSuite->PF_UpdateParamUI(in_data->effect_ref, UnderlyingType(RetroVision::eRETRO_VISION_CGA_PALETTE), params[UnderlyingType(RetroVision::eRETRO_VISION_CGA_PALETTE)]);
        paramUtilsSuite->PF_UpdateParamUI(in_data->effect_ref, UnderlyingType(RetroVision::eRETRO_VISION_CGA_INTTENCITY_BIT), params[UnderlyingType(RetroVision::eRETRO_VISION_CGA_INTTENCITY_BIT)]);
        paramUtilsSuite->PF_UpdateParamUI(in_data->effect_ref, UnderlyingType(RetroVision::eRETRO_VISION_EGA_PALETTE), params[UnderlyingType(RetroVision::eRETRO_VISION_EGA_PALETTE)]);
        paramUtilsSuite->PF_UpdateParamUI(in_data->effect_ref, UnderlyingType(RetroVision::eRETRO_VISION_VGA_PALETTE), params[UnderlyingType(RetroVision::eRETRO_VISION_VGA_PALETTE)]);
    }

    return PF_Err_NONE;
}


static PF_Err
RetroVision_UpdateMonitor_UI
(
    PF_InData	*in_data,
    PF_OutData	*out_data,
    PF_ParamDef	*params[],
    const RetroMonitor& monitor,
    const AEFX_SuiteScoper<PF_ParamUtilsSuite3>& paramUtilsSuite
)
{
    switch (monitor)
    {
        case RetroMonitor::eRETRO_BITMAP_CGA:
        {
            SetBitmapIdx(1);
            params[UnderlyingType(RetroVision::eRETRO_VISION_CGA_PALETTE)]->ui_flags        &= ~PF_PUI_DISABLED;
            params[UnderlyingType(RetroVision::eRETRO_VISION_CGA_INTTENCITY_BIT)]->ui_flags &= ~PF_PUI_DISABLED;
            params[UnderlyingType(RetroVision::eRETRO_VISION_EGA_PALETTE)]->ui_flags        |= PF_PUI_DISABLED;
            params[UnderlyingType(RetroVision::eRETRO_VISION_VGA_PALETTE)]->ui_flags        |= PF_PUI_DISABLED;
        }
        break;

        case RetroMonitor::eRETRO_BITMAP_EGA:
        {
            params[UnderlyingType(RetroVision::eRETRO_VISION_CGA_PALETTE)]->ui_flags        |= PF_PUI_DISABLED;
            params[UnderlyingType(RetroVision::eRETRO_VISION_CGA_INTTENCITY_BIT)]->ui_flags |= PF_PUI_DISABLED;
            params[UnderlyingType(RetroVision::eRETRO_VISION_EGA_PALETTE)]->ui_flags        &= ~PF_PUI_DISABLED;
            params[UnderlyingType(RetroVision::eRETRO_VISION_VGA_PALETTE)]->ui_flags        |= PF_PUI_DISABLED;
        }
        break;

        case RetroMonitor::eRETRO_BITMAP_VGA:
        {
            params[UnderlyingType(RetroVision::eRETRO_VISION_CGA_PALETTE)]->ui_flags        |= PF_PUI_DISABLED;
            params[UnderlyingType(RetroVision::eRETRO_VISION_CGA_INTTENCITY_BIT)]->ui_flags |= PF_PUI_DISABLED;
            params[UnderlyingType(RetroVision::eRETRO_VISION_EGA_PALETTE)]->ui_flags        |= PF_PUI_DISABLED;
            params[UnderlyingType(RetroVision::eRETRO_VISION_VGA_PALETTE)]->ui_flags        &= ~PF_PUI_DISABLED;
        }
        break;

        case RetroMonitor::eRETRO_BITMAP_HERCULES:
        {
            params[UnderlyingType(RetroVision::eRETRO_VISION_CGA_PALETTE)]->ui_flags        |= PF_PUI_DISABLED;
            params[UnderlyingType(RetroVision::eRETRO_VISION_CGA_INTTENCITY_BIT)]->ui_flags |= PF_PUI_DISABLED;
            params[UnderlyingType(RetroVision::eRETRO_VISION_EGA_PALETTE)]->ui_flags        |= PF_PUI_DISABLED;
            params[UnderlyingType(RetroVision::eRETRO_VISION_VGA_PALETTE)]->ui_flags        |= PF_PUI_DISABLED;
        }
        break;

        default:
        break;

    }

    paramUtilsSuite->PF_UpdateParamUI(in_data->effect_ref, UnderlyingType(RetroVision::eRETRO_VISION_CGA_PALETTE), params[UnderlyingType(RetroVision::eRETRO_VISION_CGA_PALETTE)]);
    paramUtilsSuite->PF_UpdateParamUI(in_data->effect_ref, UnderlyingType(RetroVision::eRETRO_VISION_CGA_INTTENCITY_BIT), params[UnderlyingType(RetroVision::eRETRO_VISION_CGA_INTTENCITY_BIT)]);
    paramUtilsSuite->PF_UpdateParamUI(in_data->effect_ref, UnderlyingType(RetroVision::eRETRO_VISION_EGA_PALETTE), params[UnderlyingType(RetroVision::eRETRO_VISION_EGA_PALETTE)]);
    paramUtilsSuite->PF_UpdateParamUI(in_data->effect_ref, UnderlyingType(RetroVision::eRETRO_VISION_VGA_PALETTE), params[UnderlyingType(RetroVision::eRETRO_VISION_VGA_PALETTE)]);

    return PF_Err_NONE;
}



PF_Err
RetroVision_UserChangedParam
(
    PF_InData						*in_data,
    PF_OutData						*out_data,
    PF_ParamDef						*params[],
    PF_LayerDef						*outputP,
    const PF_UserChangedParamExtra	*which_hitP
)
{
    PF_Err err = PF_Err_NONE;
    AEFX_SuiteScoper<PF_ParamUtilsSuite3> paramUtilsSuite = AEFX_SuiteScoper<PF_ParamUtilsSuite3>(in_data, kPFParamUtilsSuite, kPFParamUtilsSuiteVersion3, out_data);

    switch (which_hitP->param_index)
    {
        case UnderlyingType(RetroVision::eRETRO_VISION_ENABLE):
        {
            const bool selected = (0 != params[UnderlyingType(RetroVision::eRETRO_VISION_ENABLE)]->u.bd.value);
            err = RetroVision_UpdateControls_UI (in_data, out_data, params, selected, paramUtilsSuite);
        }
        break;

        case UnderlyingType(RetroVision::eRETRO_VISION_DISPLAY):
        {
            const RetroMonitor monitor = static_cast<RetroMonitor>(params[UnderlyingType(RetroVision::eRETRO_VISION_DISPLAY)]->u.pd.value - 1);
            err = RetroVision_UpdateMonitor_UI (in_data, out_data, params, monitor, paramUtilsSuite);
        }
        break;

        case UnderlyingType(RetroVision::eRETRO_VISION_CGA_PALETTE):
        break;

        case UnderlyingType(RetroVision::eRETRO_VISION_CGA_INTTENCITY_BIT):
        break;

        case UnderlyingType(RetroVision::eRETRO_VISION_EGA_PALETTE):
        break;

        case UnderlyingType(RetroVision::eRETRO_VISION_VGA_PALETTE):
        break;

        default:
        // nothing TODO
        break;
    }

    return err;
}