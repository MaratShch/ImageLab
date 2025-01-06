#include "BilateralFilter.hpp"
#include "BilateralFilterEnum.hpp"
#include "GaussMesh.hpp"
#include "PrSDKAESupport.h"
#include "ImageLabMemInterface.hpp"

static GaussMesh* gGaussMeshInstance = nullptr;
GaussMesh* getMeshHandler(void) { return gGaussMeshInstance; }


static PF_Err
About(
	PF_InData		*in_data,
	PF_OutData		*out_data,
	PF_ParamDef		*params[],
	PF_LayerDef		*output)
{
	PF_SPRINTF(
		out_data->return_msg,
		"%s, v%d.%d\r%s",
		strName,
		BilateralFilter_VersionMajor,
		BilateralFilter_VersionMinor,
		strCopyright);

	return PF_Err_NONE;
}

static PF_Err
GlobalSetup(
	PF_InData		*in_data,
	PF_OutData		*out_data,
	PF_ParamDef		*params[],
	PF_LayerDef		*output)
{
    if (false == LoadMemoryInterfaceProvider(in_data))
        return PF_Err_INTERNAL_STRUCT_DAMAGED;

    if (nullptr == (gGaussMeshInstance = CreateGaussMeshHandler()))
        return PF_Err_INTERNAL_STRUCT_DAMAGED;

	constexpr PF_OutFlags out_flags1 =
		PF_OutFlag_PIX_INDEPENDENT       |
		PF_OutFlag_SEND_UPDATE_PARAMS_UI |
		PF_OutFlag_USE_OUTPUT_EXTENT     |
		PF_OutFlag_DEEP_COLOR_AWARE      |
		PF_OutFlag_WIDE_TIME_INPUT;

	constexpr PF_OutFlags out_flags2 =
		PF_OutFlag2_PARAM_GROUP_START_COLLAPSED_FLAG |
		PF_OutFlag2_DOESNT_NEED_EMPTY_PIXELS         |
		PF_OutFlag2_AUTOMATIC_WIDE_TIME_INPUT;

	out_data->my_version =
		PF_VERSION(
			BilateralFilter_VersionMajor,
			BilateralFilter_VersionMinor,
			BilateralFilter_VersionSub,
			BilateralFilter_VersionStage,
			BilateralFilter_VersionBuild
		);

	out_data->out_flags  = out_flags1;
	out_data->out_flags2 = out_flags2;

	/* For Premiere - declare supported pixel formats */
	if (PremierId == in_data->appl_id)
	{
		auto const& pixelFormatSuite{ AEFX_SuiteScoper<PF_PixelFormatSuite1>(in_data, kPFPixelFormatSuite, kPFPixelFormatSuiteVersion1, out_data) };

		/*	Add the pixel formats we support in order of preference. */
		(*pixelFormatSuite->ClearSupportedPixelFormats)(in_data->effect_ref);

        /* Bilateral Filter as standalone PlugIn will support BGRA/ARGB formats only */
		(*pixelFormatSuite->AddSupportedPixelFormat)(in_data->effect_ref, PrPixelFormat_BGRA_4444_8u);
        (*pixelFormatSuite->AddSupportedPixelFormat)(in_data->effect_ref, PrPixelFormat_BGRA_4444_16u);
        (*pixelFormatSuite->AddSupportedPixelFormat)(in_data->effect_ref, PrPixelFormat_BGRA_4444_32f);
        (*pixelFormatSuite->AddSupportedPixelFormat)(in_data->effect_ref, PrPixelFormat_ARGB_4444_8u);
        (*pixelFormatSuite->AddSupportedPixelFormat)(in_data->effect_ref, PrPixelFormat_ARGB_4444_16u);
        (*pixelFormatSuite->AddSupportedPixelFormat)(in_data->effect_ref, PrPixelFormat_ARGB_4444_32f);
	}

	return PF_Err_NONE;
}


static PF_Err
GlobalSetdown(
	PF_InData		*in_data,
	PF_OutData		*out_data,
	PF_ParamDef		*params[],
	PF_LayerDef		*output)
{
    ReleaseGaussMeshHandler(nullptr);
    gGaussMeshInstance = nullptr;

	return PF_Err_NONE;
}



static PF_Err
ParamsSetup(
	PF_InData		*in_data,
	PF_OutData		*out_data,
	PF_ParamDef		*params[],
	PF_LayerDef		*output)
{
    PF_ParamDef	def;
    constexpr PF_ParamFlags flags{ PF_ParamFlag_SUPERVISE | PF_ParamFlag_CANNOT_TIME_VARY | PF_ParamFlag_CANNOT_INTERP };
    constexpr PF_ParamUIFlags ui_flags{ PF_PUI_NONE };
    constexpr PF_ParamUIFlags ui_flags_control_disabled{ ui_flags | PF_PUI_DISABLED };

    AEFX_INIT_PARAM_STRUCTURE(def, flags, ui_flags);
    PF_ADD_SLIDER(
        FilterWindowSizeStr,
        bilateralMinRadius,
        bilateralMaxRadius,
        bilateralMinRadius,
        bilateralMaxRadius,
        bilateralDefRadius,
        eBILATERAL_FILTER_RADIUS);

    AEFX_INIT_PARAM_STRUCTURE(def, flags, ui_flags_control_disabled);
    PF_ADD_FLOAT_SLIDERX(
        FilterSigmaStr,
        fSigmaValMin,
        fSigmaValMax,
        fSigmaValMin,
        fSigmaValMax,
        fSigmaValDefault,
        PF_Precision_TENTHS,
        0,
        0,
        eBILATERAL_FILTER_SIGMA);

    out_data->num_params = eBILATERAL_TOTAL_CONTROLS;

	return PF_Err_NONE;
}


static PF_Err
Render(
	PF_InData		*in_data,
	PF_OutData		*out_data,
	PF_ParamDef		*params[],
	PF_LayerDef		*output)
{
	return ((PremierId == in_data->appl_id ? ProcessImgInPR(in_data, out_data, params, output) : ProcessImgInAE(in_data, out_data, params, output)));
}



static PF_Err
SmartRender(
	PF_InData				*in_data,
	PF_OutData				*out_data,
	PF_SmartRenderExtra		*extraP
)
{
	PF_Err	err = PF_Err_NONE;
	return err;
}


static PF_Err
UserChangedParam(
    PF_InData						*in_data,
    PF_OutData						*out_data,
    PF_ParamDef						*params[],
    PF_LayerDef						*outputP,
    const PF_UserChangedParamExtra	*which_hitP
)
{
    PF_Err err = PF_Err_NONE;

    switch (which_hitP->param_index)
    {
        case eBILATERAL_FILTER_RADIUS:
        {
            const auto& sliderValue = params[eBILATERAL_FILTER_RADIUS]->u.sd.value;
            if (sliderValue > 0)
            {
                if (true == IsDisabledUI(params[eBILATERAL_FILTER_SIGMA]->ui_flags))
                {
                    params[eBILATERAL_FILTER_SIGMA]->ui_flags &= ~PF_PUI_DISABLED;
                    err = AEFX_SuiteScoper<PF_ParamUtilsSuite3>(in_data, kPFParamUtilsSuite, kPFParamUtilsSuiteVersion3, out_data)->
                        PF_UpdateParamUI(in_data->effect_ref, eBILATERAL_FILTER_SIGMA, params[eBILATERAL_FILTER_SIGMA]);
                }
            }
            else
            {
                if (false == IsDisabledUI(params[eBILATERAL_FILTER_SIGMA]->ui_flags))
                {
                    params[eBILATERAL_FILTER_SIGMA]->ui_flags |= PF_PUI_DISABLED;
                    err = AEFX_SuiteScoper<PF_ParamUtilsSuite3>(in_data, kPFParamUtilsSuite, kPFParamUtilsSuiteVersion3, out_data)->
                        PF_UpdateParamUI(in_data->effect_ref, eBILATERAL_FILTER_SIGMA, params[eBILATERAL_FILTER_SIGMA]);
                }
            }
        }
        break;

        default: // nothing todo
        break;
    }

    return err;
}


static PF_Err
UpdateParameterUI(
    PF_InData			*in_data,
    PF_OutData			*out_data,
    PF_ParamDef			*params[],
    PF_LayerDef			*output
)
{
    return PF_Err_NONE;
}


PLUGIN_ENTRY_POINT_CALL PF_Err
EffectMain(
	PF_Cmd			cmd,
	PF_InData		*in_data,
	PF_OutData		*out_data,
	PF_ParamDef		*params[],
	PF_LayerDef		*output,
	void			*extra)
{
	PF_Err err{ PF_Err_NONE };

	try {
		switch (cmd)
		{
			case PF_Cmd_ABOUT:
				ERR(About(in_data, out_data, params, output));
			break;

			case PF_Cmd_GLOBAL_SETUP:
				ERR(GlobalSetup(in_data, out_data, params, output));
			break;

			case PF_Cmd_GLOBAL_SETDOWN:
				ERR(GlobalSetdown(in_data, out_data, params, output));
			break;

			case PF_Cmd_PARAMS_SETUP:
				ERR(ParamsSetup(in_data, out_data, params, output));
			break;

			case PF_Cmd_RENDER:
				ERR(Render(in_data, out_data, params, output));
			break;

            case PF_Cmd_USER_CHANGED_PARAM:
                ERR(UserChangedParam(in_data, out_data, params, output, reinterpret_cast<const PF_UserChangedParamExtra*>(extra)));
            break;
            
            // Handling this selector will ensure that the UI will be properly initialized,
            // even before the user starts changing parameters to trigger PF_Cmd_USER_CHANGED_PARAM
            case PF_Cmd_UPDATE_PARAMS_UI:
                ERR(UpdateParameterUI(in_data, out_data, params, output));
            break;

            default:
			break;
		}
	}
	catch (PF_Err& thrown_err)
	{
		err = thrown_err;
	}

	return err;
}