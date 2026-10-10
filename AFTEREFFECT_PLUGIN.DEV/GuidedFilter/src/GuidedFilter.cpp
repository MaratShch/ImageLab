#include "GuidedFilter.hpp"
#include "GuidedFilterEnum.hpp"
#include "PrSDKAESupport.h"
#include "ImageLabMemInterface.hpp"


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
		GuidedFilter_VersionMajor,
		GuidedFilter_VersionMinor,
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
	PF_Err	err = PF_Err_NONE;

    if (false == LoadMemoryInterfaceProvider(in_data))
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
        PF_OutFlag2_AUTOMATIC_WIDE_TIME_INPUT        |
        PF_OutFlag2_FLOAT_COLOR_AWARE                |
        PF_OutFlag2_SUPPORTS_SMART_RENDER;

	out_data->my_version =
		PF_VERSION(
			GuidedFilter_VersionMajor,
			GuidedFilter_VersionMinor,
			GuidedFilter_VersionSub,
			GuidedFilter_VersionStage,
			GuidedFilter_VersionBuild
		);

	out_data->out_flags  = out_flags1;
	out_data->out_flags2 = out_flags2;

	/* For Premiere - declare supported pixel formats */
	if (PremierId == in_data->appl_id)
	{
		auto const pixelFormatSuite{ AEFX_SuiteScoper<PF_PixelFormatSuite1>(in_data, kPFPixelFormatSuite, kPFPixelFormatSuiteVersion1, out_data) };

		/*	Add the pixel formats we support in order of preference. */
		(*pixelFormatSuite->ClearSupportedPixelFormats)(in_data->effect_ref);

        (*pixelFormatSuite->AddSupportedPixelFormat)(in_data->effect_ref, PrPixelFormat_BGRA_4444_8u);
/*
        (*pixelFormatSuite->AddSupportedPixelFormat)(in_data->effect_ref, PrPixelFormat_BGRA_4444_16u);
        (*pixelFormatSuite->AddSupportedPixelFormat)(in_data->effect_ref, PrPixelFormat_BGRA_4444_32f);
        (*pixelFormatSuite->AddSupportedPixelFormat)(in_data->effect_ref, PrPixelFormat_BGRA_4444_32f_Linear);
        (*pixelFormatSuite->AddSupportedPixelFormat)(in_data->effect_ref, PrPixelFormat_BGRP_4444_8u);
        (*pixelFormatSuite->AddSupportedPixelFormat)(in_data->effect_ref, PrPixelFormat_BGRP_4444_16u);
        (*pixelFormatSuite->AddSupportedPixelFormat)(in_data->effect_ref, PrPixelFormat_BGRP_4444_32f);
        (*pixelFormatSuite->AddSupportedPixelFormat)(in_data->effect_ref, PrPixelFormat_BGRP_4444_32f_Linear);
        (*pixelFormatSuite->AddSupportedPixelFormat)(in_data->effect_ref, PrPixelFormat_BGRX_4444_8u);
        (*pixelFormatSuite->AddSupportedPixelFormat)(in_data->effect_ref, PrPixelFormat_BGRX_4444_16u);
        (*pixelFormatSuite->AddSupportedPixelFormat)(in_data->effect_ref, PrPixelFormat_BGRX_4444_32f);
        (*pixelFormatSuite->AddSupportedPixelFormat)(in_data->effect_ref, PrPixelFormat_BGRX_4444_32f_Linear);
        (*pixelFormatSuite->AddSupportedPixelFormat)(in_data->effect_ref, PrPixelFormat_VUYA_4444_8u_709);
        (*pixelFormatSuite->AddSupportedPixelFormat)(in_data->effect_ref, PrPixelFormat_VUYA_4444_8u);
        (*pixelFormatSuite->AddSupportedPixelFormat)(in_data->effect_ref, PrPixelFormat_VUYA_4444_32f_709);
        (*pixelFormatSuite->AddSupportedPixelFormat)(in_data->effect_ref, PrPixelFormat_VUYA_4444_32f);
        (*pixelFormatSuite->AddSupportedPixelFormat)(in_data->effect_ref, PrPixelFormat_VUYP_4444_8u_709);
        (*pixelFormatSuite->AddSupportedPixelFormat)(in_data->effect_ref, PrPixelFormat_VUYP_4444_8u);
        (*pixelFormatSuite->AddSupportedPixelFormat)(in_data->effect_ref, PrPixelFormat_VUYP_4444_32f_709);
        (*pixelFormatSuite->AddSupportedPixelFormat)(in_data->effect_ref, PrPixelFormat_VUYP_4444_32f);
        (*pixelFormatSuite->AddSupportedPixelFormat)(in_data->effect_ref, PrPixelFormat_VUYX_4444_8u_709);
        (*pixelFormatSuite->AddSupportedPixelFormat)(in_data->effect_ref, PrPixelFormat_VUYX_4444_8u);
        (*pixelFormatSuite->AddSupportedPixelFormat)(in_data->effect_ref, PrPixelFormat_VUYX_4444_32f_709);
        (*pixelFormatSuite->AddSupportedPixelFormat)(in_data->effect_ref, PrPixelFormat_VUYX_4444_32f);
        (*pixelFormatSuite->AddSupportedPixelFormat)(in_data->effect_ref, PrPixelFormat_ARGB_4444_8u);
        (*pixelFormatSuite->AddSupportedPixelFormat)(in_data->effect_ref, PrPixelFormat_ARGB_4444_16u);
        (*pixelFormatSuite->AddSupportedPixelFormat)(in_data->effect_ref, PrPixelFormat_ARGB_4444_32f);
        (*pixelFormatSuite->AddSupportedPixelFormat)(in_data->effect_ref, PrPixelFormat_PRGB_4444_32f);
        (*pixelFormatSuite->AddSupportedPixelFormat)(in_data->effect_ref, PrPixelFormat_XRGB_4444_32f);
        (*pixelFormatSuite->AddSupportedPixelFormat)(in_data->effect_ref, PrPixelFormat_ARGB_4444_32f_Linear);
        (*pixelFormatSuite->AddSupportedPixelFormat)(in_data->effect_ref, PrPixelFormat_PRGB_4444_32f_Linear);
        (*pixelFormatSuite->AddSupportedPixelFormat)(in_data->effect_ref, PrPixelFormat_XRGB_4444_32f_Linear);
        (*pixelFormatSuite->AddSupportedPixelFormat)(in_data->effect_ref, PrPixelFormat_RGB_444_10u);
*/
    }

	return err;
}


static PF_Err
GlobalSetdown(
	PF_InData		*in_data,
	PF_OutData		*out_data,
	PF_ParamDef		*params[],
	PF_LayerDef		*output)
{
    UnloadMemoryInterfaceProvider();
    return PF_Err_NONE;
}



static PF_Err
ParamsSetup(
	PF_InData		*in_data,
	PF_OutData		*out_data,
	PF_ParamDef		*params[],
	PF_LayerDef		*output)
{
    CACHE_ALIGN PF_ParamDef	def;

    constexpr PF_ParamFlags   flags = PF_ParamFlag_SUPERVISE | PF_ParamFlag_CANNOT_TIME_VARY | PF_ParamFlag_CANNOT_INTERP;
    constexpr PF_ParamUIFlags ui_flags = PF_PUI_CONTROL;

    // add 'Filter Mode' popup
    AEFX_INIT_PARAM_STRUCTURE(def, flags, ui_flags);
    PF_ADD_POPUP(
        ctrlItemNames[0],
        UnderlyingType(FilterMode::TotalModes),
        UnderlyingType(FilterMode::Entire_Image),
        gFilterModeStr,
        UnderlyingType(CtrlItems::gFILTER_MODE));

    // add 'Radius' slider
    AEFX_INIT_PARAM_STRUCTURE(def, flags, ui_flags);
    PF_ADD_SLIDER(
        ctrlItemNames[1],
        AlgoRadiusMin,
        AlgoRadiusMax,
        AlgoRadiusMin,
        AlgoRadiusMax,
        AlgoRadiusDef,
        UnderlyingType(CtrlItems::gFILTER_RADIUS));

    // add 'Epsilon' float slider
    AEFX_INIT_PARAM_STRUCTURE(def, flags, ui_flags);
    PF_ADD_FLOAT_SLIDERX(
        ctrlItemNames[2],
        AlgoEpsilonMin,
        AlgoEpsilonMax,
        AlgoEpsilonMin,
        AlgoEpsilonMax,
        AlgoEpsilonDef,
        PF_Precision_HUNDREDTHS,
        0,
        0,
        UnderlyingType(CtrlItems::gFILTER_EPSILON));

    // add 'Show Map' check-box
    AEFX_INIT_PARAM_STRUCTURE(def, flags, ui_flags);
    PF_ADD_CHECKBOXX(
        ctrlItemNames[3],
        FALSE,
        0,
        UnderlyingType(CtrlItems::gFILTER_SHOW_MAP));

    // add 'Skin Tolerance' float slider
    AEFX_INIT_PARAM_STRUCTURE(def, flags, ui_flags);
    PF_ADD_FLOAT_SLIDERX(
        ctrlItemNames[4],
        AlgoSkinToleranceMin,
        AlgoSkinToleranceMax,
        AlgoSkinToleranceMin,
        AlgoSkinToleranceMax,
        AlgoSkinToleranceDef,
        PF_Precision_HUNDREDTHS,
        0,
        0,
        UnderlyingType(CtrlItems::gFILTER_SKIN_TOLERANCE));

    // add 'Skin Softness' float slider
    AEFX_INIT_PARAM_STRUCTURE(def, flags, ui_flags);
    PF_ADD_FLOAT_SLIDERX(
        ctrlItemNames[5],
        AlgoSkinSoftnessMin,
        AlgoSkinSoftnessMax,
        AlgoSkinSoftnessMin,
        AlgoSkinSoftnessMax,
        AlgoSkinSoftnessDef,
        PF_Precision_HUNDREDTHS,
        0,
        0,
        UnderlyingType(CtrlItems::gFILTER_SKIN_SOFTNESS));

    // Set total number of control items
    out_data->num_params = UnderlyingType(CtrlItems::gFILTER_TOTAL_PARAMETERS);

	return PF_Err_NONE;
}


inline PF_Err Render(
	PF_InData		*in_data,
	PF_OutData		*out_data,
	PF_ParamDef		*params[],
	PF_LayerDef		*output)
{
	return ((PremierId == in_data->appl_id ? ProcessImgInPR(in_data, out_data, params, output) : ProcessImgInAE(in_data, out_data, params, output)));
}


static PF_Err
UpdateParameterUI(
    PF_InData			*in_data,
    PF_OutData			*out_data,
    PF_ParamDef			*params[],
    PF_LayerDef			*output
)
{
    PF_Err	err = PF_Err_NONE;
    return err;
}


inline PF_Err SmartRender(
	PF_InData				*in_data,
	PF_OutData				*out_data,
	PF_SmartRenderExtra		*extraP
)
{
    return GuidedFilter_SmartRender(in_data, out_data, extraP);
}


inline PF_Err
PreRender(
    PF_InData				*in_data,
    PF_OutData				*out_data,
    PF_PreRenderExtra		*extraP
)
{
    return GuidedFilter_PreRender(in_data, out_data, extraP);
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
	PF_Err		err{ PF_Err_NONE };

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

            // Handling this selector will ensure that the UI will be properly initialized,
            // even before the user starts changing parameters to trigger PF_Cmd_USER_CHANGED_PARAM
            case PF_Cmd_UPDATE_PARAMS_UI:
                ERR(UpdateParameterUI(in_data, out_data, params, output));
            break;

            case PF_Cmd_SMART_PRE_RENDER:
                ERR(PreRender(in_data, out_data, reinterpret_cast<PF_PreRenderExtra*>(extra)));
            break;

            case PF_Cmd_SMART_RENDER:
                ERR(SmartRender(in_data, out_data, reinterpret_cast<PF_SmartRenderExtra*>(extra)));
            break;

            default:
			break;
		}
	}
	catch (PF_Err &thrown_err)
	{
		err = thrown_err;
	}

	return err;
}