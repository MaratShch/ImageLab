#include "ColorCorrectionCieLAB.hpp"
#include "ColorCorrectionCieLABEnums.hpp"
#include "PrSDKAESupport.h"

static PF_Err
About(
	PF_InData		*in_data,
	PF_OutData		*out_data,
	PF_ParamDef		*params[],
	PF_LayerDef		*output)
{
	PF_SPRINTF(out_data->return_msg,
		"%s, v%d.%d\r%s",
		strName,
		ColorCorrection_VersionMajor,
		ColorCorrection_VersionMinor,
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

	constexpr PF_OutFlags out_flags1 =
		PF_OutFlag_WIDE_TIME_INPUT		|
		PF_OutFlag_USE_OUTPUT_EXTENT	|
		PF_OutFlag_PIX_INDEPENDENT		|
		PF_OutFlag_CUSTOM_UI			|
		PF_OutFlag_SEND_UPDATE_PARAMS_UI|
		PF_OutFlag_DEEP_COLOR_AWARE;

	constexpr PF_OutFlags out_flags2 =
		PF_OutFlag2_PARAM_GROUP_START_COLLAPSED_FLAG|
		PF_OutFlag2_DOESNT_NEED_EMPTY_PIXELS		|
		PF_OutFlag2_AUTOMATIC_WIDE_TIME_INPUT;

	out_data->my_version =
		PF_VERSION(
			ColorCorrection_VersionMajor,
			ColorCorrection_VersionMinor,
			ColorCorrection_VersionSub,
			ColorCorrection_VersionStage,
			ColorCorrection_VersionBuild
		);

	out_data->out_flags = out_flags1;
	out_data->out_flags2 = out_flags2;

	/* For Premiere - declare supported pixel formats */
	if (PremierId == in_data->appl_id)
	{
		AEFX_SuiteScoper<PF_PixelFormatSuite1> pixelFormatSuite =
			AEFX_SuiteScoper<PF_PixelFormatSuite1>(in_data, kPFPixelFormatSuite, kPFPixelFormatSuiteVersion1, out_data);

		/*	Add the pixel formats we support in order of preference. */
		(*pixelFormatSuite->ClearSupportedPixelFormats)(in_data->effect_ref);
//		(*pixelFormatSuite->AddSupportedPixelFormat)(in_data->effect_ref, PrPixelFormat_VUYA_4444_8u_709);
//		(*pixelFormatSuite->AddSupportedPixelFormat)(in_data->effect_ref, PrPixelFormat_VUYA_4444_8u);
//		(*pixelFormatSuite->AddSupportedPixelFormat)(in_data->effect_ref, PrPixelFormat_VUYA_4444_32f_709);
//		(*pixelFormatSuite->AddSupportedPixelFormat)(in_data->effect_ref, PrPixelFormat_VUYA_4444_32f);
		(*pixelFormatSuite->AddSupportedPixelFormat)(in_data->effect_ref, PrPixelFormat_BGRA_4444_8u);
//		(*pixelFormatSuite->AddSupportedPixelFormat)(in_data->effect_ref, PrPixelFormat_BGRA_4444_16u);
//		(*pixelFormatSuite->AddSupportedPixelFormat)(in_data->effect_ref, PrPixelFormat_BGRA_4444_32f);
	}

	return err;
}


static PF_Err
ParamsSetup(
	PF_InData		*in_data,
	PF_OutData		*out_data,
	PF_ParamDef		*params[],
	PF_LayerDef		*output)
{
	PF_ParamDef	def{};
	PF_Err		err = PF_Err_NONE;
	constexpr PF_ParamFlags flags = PF_ParamFlag_SUPERVISE | PF_ParamFlag_CANNOT_TIME_VARY | PF_ParamFlag_CANNOT_INTERP;
	constexpr PF_ParamUIFlags ui_flags = PF_PUI_NONE;

	AEFX_INIT_PARAM_STRUCTURE (def, flags, ui_flags);
	PF_ADD_SLIDER(
		strLcoarse,
		coarse_min_level,
		coarse_max_level,
		coarse_min_level,
		coarse_max_level,
		coarse_def_level,
		eCIELAB_SLIDER_L_COARSE);

	AEFX_INIT_PARAM_STRUCTURE(def, flags, ui_flags);
	PF_ADD_SLIDER(
		strLfine,
		fine_min_level,
		fine_max_level,
		fine_min_level,
		fine_max_level,
		fine_def_level,
		eCIELAB_SLIDER_L_FINE);

	AEFX_INIT_PARAM_STRUCTURE(def, flags, ui_flags);
	PF_ADD_SLIDER(
		strAcoarse,
		coarse_min_level,
		coarse_max_level,
		coarse_min_level,
		coarse_max_level,
		coarse_def_level,
		eCIELAB_SLIDER_A_COARSE);

	AEFX_INIT_PARAM_STRUCTURE(def, flags, ui_flags);
	PF_ADD_SLIDER(
		strAfine,
		fine_min_level,
		fine_max_level,
		fine_min_level,
		fine_max_level,
		fine_def_level,
		eCIELAB_SLIDER_A_FINE);

	AEFX_INIT_PARAM_STRUCTURE(def, flags, ui_flags);
	PF_ADD_SLIDER(
		strBcoarse,
		coarse_min_level,
		coarse_max_level,
		coarse_min_level,
		coarse_max_level,
		coarse_def_level,
		eCIELAB_SLIDER_B_COARSE);

	AEFX_INIT_PARAM_STRUCTURE(def, flags, ui_flags);
	PF_ADD_SLIDER(
		strBfine,
		fine_min_level,
		fine_max_level,
		fine_min_level,
		fine_max_level,
		fine_def_level,
		eCIELAB_SLIDER_B_FINE);

	out_data->num_params = eCIELAB_TOTAL_PARAMS;
	return PF_Err_NONE;
}


static PF_Err
GlobalSetdown(
	PF_InData* in_data
)
{
	/* nothing to do  */
	return PF_Err_NONE;
}


static PF_Err
Render(
	PF_InData		*in_data,
	PF_OutData		*out_data,
	PF_ParamDef		*params[],
	PF_LayerDef		*output)
{
	return ((PremierId == in_data->appl_id) ? ProcessImgInPR(in_data, out_data, params, output) : ProcessImgInAE(in_data, out_data, params, output));
}


static PF_Err
SmartRender(
	PF_InData				*in_data,
	PF_OutData				*out_data,
	PF_SmartRenderExtra		*extraP
)
{
	PF_Err err = PF_Err_NONE;
	return err;
}


inline void
ResetParams(
	PF_InData						*in_data,
	PF_OutData						*out_data,
	PF_ParamDef						*params[]
) noexcept
{
	return;
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
	return err;
}



static PF_Err
UpdateParameterUI(
	PF_InData			*in_data,
	PF_OutData			*out_data,
	PF_ParamDef			*params[],
	PF_LayerDef			*outputP
)
{
	PF_Err err = PF_Err_NONE;
	return err;
}


static PF_Err
HandleEvent(
	PF_InData		*in_data,
	PF_OutData		*out_data,
	PF_ParamDef		*params[],
	PF_LayerDef		*output,
	PF_EventExtra	*extra
)
{
	PF_Err		err = PF_Err_NONE;

	switch (extra->e_type)
	{
		case PF_Event_NEW_CONTEXT:
		break;
		
		case PF_Event_ACTIVATE:
		break;

		case PF_Event_DO_CLICK:
		break;

		case PF_Event_DRAG:
		break;

		case PF_Event_DRAW:
		break;

		case PF_Event_ADJUST_CURSOR:
		break;
		
		default:
		break;
	}
	return err;
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
	PF_Err		err = PF_Err_NONE;

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
				ERR(GlobalSetdown(in_data));
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

			case PF_Cmd_UPDATE_PARAMS_UI:
				ERR(UpdateParameterUI(in_data, out_data, params, output));
			break;

//			case PF_Cmd_EVENT:
//				err = HandleEvent(in_data, out_data, params, output, reinterpret_cast<PF_EventExtra*>(extra));
	//		break;

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