#include "ImageStylization.hpp"
#include "ImageLabMemInterface.hpp"
#include "StylizationStructs.hpp"
#include "PrSDKAESupport.h"



inline void setGlassySlider(PF_InData *in_data, PF_OutData *out_data, PF_ParamDef *params[], const bool& bEnable)
{
	if (false == bEnable)
	{
		strncpy_s(params[IMAGE_STYLE_SLIDER1]->name, StyleSlider1[0], PF_MAX_EFFECT_PARAM_NAME_LEN);
		params[IMAGE_STYLE_SLIDER1]->ui_flags |= PF_PUI_DISABLED;
	}
	else
	{
		strncpy_s(params[IMAGE_STYLE_SLIDER1]->name, StyleSlider1[1], PF_MAX_EFFECT_PARAM_NAME_LEN);
		params[IMAGE_STYLE_SLIDER1]->ui_flags &= ~PF_PUI_DISABLED;

	}

	AEFX_SuiteScoper<PF_ParamUtilsSuite3>(in_data, kPFParamUtilsSuite, kPFParamUtilsSuiteVersion3, out_data)->
		PF_UpdateParamUI(in_data->effect_ref, IMAGE_STYLE_SLIDER1, params[IMAGE_STYLE_SLIDER1]);

	return;
}


static PF_Err
About
(
	PF_InData		*in_data,
	PF_OutData		*out_data,
	PF_ParamDef		*params[],
	PF_LayerDef		*output
)
{
	PF_SPRINTF(out_data->return_msg,
		"%s, v%d.%d\r%s",
		strName,
		ImageStyle_VersionMajor,
		ImageStyle_VersionMinor,
		strCopyright);

	return PF_Err_NONE;
}


static PF_Err
GlobalSetup
(
	PF_InData		*in_data,
	PF_OutData		*out_data,
	PF_ParamDef		*params[],
	PF_LayerDef		*output
)
{
	PF_Err	err = PF_Err_NONE;

    LoadMemoryInterfaceProvider (in_data);

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
			ImageStyle_VersionMajor,
			ImageStyle_VersionMinor,
			ImageStyle_VersionSub,
			ImageStyle_VersionStage,
			ImageStyle_VersionBuild
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

		(*pixelFormatSuite->AddSupportedPixelFormat)(in_data->effect_ref, PrPixelFormat_VUYA_4444_32f_709);
		(*pixelFormatSuite->AddSupportedPixelFormat)(in_data->effect_ref, PrPixelFormat_VUYA_4444_32f);
		(*pixelFormatSuite->AddSupportedPixelFormat)(in_data->effect_ref, PrPixelFormat_BGRA_4444_32f);
		(*pixelFormatSuite->AddSupportedPixelFormat)(in_data->effect_ref, PrPixelFormat_BGRA_4444_16u);
		(*pixelFormatSuite->AddSupportedPixelFormat)(in_data->effect_ref, PrPixelFormat_BGRA_4444_8u);
		(*pixelFormatSuite->AddSupportedPixelFormat)(in_data->effect_ref, PrPixelFormat_VUYA_4444_8u_709);
		(*pixelFormatSuite->AddSupportedPixelFormat)(in_data->effect_ref, PrPixelFormat_VUYA_4444_8u);
	}

	/* generate random values in buffer used in Glassy Effect */
	utils_create_random_buffer();

	return err;
}


static PF_Err
GlobalSetDown
(
	PF_InData		*in_data,
	PF_OutData		*out_data)
{
    UnloadMemoryInterfaceProvider();
    return PF_Err_NONE;
}


static PF_Err
SequenceSetup
(
	PF_InData		*in_data,
	PF_OutData		*out_data
)
{
    PF_Err	err = PF_Err_NONE;
	return err;
}


static PF_Err
SequenceReSetup
(
	PF_InData		*in_data,
	PF_OutData		*out_data
)
{
    PF_Err err = PF_Err_NONE;
    return err;
}


static PF_Err
SequenceSetdown
(
	PF_InData		*in_data,
	PF_OutData		*out_data
)
{
    return PF_Err_NONE;
}


static PF_Err
ParamsSetup
(
	PF_InData		*in_data,
	PF_OutData		*out_data,
	PF_ParamDef		*params[],
	PF_LayerDef		*output)
{
	PF_ParamDef	def;
	constexpr PF_ParamFlags flags = PF_ParamFlag_SUPERVISE | PF_ParamFlag_CANNOT_TIME_VARY | PF_ParamFlag_CANNOT_INTERP;
	constexpr PF_ParamUIFlags ui_flags = PF_PUI_NONE;

    AEFX_INIT_PARAM_STRUCTURE(def, flags, ui_flags);
	PF_ADD_POPUP(
		strStylePopup,			/* pop-up name			*/
		eSTYLE_TOTAL_EFFECTS,	/* number of variants	*/
		eSTYLE_NONE,			/* default variant		*/
		strStyleEffect,			/* string for pop-up	*/
		IMAGE_STYLE_POPUP);		/* control ID			*/

    AEFX_INIT_PARAM_STRUCTURE(def, flags, ui_flags);
    PF_ADD_SLIDER(
		StyleSlider1[0],
		glassyMin,
		glassyMax,
		glassyMin,
		glassyMax,
		glassyDefault,
		IMAGE_STYLE_SLIDER1);

	out_data->num_params = IMAGE_STYLE_TOTAL_PARAMS;

	return PF_Err_NONE;

}


static PF_Err
Render
(
	PF_InData		*in_data,
	PF_OutData		*out_data,
	PF_ParamDef		*params[],
	PF_LayerDef		*output
)
{
	PF_Err const& err = (PremierId == in_data->appl_id ? 
		ProcessImgInPR(in_data, out_data, params, output) : ProcessImgInAE(in_data, out_data, params, output));
	return err;
}


static PF_Err
UserChangedParam
(
	PF_InData						*in_data,
	PF_OutData						*out_data,
	PF_ParamDef						*params[],
	PF_LayerDef						*outputP,
	const PF_UserChangedParamExtra	*which_hitP
)
{
	switch (which_hitP->param_index)
	{
		case IMAGE_STYLE_POPUP:
		{
			eSTYLIZATION const& styleType = static_cast<eSTYLIZATION>(params[IMAGE_STYLE_POPUP]->u.pd.value - 1);
			if (eSTYLE_GLASSY_EFFECT == styleType)
				setGlassySlider(in_data, out_data, params, true);
			else
				setGlassySlider(in_data, out_data, params, false);
		}
		break;

		default:
		break;
	}

	return PF_Err_NONE;
}


static PF_Err
UpdateParameterUI
(
	PF_InData			*in_data,
	PF_OutData			*out_data,
	PF_ParamDef			*params[],
	PF_LayerDef			*output
)
{
	PF_Err		err = PF_Err_NONE;
	return err;
}



PLUGIN_ENTRY_POINT_CALL  PF_Err
EffectMain
(
	PF_Cmd			cmd,
	PF_InData		*in_data,
	PF_OutData		*out_data,
	PF_ParamDef		*params[],
	PF_LayerDef		*output,
	void			*extra
)
{
	PF_Err		err = PF_Err_NONE;

	_MM_SET_FLUSH_ZERO_MODE(_MM_FLUSH_ZERO_ON);
	_MM_SET_DENORMALS_ZERO_MODE(_MM_DENORMALS_ZERO_ON);

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
				ERR(GlobalSetDown(in_data, out_data));
			break;

			case PF_Cmd_SEQUENCE_SETUP:
				ERR(SequenceSetup(in_data, out_data));
			break;

			case PF_Cmd_SEQUENCE_RESETUP:
				ERR(SequenceReSetup(in_data, out_data));
			break;

			case PF_Cmd_SEQUENCE_SETDOWN:
				ERR(SequenceSetdown(in_data, out_data));
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
		} /* switch (cmd) */

	} /* try */
	catch (PF_Err& thrown_err)
	{
		err = thrown_err;
	}

	return err;
}