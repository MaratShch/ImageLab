#include "ColorTemperature.hpp"
#include "ColorTemperatureEnums.hpp"
#include "PrSDKAESupport.h"



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
		ColorTemperature_VersionMajor,
		ColorTemperature_VersionMinor,
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
			ColorTemperature_VersionMajor,
			ColorTemperature_VersionMinor,
			ColorTemperature_VersionSub,
			ColorTemperature_VersionStage,
			ColorTemperature_VersionBuild
		);

	out_data->out_flags  = out_flags1;
	out_data->out_flags2 = out_flags2;

	/* For Premiere - declare supported pixel formats */
	if (PremierId == in_data->appl_id)
	{
		auto const& pixelFormatSuite{ AEFX_SuiteScoper<PF_PixelFormatSuite1>(in_data, kPFPixelFormatSuite, kPFPixelFormatSuiteVersion1, out_data) };

		/*	Add the pixel formats we support in order of preference. */
		(*pixelFormatSuite->ClearSupportedPixelFormats)(in_data->effect_ref);

		(*pixelFormatSuite->AddSupportedPixelFormat)(in_data->effect_ref, PrPixelFormat_BGRA_4444_8u);
		(*pixelFormatSuite->AddSupportedPixelFormat)(in_data->effect_ref, PrPixelFormat_BGRA_4444_16u);
		(*pixelFormatSuite->AddSupportedPixelFormat)(in_data->effect_ref, PrPixelFormat_BGRA_4444_32f);
		(*pixelFormatSuite->AddSupportedPixelFormat)(in_data->effect_ref, PrPixelFormat_VUYA_4444_8u_709);
		(*pixelFormatSuite->AddSupportedPixelFormat)(in_data->effect_ref, PrPixelFormat_VUYA_4444_8u);
		(*pixelFormatSuite->AddSupportedPixelFormat)(in_data->effect_ref, PrPixelFormat_VUYA_4444_32f_709);
		(*pixelFormatSuite->AddSupportedPixelFormat)(in_data->effect_ref, PrPixelFormat_VUYA_4444_32f);
		(*pixelFormatSuite->AddSupportedPixelFormat)(in_data->effect_ref, PrPixelFormat_RGB_444_10u);
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
	/* nothing to do */
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
	constexpr PF_ParamFlags   flags = PF_ParamFlag_SUPERVISE | PF_ParamFlag_CANNOT_TIME_VARY | PF_ParamFlag_CANNOT_INTERP;
	constexpr PF_ParamUIFlags ui_flags = PF_PUI_NONE;
	
	AEFX_INIT_PARAM_STRUCTURE(def, flags, ui_flags);
	PF_ADD_POPUP(
		controlName[0],						/* pop-up name			*/
		eTEMP_STANDARD_TOTAL,				/* number of variants	*/
		eTEMP_STANDARD_BLACK_BODY,			/* default variant		*/
		strStandardName,					/* string for pop-up	*/
		COLOR_TEMPERATURE_STANDARD_POPUP);	/* control ID			*/

	AEFX_INIT_PARAM_STRUCTURE(def, flags, ui_flags);
	PF_ADD_POPUP(
		controlName[1],						/* pop-up name			*/
		eTEMP_GAMMA_VALUE_TOTAL,			/* number of variants	*/
		eTEMP_GAMMA_VALUE_10,		    	/* default variant		*/
		strGammaValueName,					/* string for pop-up	*/
		COLOR_TEMPERATURE_GAMMA_POPUP);		/* control ID			*/

	AEFX_INIT_PARAM_STRUCTURE(def, flags, ui_flags);
	PF_ADD_FLOAT_SLIDERX(
		controlName[2],
		colorTemperature2Slider(algoColorTempMin),
		colorTemperature2Slider(algoColorTempMax),
		colorTemperature2Slider(algoColorTempMin),
		colorTemperature2Slider(algoColorTempMax),
		colorTemperature2Slider(algoColorWhitePoint),
		PF_Precision_TENTHS,
		0,
		0,
		COLOR_TEMPERATURE_VALUE_SLIDER);


	AEFX_INIT_PARAM_STRUCTURE(def, flags, ui_flags);
	PF_ADD_SLIDER(
		controlName[3],
		algoColorTintMin,
		algoColorTintMax,
		algoColorTintMin,
		algoColorTintMax,
		algoColorTintDefault,
		COLOR_TEMPERATURE_TINT_SLIDER);

	out_data->num_params = COLOR_TEMPERATURE_TOTAL_CONTROLS;

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
HandleEvent(
	PF_InData		*in_data,
	PF_OutData		*out_data,
	PF_ParamDef		*params[],
	PF_LayerDef		*output,
	PF_EventExtra	*extra)
{
	PF_Err		err = PF_Err_NONE;
	
	switch (extra->e_type)
	{
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

			case PF_Cmd_EVENT:
				ERR(HandleEvent(in_data, out_data, params, output, reinterpret_cast<PF_EventExtra*>(extra)));
			break;

			default:
			break;
		}
	}
	catch (PF_Err & thrown_err)
	{
		err = thrown_err;
	}

	return err;
}