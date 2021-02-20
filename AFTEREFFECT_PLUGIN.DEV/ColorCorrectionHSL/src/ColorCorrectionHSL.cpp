#include "ColorCorrectionHSL.hpp"
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
		PF_OutFlag_PIX_INDEPENDENT |
		PF_OutFlag_SEND_UPDATE_PARAMS_UI |
		PF_OutFlag_USE_OUTPUT_EXTENT |
		PF_OutFlag_DEEP_COLOR_AWARE |
		PF_OutFlag_WIDE_TIME_INPUT;

	constexpr PF_OutFlags out_flags2 =
		PF_OutFlag2_PARAM_GROUP_START_COLLAPSED_FLAG |
		PF_OutFlag2_DOESNT_NEED_EMPTY_PIXELS |
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
//		(*pixelFormatSuite->AddSupportedPixelFormat)(in_data->effect_ref, PrPixelFormat_RGB_444_10u);
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
	PF_ParamDef	def;
	PF_Err		err = PF_Err_NONE;
	constexpr PF_ParamFlags flags = PF_ParamFlag_SUPERVISE | PF_ParamFlag_CANNOT_TIME_VARY | PF_ParamFlag_CANNOT_INTERP;
	constexpr PF_ParamUIFlags ui_flags = PF_PUI_NONE;

	def.flags = flags;
	def.ui_flags = ui_flags;

	AEFX_CLR_STRUCT_EX(def);
	PF_ADD_POPUP(
		ColorSpaceType,					/* pop-up name			*/
		COLOR_SPACE_MAX_TYPES,			/* numbver of variants	*/
		COLOR_SPACE_HSL,				/* default variant		*/
		ColorSpace,						/* string for pop-up	*/
		COLOR_CORRECT_SPACE_POPUP);		/* control ID			*/

	AEFX_CLR_STRUCT_EX(def);
	PF_ADD_ANGLE(
		ColorHueCoarseType,
		hue_coarse_default,
		COLOR_CORRECT_HUE_COARSE_LEVEL
	);

	AEFX_CLR_STRUCT_EX(def);
	PF_ADD_FLOAT_SLIDERX(
		ColorHueFineLevel,
		hue_fine_min_level,
		hue_fine_max_level,
		hue_fine_min_level,
		hue_fine_max_level,
		hue_fine_def_level,
		PF_Precision_TENTHS,
		0,
		0,
		COLOR_HUE_FINE_LEVEL_SLIDER);

	AEFX_CLR_STRUCT_EX(def);
	PF_ADD_SLIDER(
		ColorSaturationCoarseLevel,
		sat_coarse_min_level,
		sat_coarse_max_level,
		sat_coarse_min_level,
		sat_coarse_max_level,
		sat_coarse_def_level,
		COLOR_SATURATION_COARSE_LEVEL_SLIDER);

	AEFX_CLR_STRUCT_EX(def);
	PF_ADD_FLOAT_SLIDERX(
		ColorSaturationFineLevel,
		sat_fine_min_level,
		sat_fine_max_level,
		sat_fine_min_level,
		sat_fine_max_level,
		sat_fine_def_level,
		PF_Precision_TENTHS,
		0,
		0,
		COLOR_SATURATION_FINE_LEVEL_SLIDER);

	AEFX_CLR_STRUCT_EX(def);
	PF_ADD_SLIDER(
		ColorLWIPCoarseLevel,
		lwip_coarse_min_level,
		lwip_coarse_max_level,
		lwip_coarse_min_level,
		lwip_coarse_max_level,
		lwip_coarse_def_level,
		COLOR_LWIP_COARSE_LEVEL_SLIDER);

	AEFX_CLR_STRUCT_EX(def);
	PF_ADD_FLOAT_SLIDERX(
		ColorLWIPFineLevel,
		lwip_fine_min_level,
		lwip_fine_max_level,
		lwip_fine_min_level,
		lwip_fine_max_level,
		lwip_fine_def_level,
		PF_Precision_TENTHS,
		0,
		0,
		COLOR_LWIP_FINE_LEVEL_SLIDER);

	AEFX_CLR_STRUCT_EX(def);
	PF_ADD_BUTTON(
		LoadSettingName,
		LoadSetting,
		0,
		PF_ParamFlag_SUPERVISE,
		COLOR_LOAD_SETTING_BUTTON
	);

	AEFX_CLR_STRUCT_EX(def);
	PF_ADD_BUTTON(
		SaveSettingName,
		SaveSetting,
		0,
		PF_ParamFlag_SUPERVISE,
		COLOR_SAVE_SETTING_BUTTON
	);

	AEFX_CLR_STRUCT_EX(def);
	PF_ADD_BUTTON(
		ResetSettingName,
		ResetSetting,
		0,
		PF_ParamFlag_SUPERVISE,
		COLOR_RESET_SETTING_BUTTON
	);

	out_data->num_params = COLOR_CORRECT_TOTAL_PARAMS;

	return PF_Err_NONE;
}

static PF_Err
GlobalSetdown(
	PF_InData* in_data
)
{
	return PF_Err_NONE;
}

bool IsProcActivated(PF_InData* in_data, PF_ParamDef* params[]) noexcept
{
	bool bRet = true;

	if (0   == params[COLOR_CORRECT_HUE_COARSE_LEVEL]->u.sd.value        &&
		0.f == params[COLOR_HUE_FINE_LEVEL_SLIDER]->u.fs_d.value         &&
		0   == params[COLOR_SATURATION_COARSE_LEVEL_SLIDER]->u.sd.value  &&
		0.f == params[COLOR_SATURATION_FINE_LEVEL_SLIDER]->u.fs_d.value  &&
		0   == params[COLOR_LWIP_COARSE_LEVEL_SLIDER]->u.sd.value        &&
		0.f == params[COLOR_LWIP_FINE_LEVEL_SLIDER]->u.fs_d.value)
	{
		bRet = false;
	}

	return bRet;
}

static PF_Err
Render(
	PF_InData		*in_data,
	PF_OutData		*out_data,
	PF_ParamDef		*params[],
	PF_LayerDef		*output)
{
	PF_Err	err = PF_Err_NONE;

	/* any sliders not moved - so processing not activated yet. Lets, simply, copy input data to output buffer */
	if (false == IsProcActivated(in_data, params))
	{
		/* no acion items, just copy input buffer to outpu without processing */
		if (PremierId != in_data->appl_id)
		{
			AEFX_SuiteScoper<PF_WorldTransformSuite1> worldTransformSuite =
				AEFX_SuiteScoper<PF_WorldTransformSuite1>(in_data, kPFPixelFormatSuite, kPFPixelFormatSuiteVersion1, out_data);

			err = (PF_Quality_HI == in_data->quality) ?
				worldTransformSuite->copy_hq(in_data->effect_ref, &params[COLOR_CORRECT_INPUT]->u.ld, output, NULL, NULL) :
				worldTransformSuite->copy(in_data->effect_ref, &params[COLOR_CORRECT_INPUT]->u.ld, output, NULL, NULL);
		}
		else {
			err = PF_COPY(&params[COLOR_CORRECT_INPUT]->u.ld, output, NULL, NULL);
		}
		return err;
	}

	err = ((PremierId == in_data->appl_id) ?
		ProcessImgInPR(in_data, out_data, params, output) :
		ProcessImgInAE(in_data, out_data, params, output));

	return err;
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


static PF_Err
UserChangedParam(
	PF_InData						*in_data,
	PF_OutData						*out_data,
	PF_ParamDef						*params[],
	PF_LayerDef						*outputP,
	const PF_UserChangedParamExtra	*which_hitP
) noexcept
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
) noexcept
{
	PF_Err err = PF_Err_NONE;
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