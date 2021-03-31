#include "ColorCorrectionCMYK.hpp"
#include "ColorCorrectionEnums.hpp"
#include "PrSDKAESupport.h"
#include <memory>

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
		std::unique_ptr<AEFX_SuiteScoper<PF_PixelFormatSuite1>> pixelFormatSuite =
			std::make_unique<AEFX_SuiteScoper<PF_PixelFormatSuite1>>(in_data, kPFPixelFormatSuite, kPFPixelFormatSuiteVersion1, out_data);

		/*	Add the pixel formats we support in order of preference. */
		((*pixelFormatSuite)->ClearSupportedPixelFormats)(in_data->effect_ref);

//		((*pixelFormatSuite)->AddSupportedPixelFormat)(in_data->effect_ref, PrPixelFormat_VUYA_4444_8u_709);
//		((*pixelFormatSuite)->AddSupportedPixelFormat)(in_data->effect_ref, PrPixelFormat_VUYA_4444_8u);
//		((*pixelFormatSuite)->AddSupportedPixelFormat)(in_data->effect_ref, PrPixelFormat_VUYA_4444_32f_709);
//		((*pixelFormatSuite)->AddSupportedPixelFormat)(in_data->effect_ref, PrPixelFormat_VUYA_4444_32f);
///		((*pixelFormatSuite)->AddSupportedPixelFormat)(in_data->effect_ref, PrPixelFormat_BGRA_4444_8u);
///		((*pixelFormatSuite)->AddSupportedPixelFormat)(in_data->effect_ref, PrPixelFormat_BGRA_4444_16u);
		((*pixelFormatSuite)->AddSupportedPixelFormat)(in_data->effect_ref, PrPixelFormat_BGRA_4444_32f);
//		((*pixelFormatSuite)->AddSupportedPixelFormat)(in_data->effect_ref, PrPixelFormat_RGB_444_10u);
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
	constexpr PF_ParamUIFlags ui_flags = PF_PUI_TOPIC | PF_PUI_CONTROL;
	constexpr int32_t defaultSliders = static_cast<int32_t>(eCMYK);

	AEFX_CLR_STRUCT_EX(def);
	def.flags = flags;
	def.ui_flags = PF_PUI_NONE;
	PF_ADD_POPUP(
		ColorSpaceType,					/* pop-up name			*/
		eTOTAL_COLOR_DOMAINS,			/* number of variants	*/
		eCMYK,							/* default variant		*/
		ColorSpace,						/* string for pop-up	*/
		COLOR_CORRECT_SPACE_POPUP);		/* control ID			*/

	AEFX_CLR_STRUCT_EX(def);
	def.flags = flags;
	def.ui_flags = ui_flags;
	PF_ADD_SLIDER(
		ColorSlider1[defaultSliders],
		coarse_min_level,
		coarse_max_level,
		coarse_min_level,
		coarse_max_level,
		coarse_def_level,
		COLOR_CORRECT_SLIDER1);

	AEFX_CLR_STRUCT_EX(def);
	def.flags = flags;
	def.ui_flags = ui_flags;
	PF_ADD_FLOAT_SLIDERX(
		ColorSlider2[defaultSliders],
		fine_min_level,
		fine_max_level,
		fine_min_level,
		fine_max_level,
		fine_def_level,
		PF_Precision_TENTHS,
		0,
		0,
		COLOR_CORRECT_SLIDER2);

	AEFX_CLR_STRUCT_EX(def);
	def.flags = flags;
	def.ui_flags = ui_flags;
	PF_ADD_SLIDER(
		ColorSlider3[defaultSliders],
		coarse_min_level,
		coarse_max_level,
		coarse_min_level,
		coarse_max_level,
		coarse_def_level,
		COLOR_CORRECT_SLIDER3);

	AEFX_CLR_STRUCT_EX(def);
	def.flags = flags;
	def.ui_flags = ui_flags;
	PF_ADD_FLOAT_SLIDERX(
		ColorSlider4[defaultSliders],
		fine_min_level,
		fine_max_level,
		fine_min_level,
		fine_max_level,
		fine_def_level,
		PF_Precision_TENTHS,
		0,
		0,
		COLOR_CORRECT_SLIDER4);

	AEFX_CLR_STRUCT_EX(def);
	def.flags = flags;
	def.ui_flags = ui_flags;
	PF_ADD_SLIDER(
		ColorSlider5[defaultSliders],
		coarse_min_level,
		coarse_max_level,
		coarse_min_level,
		coarse_max_level,
		coarse_def_level,
		COLOR_CORRECT_SLIDER5);

	AEFX_CLR_STRUCT_EX(def);
	def.flags = flags;
	def.ui_flags = ui_flags;
	PF_ADD_FLOAT_SLIDERX(
		ColorSlider6[defaultSliders],
		fine_min_level,
		fine_max_level,
		fine_min_level,
		fine_max_level,
		fine_def_level,
		PF_Precision_TENTHS,
		0,
		0,
		COLOR_CORRECT_SLIDER6);

	AEFX_CLR_STRUCT_EX(def);
	def.flags = flags;
	def.ui_flags = ui_flags;
	PF_ADD_SLIDER(
		ColorSlider7[defaultSliders],
		coarse_min_level,
		coarse_max_level,
		coarse_min_level,
		coarse_max_level,
		coarse_def_level,
		COLOR_CORRECT_SLIDER7);

	AEFX_CLR_STRUCT_EX(def);
	def.flags = flags;
	def.ui_flags = ui_flags;
	PF_ADD_FLOAT_SLIDERX(
		ColorSlider8[defaultSliders],
		fine_min_level,
		fine_max_level,
		fine_min_level,
		fine_max_level,
		fine_def_level,
		PF_Precision_TENTHS,
		0,
		0,
		COLOR_CORRECT_SLIDER8);

	AEFX_CLR_STRUCT_EX(def);

	return err;
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

	if (0   == params[COLOR_CORRECT_SLIDER1]->u.sd.value	&&
		0.0 == params[COLOR_CORRECT_SLIDER2]->u.fs_d.value	&&
		0   == params[COLOR_CORRECT_SLIDER3]->u.sd.value	&&
		0.0 == params[COLOR_CORRECT_SLIDER4]->u.fs_d.value	&&
		0   == params[COLOR_CORRECT_SLIDER5]->u.sd.value	&&
		0.0 == params[COLOR_CORRECT_SLIDER6]->u.fs_d.value	&&
		0   == params[COLOR_CORRECT_SLIDER7]->u.sd.value	&&
		0.0 == params[COLOR_CORRECT_SLIDER8]->u.fs_d.value)
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
				AEFX_SuiteScoper<PF_WorldTransformSuite1>(in_data, kPFWorldTransformSuite, kPFWorldTransformSuiteVersion1, out_data);

			err = (PF_Quality_HI == in_data->quality) ?
				worldTransformSuite->copy_hq(in_data->effect_ref, &params[COLOR_CORRECT_INPUT]->u.ld, output, NULL, NULL) :
				worldTransformSuite->copy(in_data->effect_ref, &params[COLOR_CORRECT_INPUT]->u.ld, output, NULL, NULL);
		}
		else {
			err = PF_COPY(&params[COLOR_CORRECT_INPUT]->u.ld, output, NULL, NULL);
		}
	}
	else
	{
		err = ((PremierId == in_data->appl_id) ?
			ProcessImgInPR(in_data, out_data, params, output) :
			ProcessImgInAE(in_data, out_data, params, output));
	}

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


inline void
ResetParams(
	PF_InData						*in_data,
	PF_OutData						*out_data,
	PF_ParamDef						*params[]
) noexcept
{
	return;
}


inline void Set_ColorSpace_CMYK (
	PF_InData	*in_data,
	PF_OutData	*out_data,
	PF_ParamDef	*params[]
) noexcept
{
	AEFX_SuiteScoper<PF_ParamUtilsSuite3> paramUtilsSite3 = 
		AEFX_SuiteScoper<PF_ParamUtilsSuite3> (in_data, kPFParamUtilsSuite, kPFParamUtilsSuiteVersion3, out_data);

	strncpy_s(params[COLOR_CORRECT_SLIDER1]->name, ColorSlider1[eCMYK], PF_MAX_EFFECT_PARAM_NAME_LEN);
	strncpy_s(params[COLOR_CORRECT_SLIDER2]->name, ColorSlider2[eCMYK], PF_MAX_EFFECT_PARAM_NAME_LEN);
	strncpy_s(params[COLOR_CORRECT_SLIDER3]->name, ColorSlider3[eCMYK], PF_MAX_EFFECT_PARAM_NAME_LEN);
	strncpy_s(params[COLOR_CORRECT_SLIDER4]->name, ColorSlider4[eCMYK], PF_MAX_EFFECT_PARAM_NAME_LEN);
	strncpy_s(params[COLOR_CORRECT_SLIDER5]->name, ColorSlider5[eCMYK], PF_MAX_EFFECT_PARAM_NAME_LEN);
	strncpy_s(params[COLOR_CORRECT_SLIDER6]->name, ColorSlider6[eCMYK], PF_MAX_EFFECT_PARAM_NAME_LEN);
	strncpy_s(params[COLOR_CORRECT_SLIDER7]->name, ColorSlider7[eCMYK], PF_MAX_EFFECT_PARAM_NAME_LEN);
	strncpy_s(params[COLOR_CORRECT_SLIDER8]->name, ColorSlider8[eCMYK], PF_MAX_EFFECT_PARAM_NAME_LEN);

	params[COLOR_CORRECT_SLIDER7]->u.sd.value = 0;
	params[COLOR_CORRECT_SLIDER7]->ui_flags &= ~PF_PUI_DISABLED;
	params[COLOR_CORRECT_SLIDER8]->u.fs_d.value = 0.0;
	params[COLOR_CORRECT_SLIDER8]->ui_flags &= ~PF_PUI_DISABLED;

	paramUtilsSite3->PF_UpdateParamUI(in_data->effect_ref, COLOR_CORRECT_SLIDER1, params[COLOR_CORRECT_SLIDER1]);
	paramUtilsSite3->PF_UpdateParamUI(in_data->effect_ref, COLOR_CORRECT_SLIDER2, params[COLOR_CORRECT_SLIDER2]);
	paramUtilsSite3->PF_UpdateParamUI(in_data->effect_ref, COLOR_CORRECT_SLIDER3, params[COLOR_CORRECT_SLIDER3]);
	paramUtilsSite3->PF_UpdateParamUI(in_data->effect_ref, COLOR_CORRECT_SLIDER4, params[COLOR_CORRECT_SLIDER4]);
	paramUtilsSite3->PF_UpdateParamUI(in_data->effect_ref, COLOR_CORRECT_SLIDER5, params[COLOR_CORRECT_SLIDER5]);
	paramUtilsSite3->PF_UpdateParamUI(in_data->effect_ref, COLOR_CORRECT_SLIDER6, params[COLOR_CORRECT_SLIDER6]);
	paramUtilsSite3->PF_UpdateParamUI(in_data->effect_ref, COLOR_CORRECT_SLIDER7, params[COLOR_CORRECT_SLIDER7]);
	paramUtilsSite3->PF_UpdateParamUI(in_data->effect_ref, COLOR_CORRECT_SLIDER8, params[COLOR_CORRECT_SLIDER8]);
}

inline void Set_ColorSpace_RGB (
	PF_InData	*in_data,
	PF_OutData	*out_data,
	PF_ParamDef	*params[]
) noexcept
{
	AEFX_SuiteScoper<PF_ParamUtilsSuite3> paramUtilsSite3 =
		AEFX_SuiteScoper<PF_ParamUtilsSuite3>(in_data, kPFParamUtilsSuite, kPFParamUtilsSuiteVersion3, out_data);

	strncpy_s(params[COLOR_CORRECT_SLIDER1]->name, ColorSlider1[eRGB], PF_MAX_EFFECT_PARAM_NAME_LEN);
	strncpy_s(params[COLOR_CORRECT_SLIDER2]->name, ColorSlider2[eRGB], PF_MAX_EFFECT_PARAM_NAME_LEN);
	strncpy_s(params[COLOR_CORRECT_SLIDER3]->name, ColorSlider3[eRGB], PF_MAX_EFFECT_PARAM_NAME_LEN);
	strncpy_s(params[COLOR_CORRECT_SLIDER4]->name, ColorSlider4[eRGB], PF_MAX_EFFECT_PARAM_NAME_LEN);
	strncpy_s(params[COLOR_CORRECT_SLIDER5]->name, ColorSlider5[eRGB], PF_MAX_EFFECT_PARAM_NAME_LEN);
	strncpy_s(params[COLOR_CORRECT_SLIDER6]->name, ColorSlider6[eRGB], PF_MAX_EFFECT_PARAM_NAME_LEN);
	strncpy_s(params[COLOR_CORRECT_SLIDER7]->name, ColorSlider7[eRGB], PF_MAX_EFFECT_PARAM_NAME_LEN);
	strncpy_s(params[COLOR_CORRECT_SLIDER8]->name, ColorSlider8[eRGB], PF_MAX_EFFECT_PARAM_NAME_LEN);

	params[COLOR_CORRECT_SLIDER7]->ui_flags |= PF_PUI_DISABLED;
	params[COLOR_CORRECT_SLIDER8]->ui_flags |= PF_PUI_DISABLED;

	paramUtilsSite3->PF_UpdateParamUI(in_data->effect_ref, COLOR_CORRECT_SLIDER1, params[COLOR_CORRECT_SLIDER1]);
	paramUtilsSite3->PF_UpdateParamUI(in_data->effect_ref, COLOR_CORRECT_SLIDER2, params[COLOR_CORRECT_SLIDER2]);
	paramUtilsSite3->PF_UpdateParamUI(in_data->effect_ref, COLOR_CORRECT_SLIDER3, params[COLOR_CORRECT_SLIDER3]);
	paramUtilsSite3->PF_UpdateParamUI(in_data->effect_ref, COLOR_CORRECT_SLIDER4, params[COLOR_CORRECT_SLIDER4]);
	paramUtilsSite3->PF_UpdateParamUI(in_data->effect_ref, COLOR_CORRECT_SLIDER5, params[COLOR_CORRECT_SLIDER5]);
	paramUtilsSite3->PF_UpdateParamUI(in_data->effect_ref, COLOR_CORRECT_SLIDER6, params[COLOR_CORRECT_SLIDER6]);
	paramUtilsSite3->PF_UpdateParamUI(in_data->effect_ref, COLOR_CORRECT_SLIDER7, params[COLOR_CORRECT_SLIDER7]);
	paramUtilsSite3->PF_UpdateParamUI(in_data->effect_ref, COLOR_CORRECT_SLIDER8, params[COLOR_CORRECT_SLIDER8]);
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
		case COLOR_CORRECT_SPACE_POPUP:
		{
			auto const& cType = params[COLOR_CORRECT_SPACE_POPUP]->u.pd.value;
			if (COLOR_SPACE_CMYK == static_cast<eCOLOR_SPACE_TYPE const>(cType - 1))
				Set_ColorSpace_CMYK (in_data, out_data, params);
			else
				Set_ColorSpace_RGB (in_data, out_data, params);
		}
		break;

		default:
		break;
	}

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
	CACHE_ALIGN PF_ParamDef param_copy[COLOR_CORRECT_TOTAL_PARAMS]{};
	MakeParamCopy(params, param_copy, COLOR_CORRECT_TOTAL_PARAMS);

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

			case PF_Cmd_EVENT:
				err = HandleEvent(in_data, out_data, params, output, reinterpret_cast<PF_EventExtra*>(extra));
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