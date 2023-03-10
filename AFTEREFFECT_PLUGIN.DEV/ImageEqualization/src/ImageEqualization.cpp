#include "ImageEqualization.hpp"
#include "PrSDKAESupport.h"

#ifdef _DEBUG
#include <atomic>
constexpr uint32_t traceBufferSize = 1024u;
std::atomic<uint32_t> traceCmdIdx = 0u;
CACHE_ALIGN static uint32_t traceCmdBuffer[traceBufferSize]{};

#ifndef _DBG_TRACE_COMMAND
#define _DBG_TRACE_COMMAND(cmd)				\
 {											\
	if (traceCmdIdx < traceBufferSize)		\
		traceCmdBuffer[traceCmdIdx++] = cmd;\
	else									\
		traceCmdIdx.exchange(0u);			\
 }
#endif // _DBG_TRACE_COMMAND
#else
#define _DBG_TRACE_COMMAND(cmd)		
#endif // _DEBUG


static PF_Err
About (
	PF_InData		*in_data,
	PF_OutData		*out_data,
	PF_ParamDef		*params[],
	PF_LayerDef		*output)
{
	PF_SPRINTF(out_data->return_msg,
		"%s, v%d.%d\r%s",
		strName,
		EqualizationFilter_VersionMajor,
		EqualizationFilter_VersionMinor,
		strCopyright);

	return PF_Err_NONE;
}


static PF_Err
GlobalSetup (
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
			EqualizationFilter_VersionMajor,
			EqualizationFilter_VersionMinor,
			EqualizationFilter_VersionSub,
			EqualizationFilter_VersionStage,
			EqualizationFilter_VersionBuild
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

		(*pixelFormatSuite->AddSupportedPixelFormat)(in_data->effect_ref, PrPixelFormat_VUYA_4444_8u_709);
		(*pixelFormatSuite->AddSupportedPixelFormat)(in_data->effect_ref, PrPixelFormat_VUYA_4444_8u);
//		(*pixelFormatSuite->AddSupportedPixelFormat)(in_data->effect_ref, PrPixelFormat_VUYA_4444_32f_709);
//		(*pixelFormatSuite->AddSupportedPixelFormat)(in_data->effect_ref, PrPixelFormat_VUYA_4444_32f);
//		(*pixelFormatSuite->AddSupportedPixelFormat)(in_data->effect_ref, PrPixelFormat_BGRA_4444_8u);
//		(*pixelFormatSuite->AddSupportedPixelFormat)(in_data->effect_ref, PrPixelFormat_BGRA_4444_16u);
//		(*pixelFormatSuite->AddSupportedPixelFormat)(in_data->effect_ref, PrPixelFormat_BGRA_4444_32f);
//		(*pixelFormatSuite->AddSupportedPixelFormat)(in_data->effect_ref, PrPixelFormat_RGB_444_10u);
	}


	return err;
}


static PF_Err
GlobalSetDown (
	PF_InData		*in_data,
	PF_OutData		*out_data)
{
	PF_Err	err = PF_Err_NONE;
	return err;
}



static PF_Err
ParamsSetup (
	PF_InData		*in_data,
	PF_OutData		*out_data,
	PF_ParamDef		*params[],
	PF_LayerDef		*output)
{
	PF_ParamDef	def;
	PF_Err		err = PF_Err_NONE;
	constexpr PF_ParamFlags flags{ PF_ParamFlag_SUPERVISE | PF_ParamFlag_CANNOT_TIME_VARY | PF_ParamFlag_CANNOT_INTERP };
	constexpr PF_ParamUIFlags ui_flags{ PF_PUI_NONE };
	constexpr PF_ParamUIFlags ui_flags_sliders{ ui_flags | PF_PUI_DISABLED };

	AEFX_CLR_STRUCT_EX(def);
	def.flags = flags;
	def.ui_flags = ui_flags;
	PF_ADD_POPUP(
		STR_EQ_ALGO_POPUP,          /* pop-up name          */
		IMAGE_EQ_ALGO_TOTALS,       /* number of operations */
		IMAGE_EQ_NONE,				/* default operation    */
		STR_EQ_ALGO_TYPE,           /* string for pop-up    */
		IMAGE_EQUALIZATION_POPUP_PRESET); /* control ID           */

	AEFX_CLR_STRUCT_EX(def);
	def.flags = flags;
	def.ui_flags = ui_flags_sliders;
	PF_ADD_SLIDER(
		STR_EQ_DARK_SLIDER,
		channelSliderMin,
		channelSliderMax,
		channelSliderMin,
		channelSliderMax,
		channelSLiderDef,
		IMAGE_EQUALIZATION_DARK_DETAILS_SLIDER);

	AEFX_CLR_STRUCT_EX(def);
	def.flags = flags;
	def.ui_flags = ui_flags_sliders;
	PF_ADD_SLIDER(
		STR_EQ_LIGHT_SLIDER,
		channelSliderMin,
		channelSliderMax,
		channelSliderMin,
		channelSliderMax,
		channelSLiderDef,
		IMAGE_EQUALIZATION_LIGHT_DETAILS_SLIDER);

	AEFX_CLR_STRUCT_EX(def);
	def.flags = flags;
	def.ui_flags = ui_flags_sliders;
	PF_ADD_SLIDER(
		STR_EQ_PEDESTAL_SLIDER,
		channelSliderMin,
		channelSliderMax,
		channelSliderMin,
		channelSliderMax,
		channelSLiderDef,
		IMAGE_EQUALIZATION_DARK_PEDESTAL_SLIDER);

	AEFX_CLR_STRUCT_EX(def);
	def.flags = flags;
	def.ui_flags = ui_flags;
	PF_ADD_CHECKBOXX(
		STR_EQ_CHECKBOX_FLICK,
		FALSE,
		0,
		IMAGE_EQUALIZATION_FLICK_REMOVE_CHECKBOX);

	out_data->num_params = IMAGE_EQUALIZATION_FILTER_TOTAL_PARAMS;
	return PF_Err_NONE;
}

static PF_Err
SequenceSetup (
	PF_InData		*in_data,
	PF_OutData		*out_data,
	PF_ParamDef		*params[],
	PF_LayerDef		*output)
{
	return PF_Err_NONE;
}


static PF_Err
SequenceReSetup (
	PF_InData		*in_data,
	PF_OutData		*out_data,
	PF_ParamDef		*params[],
	PF_LayerDef		*output)
{
	return PF_Err_NONE;
}


static PF_Err
SequenceFlatten (
	PF_InData		*in_data,
	PF_OutData		*out_data,
	PF_ParamDef		*params[],
	PF_LayerDef		*output)
{
	return PF_Err_NONE;
}


static PF_Err
SequenceSetDown (
	PF_InData		*in_data,
	PF_OutData		*out_data,
	PF_ParamDef		*params[],
	PF_LayerDef		*output)
{
	return PF_Err_NONE;
}


static PF_Err
Render (
	PF_InData		*in_data,
	PF_OutData		*out_data,
	PF_ParamDef		*params[],
	PF_LayerDef		*output)
{
	return ((PremierId == in_data->appl_id ? ProcessImgInPR (in_data, out_data, params, output) : ProcessImgInAE (in_data, out_data, params, output)));
}


static PF_Err
SmartPreRender (
	PF_InData		*in_data,
	PF_OutData		*out_data,
	PF_ParamDef		*params[],
	PF_LayerDef		*output)
{
	return PF_Err_NONE;
}


static PF_Err
SmartRender (
	PF_InData		*in_data,
	PF_OutData		*out_data,
	PF_ParamDef		*params[],
	PF_LayerDef		*output)
{
	return PF_Err_NONE;
}


static PF_Err
UserChangedParam (
	PF_InData						*in_data,
	PF_OutData						*out_data,
	PF_ParamDef						*params[],
	PF_LayerDef						*outputP,
	const PF_UserChangedParamExtra	*which_hitP
)
{
	PF_Err err = PF_Err_NONE;
	uint32_t updateUI = 0u;

	switch (which_hitP->param_index)
	{
		case IMAGE_EQUALIZATION_POPUP_PRESET:
		{
			AEFX_SuiteScoper<PF_ParamUtilsSuite3> paramUtilsSite3 =
				AEFX_SuiteScoper<PF_ParamUtilsSuite3>(in_data, kPFParamUtilsSuite, kPFParamUtilsSuiteVersion3, out_data);

			const ImageEqPopupAlgo algoType = static_cast<const ImageEqPopupAlgo>(params[IMAGE_EQUALIZATION_POPUP_PRESET]->u.pd.value - 1);
			if (IMAGE_EQ_MANUAL == algoType)
			{
				if (true == IsDisabledUI(params[IMAGE_EQUALIZATION_DARK_DETAILS_SLIDER]->ui_flags))
				{
					params[IMAGE_EQUALIZATION_DARK_DETAILS_SLIDER]->ui_flags &= ~PF_PUI_DISABLED;
					updateUI |= 0x1u;
				}
				if (true == IsDisabledUI(params[IMAGE_EQUALIZATION_LIGHT_DETAILS_SLIDER]->ui_flags))
				{
					params[IMAGE_EQUALIZATION_LIGHT_DETAILS_SLIDER]->ui_flags &= ~PF_PUI_DISABLED;
					updateUI |= 0x1u;
				}
				if (true == IsDisabledUI(params[IMAGE_EQUALIZATION_DARK_PEDESTAL_SLIDER]->ui_flags))
				{
					params[IMAGE_EQUALIZATION_DARK_PEDESTAL_SLIDER]->ui_flags &= ~PF_PUI_DISABLED;
					updateUI |= 0x1u;
				}
			}
			else
			{
				if (false == IsDisabledUI(params[IMAGE_EQUALIZATION_DARK_DETAILS_SLIDER]->ui_flags))
				{
					params[IMAGE_EQUALIZATION_DARK_DETAILS_SLIDER]->ui_flags |= PF_PUI_DISABLED;
					updateUI |= 0x1u;
				}
				if (false == IsDisabledUI(params[IMAGE_EQUALIZATION_LIGHT_DETAILS_SLIDER]->ui_flags))
				{
					params[IMAGE_EQUALIZATION_LIGHT_DETAILS_SLIDER]->ui_flags |= PF_PUI_DISABLED;
					updateUI |= 0x1u;
				}
				if (false == IsDisabledUI(params[IMAGE_EQUALIZATION_DARK_PEDESTAL_SLIDER]->ui_flags))
				{
					params[IMAGE_EQUALIZATION_DARK_PEDESTAL_SLIDER]->ui_flags |= PF_PUI_DISABLED;
					updateUI |= 0x1u;
				}
			}
			/* update UI */
			if (0u != updateUI)
			{
				paramUtilsSite3->PF_UpdateParamUI(in_data->effect_ref, IMAGE_EQUALIZATION_DARK_DETAILS_SLIDER,  params[IMAGE_EQUALIZATION_DARK_DETAILS_SLIDER]);
				paramUtilsSite3->PF_UpdateParamUI(in_data->effect_ref, IMAGE_EQUALIZATION_LIGHT_DETAILS_SLIDER, params[IMAGE_EQUALIZATION_LIGHT_DETAILS_SLIDER]);
				paramUtilsSite3->PF_UpdateParamUI(in_data->effect_ref, IMAGE_EQUALIZATION_DARK_PEDESTAL_SLIDER, params[IMAGE_EQUALIZATION_DARK_PEDESTAL_SLIDER]);
			}
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
	PF_LayerDef			*output
)
{
	return PF_Err_NONE;
}



PLUGIN_ENTRY_POINT_CALL  PF_Err
EffectMain (
	PF_Cmd			cmd,
	PF_InData		*in_data,
	PF_OutData		*out_data,
	PF_ParamDef		*params[],
	PF_LayerDef		*output,
	void			*extra)
{
	PF_Err		err = PF_Err_NONE;

	try {
		_DBG_TRACE_COMMAND(static_cast<uint32_t>(cmd));

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

			case PF_Cmd_PARAMS_SETUP:
				ERR(ParamsSetup(in_data, out_data, params, output));
			break;

			case PF_Cmd_SEQUENCE_SETUP:
				ERR(SequenceSetup(in_data, out_data, params, output));
			break;

			case PF_Cmd_SEQUENCE_RESETUP:
				ERR(SequenceReSetup(in_data, out_data, params, output));
			break;

			case PF_Cmd_SEQUENCE_FLATTEN:
				ERR(SequenceFlatten(in_data, out_data, params, output));
			break;

			case PF_Cmd_SEQUENCE_SETDOWN:
				ERR(SequenceSetDown(in_data, out_data, params, output));
			break;
			
			case PF_Cmd_RENDER:
				ERR(Render(in_data, out_data, params, output));
			break;

			case PF_Cmd_SMART_PRE_RENDER:
				ERR(SmartPreRender(in_data, out_data, params, output));
			break;

			case PF_Cmd_SMART_RENDER:
				ERR(SmartRender(in_data, out_data, params, output));
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