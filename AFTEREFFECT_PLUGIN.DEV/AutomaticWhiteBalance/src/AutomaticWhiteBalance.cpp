#include "AutomaticWhiteBalance.hpp"
#include "AlgMemoryHandler.hpp"


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
		AWB_VersionMajor,
		AWB_VersionMinor,
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
		PF_VERSION(	AWB_VersionMajor,
					AWB_VersionMinor,
					AWB_VersionSub,
					AWB_VersionStage,
					AWB_VersionBuild );

	out_data->out_flags = out_flags1;
	out_data->out_flags2 = out_flags2;

	/* For Premiere - declare supported pixel formats */
	if (PremierId == in_data->appl_id)
	{
		AEFX_SuiteScoper<PF_PixelFormatSuite1> pixelFormatSuite =
			AEFX_SuiteScoper<PF_PixelFormatSuite1>(in_data, kPFPixelFormatSuite, kPFPixelFormatSuiteVersion1, out_data);

		/*	Add the pixel formats we support in order of preference. */
		(*pixelFormatSuite->ClearSupportedPixelFormats)(in_data->effect_ref);

		(*pixelFormatSuite->AddSupportedPixelFormat)(in_data->effect_ref, PrPixelFormat_BGRA_4444_8u);
//		(*pixelFormatSuite->AddSupportedPixelFormat)(in_data->effect_ref, PrPixelFormat_BGRA_4444_16u);
//		(*pixelFormatSuite->AddSupportedPixelFormat)(in_data->effect_ref, PrPixelFormat_VUYA_4444_8u_709);
//		(*pixelFormatSuite->AddSupportedPixelFormat)(in_data->effect_ref, PrPixelFormat_VUYA_4444_8u);
//		(*pixelFormatSuite->AddSupportedPixelFormat)(in_data->effect_ref, PrPixelFormat_BGRA_4444_32f);
//		(*pixelFormatSuite->AddSupportedPixelFormat)(in_data->effect_ref, PrPixelFormat_VUYA_4444_32f_709);
//		(*pixelFormatSuite->AddSupportedPixelFormat)(in_data->effect_ref, PrPixelFormat_VUYA_4444_32f);
//		(*pixelFormatSuite->AddSupportedPixelFormat)(in_data->effect_ref, PrPixelFormat_RGB_444_10u);
	}

	/* pre-allocate double buffer for 1080p video */
	constexpr size_t memBytesSizeFor1080 = 1920 * 1080 * sizeof(PF_Pixel8);
	return (true == getMemoryHandler()->MemInit(memBytesSizeFor1080) ? PF_Err_NONE : PF_Err_OUT_OF_MEMORY);
}


static PF_Err
ParamsSetup(
	PF_InData		*in_data,
	PF_OutData		*out_data,
	PF_ParamDef		*params[],
	PF_LayerDef		*output)
{
	CACHE_ALIGN PF_ParamDef	def;
	PF_Err		err = PF_Err_NONE;

	constexpr PF_ParamFlags popup_flags = PF_ParamFlag_SUPERVISE | PF_ParamFlag_CANNOT_TIME_VARY | PF_ParamFlag_CANNOT_INTERP;
	constexpr PF_ParamUIFlags popup_ui_flags = PF_PUI_NONE;

	AEFX_CLR_STRUCT_EX(def);

	def.flags = popup_flags;
	def.ui_flags = popup_ui_flags;
	PF_ADD_POPUP(
		ILLUMINATE_NAME,		/* pop-up name			*/
		TOTAL_ILLUMINATES,		/* number of Illuminates*/
		ILLUMINATE_NONE,        /* default Illumnate	*/
		STR_ILLUMINATE,			/* string for pop-up	*/
		AWB_ILLUMINATE_POPUP);	/* control ID			*/

	AEFX_CLR_STRUCT_EX(def);

	def.flags = popup_flags;
	def.ui_flags = popup_ui_flags;
	PF_ADD_POPUP(
		CHROMATIC_NAME,		    /* pop-up name			*/
		TOTAL_CHROMATIC,		/* number of Adaptions  */
		CHROMATIC_CAT02,        /* default Adaptation	*/
		STR_CHROMATIC,			/* string for pop-up	*/
		AWB_CHROMATIC_POPUP);	/* control ID			*/


	AEFX_CLR_STRUCT_EX(def);

	def.flags = popup_flags;
	def.ui_flags = popup_ui_flags;
	PF_ADD_POPUP(
		COLOR_SPACE_NAME_OPT,		/* pop-up name			*/
		gTotalNumbersOfColorSpaces,	/* number of Illuminates*/
		gDefNumberOfColorSpace,		/* default color space	*/
		STR_COLOR_SPACE,    		/* string for pop-up	*/
		AWB_COLOR_SPACE_POPUP);		/* control ID			*/

	AEFX_CLR_STRUCT_EX(def);

	PF_ADD_SLIDER(
		THRESHOLD_NAME,
		gMinGrayThreshold,
		gMaxGrayThreshold,
		gMinGrayThreshold,
		gMaxGrayThreshold,
		gDefGrayThreshold,
		AWB_THRESHOLD_SLIDER);

	AEFX_CLR_STRUCT_EX(def);

	PF_ADD_SLIDER(
		ITERATIONS_NAME,
		iterMinCnt,
		iterMaxCnt,
		iterMinCnt,
		iterMaxCnt,
		iterDefCnt,
		AWB_ITERATIONS_SLIDER);

	out_data->num_params = AWB_TOTAL_CONTROLS;

	/* cleanup on exit for DBG purpose only */
	AEFX_CLR_STRUCT_EX(def);

	return err;
}


static PF_Err
Render (
	PF_InData		*in_data,
	PF_OutData		*out_data,
	PF_ParamDef		*params[],
	PF_LayerDef		*output)
{
	PF_Err	err = PF_Err_NONE;
	PF_Err errFormat = PF_Err_INVALID_INDEX;

	if (PremierId == in_data->appl_id)
	{
		/* This plugin called frop PR - check video fomat */
		AEFX_SuiteScoper<PF_PixelFormatSuite1> pixelFormatSuite =
			AEFX_SuiteScoper<PF_PixelFormatSuite1>(
				in_data,
				kPFPixelFormatSuite,
				kPFPixelFormatSuiteVersion1,
				out_data);

		PrPixelFormat destinationPixelFormat = PrPixelFormat_Invalid;
		if (PF_Err_NONE == (errFormat = pixelFormatSuite->GetPixelFormat(output, &destinationPixelFormat)))
		{
			err = ProcessImgInPR(in_data, out_data, params, output, destinationPixelFormat);
		} /* if (PF_Err_NONE == (errFormat = pixelFormatSuite->GetPixelFormat(output, &destinationPixelFormat))) */
		else
		{
			err = PF_Err_INVALID_INDEX;
		}
	}
	else
	{
		/* This plugin called from AE */
		ProcessImgInAE(in_data, out_data, params, output);
	}

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

	if (AWB_ILLUMINATE_POPUP == which_hitP->param_index)
	{

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
	PF_Err err = PF_Err_NONE;
	return err;
}



DllExport  PF_Err 
EntryPointFunc (	
	PF_Cmd			cmd,
	PF_InData		*in_data,
	PF_OutData		*out_data,
	PF_ParamDef		*params[],
	PF_LayerDef		*output,
	void			*extra)
{
#if !defined __INTEL_COMPILER 
	_MM_SET_FLUSH_ZERO_MODE(_MM_FLUSH_ZERO_ON);
	_MM_SET_DENORMALS_ZERO_MODE(_MM_DENORMALS_ZERO_ON);
#endif

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
			
			case PF_Cmd_PARAMS_SETUP:
				ERR(ParamsSetup(in_data, out_data, params, output));
			break;
			
			case PF_Cmd_RENDER:
				ERR(Render(in_data, out_data, params, output));
			break;

#if 0
			case PF_Cmd_USER_CHANGED_PARAM:
				ERR(UserChangedParam(in_data, out_data, params, output, reinterpret_cast<const PF_UserChangedParamExtra*>(extra)));
			break;

			// Handling this selector will ensure that the UI will be properly initialized,
			// even before the user starts changing parameters to trigger PF_Cmd_USER_CHANGED_PARAM
			case PF_Cmd_UPDATE_PARAMS_UI:
				ERR(UpdateParameterUI(in_data, out_data, params, output));
			break;
#endif
			default:
			break;
		}

	} catch (PF_Err& thrown_err) {
		err = (PF_Err_NONE != thrown_err ? thrown_err : PF_Err_INTERNAL_STRUCT_DAMAGED);
	}
	return err;
}