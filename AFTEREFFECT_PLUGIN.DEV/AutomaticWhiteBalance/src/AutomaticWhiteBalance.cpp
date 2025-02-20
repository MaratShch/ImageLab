#include "AutomaticWhiteBalance.hpp"
#include "ImageLabMemInterface.hpp"

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
    if (false == LoadMemoryInterfaceProvider(in_data))
        return PF_Err_INTERNAL_STRUCT_DAMAGED;

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

        (*pixelFormatSuite->AddSupportedPixelFormat)(in_data->effect_ref, PrPixelFormat_VUYA_4444_8u_709);
        (*pixelFormatSuite->AddSupportedPixelFormat)(in_data->effect_ref, PrPixelFormat_VUYA_4444_8u);
        (*pixelFormatSuite->AddSupportedPixelFormat)(in_data->effect_ref, PrPixelFormat_VUYA_4444_32f_709);
        (*pixelFormatSuite->AddSupportedPixelFormat)(in_data->effect_ref, PrPixelFormat_VUYA_4444_32f);
        (*pixelFormatSuite->AddSupportedPixelFormat)(in_data->effect_ref, PrPixelFormat_BGRA_4444_8u);
        (*pixelFormatSuite->AddSupportedPixelFormat)(in_data->effect_ref, PrPixelFormat_BGRA_4444_16u);
        (*pixelFormatSuite->AddSupportedPixelFormat)(in_data->effect_ref, PrPixelFormat_BGRA_4444_32f);
    }

	return PF_Err_NONE;
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

	constexpr PF_ParamFlags popup_flags = PF_ParamFlag_SUPERVISE | PF_ParamFlag_CANNOT_TIME_VARY | PF_ParamFlag_CANNOT_INTERP;
	constexpr PF_ParamUIFlags popup_ui_flags = PF_PUI_NONE;

    AEFX_INIT_PARAM_STRUCTURE(def, popup_flags, popup_ui_flags);
	PF_ADD_POPUP(
		ILLUMINATE_NAME,		/* pop-up name			*/
		TOTAL_ILLUMINATES,		/* number of Illuminates*/
		DAYLIGHT,               /* default Illumnate	*/
		STR_ILLUMINATE,			/* string for pop-up	*/
		AWB_ILLUMINATE_POPUP);	/* control ID			*/

    AEFX_INIT_PARAM_STRUCTURE(def, popup_flags, popup_ui_flags);
	PF_ADD_POPUP(
		CHROMATIC_NAME,		    /* pop-up name			*/
		TOTAL_CHROMATIC,		/* number of Adaptions  */
		CHROMATIC_CAT02,        /* default Adaptation	*/
		STR_CHROMATIC,			/* string for pop-up	*/
		AWB_CHROMATIC_POPUP);	/* control ID			*/


    AEFX_INIT_PARAM_STRUCTURE(def, popup_flags, popup_ui_flags);
	PF_ADD_POPUP(
		COLOR_SPACE_NAME_OPT,		/* pop-up name			*/
		gTotalNumbersOfColorSpaces,	/* number of Illuminates*/
		gDefNumberOfColorSpace,		/* default color space	*/
		STR_COLOR_SPACE,    		/* string for pop-up	*/
		AWB_COLOR_SPACE_POPUP);		/* control ID			*/

    AEFX_INIT_PARAM_STRUCTURE(def, popup_flags, popup_ui_flags);
	PF_ADD_SLIDER(
		THRESHOLD_NAME,
		gMinGrayThreshold,
		gMaxGrayThreshold,
		gMinGrayThreshold,
		gMaxGrayThreshold,
		gDefGrayThreshold,
		AWB_THRESHOLD_SLIDER);

    AEFX_INIT_PARAM_STRUCTURE(def, popup_flags, popup_ui_flags);
 	PF_ADD_SLIDER(
		ITERATIONS_NAME,
		iterMinCnt,
		iterMaxCnt,
		iterMinCnt,
		iterMaxCnt,
		iterDefCnt,
		AWB_ITERATIONS_SLIDER);

	out_data->num_params = AWB_TOTAL_CONTROLS;

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
			err = ProcessImgInPR (in_data, out_data, params, output, destinationPixelFormat);
		} /* if (PF_Err_NONE == (errFormat = pixelFormatSuite->GetPixelFormat(output, &destinationPixelFormat))) */
		else
		{
			// In Premiere Pro, this message will appear in the Events panel
			PF_STRCPY(out_data->return_msg, "Unsupoorted image format...");
			err = PF_Err_INVALID_INDEX;
		}
	}
	else
	{
		/* This plugin called from AE */
		err = ProcessImgInAE (in_data, out_data, params, output);
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

	AEFX_SuiteScoper<PF_ParamUtilsSuite3> paramUtilsSuite =
		AEFX_SuiteScoper<PF_ParamUtilsSuite3>(
			in_data,
			kPFParamUtilsSuite,
			kPFParamUtilsSuiteVersion3,
			out_data);

	switch (which_hitP->param_index)
	{
		case AWB_ILLUMINATE_POPUP:
		{
#ifdef _DEBUG
			const uint32_t illuminateIdx = params[AWB_ILLUMINATE_POPUP]->u.pd.value;
#endif
			params[AWB_ILLUMINATE_POPUP]->uu.change_flags = PF_ChangeFlag_CHANGED_VALUE;
			err = paramUtilsSuite->PF_UpdateParamUI(in_data->effect_ref, AWB_ILLUMINATE_POPUP, params[AWB_ILLUMINATE_POPUP]);
		}
		break;

		case AWB_CHROMATIC_POPUP:
		{
#ifdef _DEBUG
			const uint32_t chromaticIdx = params[AWB_CHROMATIC_POPUP]->u.pd.value;
#endif
			params[AWB_CHROMATIC_POPUP]->uu.change_flags = PF_ChangeFlag_CHANGED_VALUE;
			err = paramUtilsSuite->PF_UpdateParamUI(in_data->effect_ref, AWB_CHROMATIC_POPUP, params[AWB_CHROMATIC_POPUP]);
		}
		break;

		case AWB_COLOR_SPACE_POPUP:
		{
#ifdef _DEBUG
			const uint32_t colorSpaceIdx = params[AWB_COLOR_SPACE_POPUP]->u.pd.value;
#endif
			params[AWB_COLOR_SPACE_POPUP]->uu.change_flags = PF_ChangeFlag_CHANGED_VALUE;
			err = paramUtilsSuite->PF_UpdateParamUI(in_data->effect_ref, AWB_COLOR_SPACE_POPUP, params[AWB_COLOR_SPACE_POPUP]);
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
	PF_ParamDef	param_copy[AWB_TOTAL_CONTROLS] {};
	PF_Err		err = PF_Err_NONE;
	constexpr PF_OutFlags newFlag = PF_OutFlag_REFRESH_UI | PF_OutFlag_FORCE_RERENDER;

	for (int32_t i = 0; i < AWB_TOTAL_CONTROLS; i++)
		param_copy[i] = *(params[i]);


	AEFX_SuiteScoper<PF_ParamUtilsSuite3> paramUtilsSuite =
		AEFX_SuiteScoper<PF_ParamUtilsSuite3>(
			in_data,
			kPFParamUtilsSuite,
			kPFParamUtilsSuiteVersion3,
			out_data);

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
		pixelFormatSuite->GetPixelFormat(output, &destinationPixelFormat);
		if (PrPixelFormat_VUYA_4444_8u == destinationPixelFormat || PrPixelFormat_VUYA_4444_8u_709 == destinationPixelFormat)
		{
			/* disable color space popup */
			param_copy[AWB_COLOR_SPACE_POPUP].ui_flags |= PF_PUI_DISABLED;
		}
		else
		{
			/* enable color space popup */
			param_copy[AWB_COLOR_SPACE_POPUP].ui_flags &= ~PF_PUI_DISABLED;
		}

	}

	err |= paramUtilsSuite->PF_UpdateParamUI(in_data->effect_ref, AWB_COLOR_SPACE_POPUP, &param_copy[AWB_COLOR_SPACE_POPUP]);

	out_data->out_flags |= newFlag;

	return err;
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

	} catch (PF_Err& thrown_err) {
		err = (PF_Err_NONE != thrown_err ? thrown_err : PF_Err_INTERNAL_STRUCT_DAMAGED);
	}
	return err;
}