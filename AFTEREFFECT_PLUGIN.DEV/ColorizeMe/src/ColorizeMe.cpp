#include "ColorizeMe.hpp"


static PF_Err
About(
	PF_InData		*in_data,
	PF_OutData		*out_data,
	PF_ParamDef		*params[],
	PF_LayerDef		*output
) noexcept
{
	PF_SPRINTF(	out_data->return_msg,
				"%s, v%d.%d\r%s",
				strName,
				ColorizeMe_VersionMajor,
				ColorizeMe_VersionMinor,
				strCopyright);

	return PF_Err_NONE;
}


static PF_Err
GlobalSetup(
	PF_InData		*in_data,
	PF_OutData		*out_data,
	PF_ParamDef		*params[],
	PF_LayerDef		*output
) noexcept
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
		PF_VERSION(ColorizeMe_VersionMajor,
			       ColorizeMe_VersionMinor,
			       ColorizeMe_VersionSub,
			       ColorizeMe_VersionStage,
			       ColorizeMe_VersionBuild);


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
		(*pixelFormatSuite->AddSupportedPixelFormat)(in_data->effect_ref, PrPixelFormat_BGRA_4444_16u);
		(*pixelFormatSuite->AddSupportedPixelFormat)(in_data->effect_ref, PrPixelFormat_BGRA_4444_32f);
		(*pixelFormatSuite->AddSupportedPixelFormat)(in_data->effect_ref, PrPixelFormat_VUYA_4444_8u_709);
		(*pixelFormatSuite->AddSupportedPixelFormat)(in_data->effect_ref, PrPixelFormat_VUYA_4444_8u);
		(*pixelFormatSuite->AddSupportedPixelFormat)(in_data->effect_ref, PrPixelFormat_VUYA_4444_32f_709);
		(*pixelFormatSuite->AddSupportedPixelFormat)(in_data->effect_ref, PrPixelFormat_VUYA_4444_32f);
		(*pixelFormatSuite->AddSupportedPixelFormat)(in_data->effect_ref, PrPixelFormat_RGB_444_10u);
	}

	return PF_Err_NONE;
}

static PF_Err
GlobalSetdown(
	PF_InData* in_data
) noexcept
{
	return PF_Err_NONE;
}


static PF_Err
ParamsSetup(
	PF_InData		*in_data,
	PF_OutData		*out_data,
	PF_ParamDef		*params[],
	PF_LayerDef		*output
) noexcept
{
	PF_ParamDef	def;
	PF_Err		err = PF_Err_NONE;
	constexpr PF_ParamFlags flags = PF_ParamFlag_SUPERVISE | PF_ParamFlag_CANNOT_TIME_VARY | PF_ParamFlag_CANNOT_INTERP;
	constexpr PF_ParamUIFlags ui_flags = PF_PUI_NONE;

	def.flags = flags;
	def.ui_flags = ui_flags;

	AEFX_CLR_STRUCT_EX(def);

	PF_ADD_BUTTON(
		ButtonLutParam,
		ButtonLut,
		ui_flags,
		flags,
		COLOR_LUT_FILE_BUTTON
	);

	AEFX_CLR_STRUCT_EX(def);

	PF_ADD_CHECKBOX(
		CheckBoxParamName,
		CheckBoxName,
		FALSE,
		0,
		COLOR_NEGATE_CHECKBOX
	);

	AEFX_CLR_STRUCT_EX(def);

	PF_ADD_POPUP(
		InterpType,						/* pop-up name			*/
		COLOR_INTERPOLATION_MAX_TYPES,	/* numbver of variants	*/
		COLOR_INTERPOLATION_FAST,		/* default variant		*/
		Interpolation,					/* string for pop-up	*/
		COLOR_INTERPOLATION_POPUP);		/* control ID			*/

	AEFX_CLR_STRUCT_EX(def);

	PF_ADD_SLIDER(
		RedPedestalName,
		gMinRedPedestal,
		gMaxRedPedestal,
		gMinRedPedestal,
		gMaxRedPedestal,
		gDefRedPedestal,
		COLOR_RED_PEDESTAL_SLIDER);

	AEFX_CLR_STRUCT_EX(def);

	PF_ADD_SLIDER(
		GreenPedestalName,
		gMinGreenPedestal,
		gMaxGreenPedestal,
		gMinGreenPedestal,
		gMaxGreenPedestal,
		gDefGreenPedestal,
		COLOR_GREEN_PEDESTAL_SLIDER);

	AEFX_CLR_STRUCT_EX(def);

	PF_ADD_SLIDER(
		BluePedestalName,
		gMinBluePedestal,
		gMaxBluePedestal,
		gMinBluePedestal,
		gMaxBluePedestal,
		gDefBluePedestal,
		COLOR_BLUE_PEDESTAL_SLIDER);

	out_data->num_params = COLOR_TOTAL_PARAMS;

	/* cleanup on exit (for DBG purpose only) */
	AEFX_CLR_STRUCT_EX(def);
	return err;
}


static PF_Err
Render(
	PF_InData		*in_data,
	PF_OutData		*out_data,
	PF_ParamDef		*params[],
	PF_LayerDef		*output
) noexcept
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
//			err = ProcessImgInPR(in_data, out_data, params, output, destinationPixelFormat);
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
//		err = ProcessImgInAE(in_data, out_data, params, output);
	}

	return err;
}


DllExport	PF_Err
EntryPointFunc(
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