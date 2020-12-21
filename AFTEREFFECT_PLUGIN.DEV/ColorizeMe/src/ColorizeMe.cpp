#include "ColorizeMe.hpp"
#include "CubeLUT.h"
#include <Windows.h>

static PF_Err
About(
	PF_InData		*in_data,
	PF_OutData		*out_data,
	PF_ParamDef		*params[],
	PF_LayerDef		*output
) 
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
) 
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
) 
{
	return PF_Err_NONE;
}


static PF_Err
ParamsSetup(
	PF_InData		*in_data,
	PF_OutData		*out_data,
	PF_ParamDef		*params[],
	PF_LayerDef		*output
) 
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
		0,
		PF_ParamFlag_SUPERVISE,
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
		COLOR_INTERPOLATION_LINEAR,		/* default variant		*/
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
	
	AEFX_CLR_STRUCT_EX(def);

	PF_ADD_BUTTON(
		pedestalResetName,
		pedestalReset,
		0,
		PF_ParamFlag_SUPERVISE,
		COLOR_PEDESTAL_RESET_BUTTON
	);

	out_data->num_params = COLOR_TOTAL_PARAMS;

	/* cleanup on exit (for DBG purpose only) */
	AEFX_CLR_STRUCT_EX(def);
	return err;
}

static bool
IsProcessingActivated(
	PF_InData	*in_data,
	PF_ParamDef	*params[]
) 
{
	const bool bProc =
		(0 == params[COLOR_RED_PEDESTAL_SLIDER  ]->u.sd.value &&
		 0 == params[COLOR_GREEN_PEDESTAL_SLIDER]->u.sd.value &&
         0 == params[COLOR_BLUE_PEDESTAL_SLIDER ]->u.sd.value &&
			(nullptr == in_data->sequence_data ||
			(PF_GET_HANDLE_SIZE(in_data->sequence_data) == sizeof(CubeLUT) && (static_cast<CubeLUT*>(*in_data->sequence_data)->LutIsLoaded() == false)))
		) ? false : true;
	return bProc;
}


static PF_Err
Render(
	PF_InData		*in_data,
	PF_OutData		*out_data,
	PF_ParamDef		*params[],
	PF_LayerDef		*output
) 
{
	PF_Err	err = PF_Err_NONE;
	PF_Err errFormat = PF_Err_INVALID_INDEX;

	if (false == IsProcessingActivated(in_data, params))
	{
		/* no acion items, just copy input buffer to outpu without processing */
		if (PremierId != in_data->appl_id)
		{
			AEFX_SuiteScoper<PF_WorldTransformSuite1> worldTransformSuite =
				AEFX_SuiteScoper<PF_WorldTransformSuite1>(in_data, kPFPixelFormatSuite, kPFPixelFormatSuiteVersion1, out_data);

			err = (PF_Quality_HI == in_data->quality) ?
				worldTransformSuite->copy_hq(in_data->effect_ref, &params[COLOR_INPUT]->u.ld, output, NULL, NULL) :
				worldTransformSuite->copy(in_data->effect_ref, &params[COLOR_INPUT]->u.ld, output, NULL, NULL);
		}
		else {
			err = PF_COPY(&params[COLOR_INPUT]->u.ld, output, NULL,	NULL);
		}
	} /* if (false == IsProcessingActivated(in_data, params)) */
	else
	{
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
				// In Premiere Pro, this message will appear in the Events panel
				PF_STRCPY(out_data->return_msg, "Unsupported image format...");
				err = PF_Err_INVALID_INDEX;
			}
		}
		else
		{
			/* This plugin called from AE */
			err = ProcessImgInAE(in_data, out_data, params, output);
		}
	}

	return err;
}


static PF_Err
SequenceSetup(
	PF_InData		*in_data,
	PF_OutData		*out_data,
	PF_ParamDef		*params[],
	PF_LayerDef		*output
) 
{
	PF_Err err = PF_Err_NONE;
	CubeLUT* pLutObj = nullptr;

	if (out_data->sequence_data) {
		PF_DISPOSE_HANDLE_EX(out_data->sequence_data);
	}
	out_data->sequence_data = PF_NEW_HANDLE(LUT_OBJ_SIZE);
	if (!out_data->sequence_data) {
		return PF_Err_INTERNAL_STRUCT_DAMAGED;
	}

	memset(*out_data->sequence_data, 0, LUT_OBJ_SIZE);

	return err;
}


static PF_Err
SequenceReSetup(
	PF_InData		*in_data,
	PF_OutData		*out_data,
	PF_ParamDef		*params[],
	PF_LayerDef		*output
) 
{
	if (!in_data->sequence_data) {
		return SequenceSetup(in_data, out_data, params, output);
	}
	return PF_Err_NONE;
}


static PF_Err
SequenceSetdown(
	PF_InData		*in_data,
	PF_OutData		*out_data,
	PF_ParamDef		*params[],
	PF_LayerDef		*output
) 
{
	PF_Err err = PF_Err_NONE;

	if (PF_GET_HANDLE_SIZE(in_data->sequence_data) == sizeof(CubeLUT))
	{
		CubeLUT* pCubeLUT = static_cast<CubeLUT*>(*in_data->sequence_data);
		if (0xDEADBEEF == pCubeLUT->uId)
			pCubeLUT->~CubeLUT();

		/* free memory handler with memory cleanup */
		PF_DISPOSE_HANDLE_EX(in_data->sequence_data);
		out_data->sequence_data = nullptr;
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
	CubeLUT* pCubeLUT = nullptr;

	switch (which_hitP->param_index)
	{
		case COLOR_LUT_FILE_BUTTON:
		{
			const std::string lutName = GetLutFileName();
			if (!lutName.empty() && PF_GET_HANDLE_SIZE(in_data->sequence_data) == sizeof(CubeLUT))
			{
				pCubeLUT = static_cast<CubeLUT*>(*out_data->sequence_data);
				if (0xDEADBEEF == pCubeLUT->uId && lutName == pCubeLUT->GetLutName())
					break; /* if object already cretaed and same LUT required - just ignore the action */

				/* create new object with placement new */
				pCubeLUT = new(*out_data->sequence_data) CubeLUT;
				const CubeLUT::LUTState loadStatus = (nullptr != pCubeLUT ? pCubeLUT->LoadCubeFile(lutName) : CubeLUT::GenericError);
				err = (CubeLUT::OK == loadStatus || CubeLUT::AlreadyLoaded == loadStatus) ? PF_Err_NONE : PF_Err_INVALID_INDEX;
			}
		}
		break;

		case COLOR_PEDESTAL_RESET_BUTTON:
			params[COLOR_RED_PEDESTAL_SLIDER  ]->u.sd.value = 0;
			params[COLOR_RED_PEDESTAL_SLIDER  ]->uu.change_flags = PF_ChangeFlag_CHANGED_VALUE;
			params[COLOR_GREEN_PEDESTAL_SLIDER]->u.sd.value = 0;
			params[COLOR_GREEN_PEDESTAL_SLIDER]->uu.change_flags = PF_ChangeFlag_CHANGED_VALUE;
			params[COLOR_BLUE_PEDESTAL_SLIDER ]->u.sd.value = 0;
			params[COLOR_BLUE_PEDESTAL_SLIDER ]->uu.change_flags = PF_ChangeFlag_CHANGED_VALUE;
		break;

		default:
		break;
	} /* switch (which_hitP->param_index) */

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

			case PF_Cmd_SEQUENCE_SETUP:
				ERR(SequenceSetup(in_data, out_data, params, output));
			break;

			case PF_Cmd_SEQUENCE_RESETUP:
				ERR(SequenceReSetup(in_data, out_data, params, output));
			break;

			case PF_Cmd_SEQUENCE_SETDOWN:
				ERR(SequenceSetdown(in_data, out_data, params, output));
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
	catch (PF_Err &thrown_err)
	{
		err = thrown_err;
	}

	return err;
}

#ifdef AE_OS_WIN
BOOL WINAPI DllMain(HINSTANCE hDLL, DWORD dwReason, LPVOID lpReserved)
{
	HINSTANCE my_instance_handle = (HINSTANCE)0;

	switch (dwReason)
	{
	case DLL_PROCESS_ATTACH:
		my_instance_handle = hDLL;
		break;

	case DLL_THREAD_ATTACH:
		my_instance_handle = hDLL;
		break;
	case DLL_THREAD_DETACH:
		my_instance_handle = 0;
		break;
	case DLL_PROCESS_DETACH:
		my_instance_handle = 0;
		break;
		break;
	}
	return(TRUE);
}
#endif

