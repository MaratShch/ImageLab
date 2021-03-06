#include "Convolution.hpp"
#include "Kernels.hpp"

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
			Convolution_VersionMajor,
			Convolution_VersionMinor,
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

	/* Initialize Kernels */
	InitKernelsFactory();

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
		PF_VERSION (
			Convolution_VersionMajor,
			Convolution_VersionMinor,
			Convolution_VersionSub,
			Convolution_VersionStage,
			Convolution_VersionBuild
		);

	out_data->out_flags  = out_flags1;
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
		(*pixelFormatSuite->AddSupportedPixelFormat)(in_data->effect_ref, PrPixelFormat_VUYA_4444_8u_709);
		(*pixelFormatSuite->AddSupportedPixelFormat)(in_data->effect_ref, PrPixelFormat_VUYA_4444_8u);
		(*pixelFormatSuite->AddSupportedPixelFormat)(in_data->effect_ref, PrPixelFormat_BGRA_4444_32f);
		(*pixelFormatSuite->AddSupportedPixelFormat)(in_data->effect_ref, PrPixelFormat_VUYA_4444_32f_709);
		(*pixelFormatSuite->AddSupportedPixelFormat)(in_data->effect_ref, PrPixelFormat_VUYA_4444_32f);
		(*pixelFormatSuite->AddSupportedPixelFormat)(in_data->effect_ref, PrPixelFormat_RGB_444_10u);
	}

	return err;
}


static PF_Err
GlobalSetdown (
	PF_InData* in_data
)
{
	FreeKernelsFactory();
	return PF_Err_NONE;
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
	constexpr PF_ParamFlags flags = PF_ParamFlag_SUPERVISE | PF_ParamFlag_CANNOT_TIME_VARY | PF_ParamFlag_CANNOT_INTERP;
	constexpr PF_ParamUIFlags ui_flags = PF_PUI_NONE;

	AEFX_CLR_STRUCT_EX(def);

	def.flags = flags;
	def.ui_flags = ui_flags;

	PF_ADD_POPUP(
		KernelType,				/* pop-up name			*/		
		KERNEL_CONV_SIZE,		/* number of Kernels	*/
		KERNEL_CONV_SHARP_3x3,	/* default Kernel		*/
		strKernels,				/* string for pop-up	*/	
		KERNEL_CONV_DISK_ID);	/* control ID			*/			

	out_data->num_params = CONVLOVE_NUM_PARAMS;

	return err;
}


static PF_Err
Render(
	PF_InData		*in_data,
	PF_OutData		*out_data,
	PF_ParamDef		*params[],
	PF_LayerDef		*output)
{
	PF_Err	err = PF_Err_NONE;
	PF_Err errFormat = PF_Err_INVALID_INDEX;

	const PF_ParamValue convKernelType = params[KERNEL_POPUP]->u.pd.value;
	const uint32_t choosedKernel = static_cast<uint32_t>(convKernelType - 1);

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
			ProcessImgInPR(in_data, out_data, params, output, destinationPixelFormat, choosedKernel);
		} /* if (PF_Err_NONE == (errFormat = pixelFormatSuite->GetPixelFormat(output, &destinationPixelFormat))) */
		else
		{
			err = PF_Err_INVALID_INDEX;
		}
	}
	else
	{
		/* This plugin called from AE */
		ProcessImgInAE(in_data, out_data, params, output, choosedKernel);
	}

	return err;
}

#if 0

static PF_Err
PreRender(
	PF_InData				*in_data,
	PF_OutData				*out_data,
	PF_PreRenderExtra		*extraP
)
{
	PF_Err	err = PF_Err_NONE;

	return err;
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

#endif

static PF_Err
UserChangedParam(
	PF_InData						*in_data,
	PF_OutData						*out_data,
	PF_ParamDef						*params[],
	PF_LayerDef						*outputP,
	const PF_UserChangedParamExtra	*which_hitP
)
{
	PF_Err	err = PF_Err_NONE;

	if (KERNEL_POPUP == which_hitP->param_index)
	{
#ifdef _DEBUG
		const uint32_t kernelIdx = params[KERNEL_POPUP]->u.pd.value;
#endif
		if (PremierId == in_data->appl_id)
		{
			// In Premiere Pro, this message will appear in the Events panel
			PF_STRCPY(out_data->return_msg, "Kernel Type changed. New rendering will be forsed");
		}
		else
		{
			AEFX_SuiteScoper<PF_ParamUtilsSuite3> paramUtilsSuite =
				AEFX_SuiteScoper<PF_ParamUtilsSuite3>(
					in_data,
					kPFParamUtilsSuite,
					kPFParamUtilsSuiteVersion3,
					out_data);

			params[KERNEL_POPUP]->uu.change_flags = PF_ChangeFlag_CHANGED_VALUE;
			err = paramUtilsSuite->PF_UpdateParamUI (in_data->effect_ref, KERNEL_POPUP, params[KERNEL_POPUP]);
		}
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
	CACHE_ALIGN PF_ParamDef	param_copy[CONVLOVE_NUM_PARAMS] = { 0 };
	PF_Err		err = PF_Err_NONE;
	constexpr PF_OutFlags newFlag = PF_OutFlag_REFRESH_UI | PF_OutFlag_FORCE_RERENDER;

	param_copy[CONVOLUTION_INPUT] = *(params[CONVOLUTION_INPUT]);
	param_copy[KERNEL_POPUP]      = *(params[KERNEL_POPUP]);

	AEFX_SuiteScoper<PF_ParamUtilsSuite3> paramUtilsSuite =
		AEFX_SuiteScoper<PF_ParamUtilsSuite3>(
			in_data,
			kPFParamUtilsSuite,
			kPFParamUtilsSuiteVersion3,
			out_data);

	err = paramUtilsSuite->PF_UpdateParamUI (in_data->effect_ref, KERNEL_POPUP, &param_copy[KERNEL_POPUP]);
	out_data->out_flags |= newFlag;

	return err;
}



PLUGIN_ENTRY_POINT_CALL	PF_Err
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
		switch (cmd) {
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
#if 0			
			case PF_Cmd_SMART_PRE_RENDER:
				ERR(PreRender(in_data, out_data, (PF_PreRenderExtra*)extra));
			break;

			case PF_Cmd_SMART_RENDER:
				ERR(SmartRender(in_data, out_data, (PF_SmartRenderExtra*)extra));
			break;
#endif
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
	}
	catch (PF_Err& thrown_err)
	{
		err = thrown_err;
	}

	return err;
}