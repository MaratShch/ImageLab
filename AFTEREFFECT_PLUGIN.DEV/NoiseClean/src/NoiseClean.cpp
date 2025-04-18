#include "NoiseClean.hpp"
#include "NoiseCleanGUI.hpp"
#include "PrSDKAESupport.h"
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
		NoiseClean_VersionMajor,
		NoiseClean_VersionMinor,
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
		PF_VERSION(NoiseClean_VersionMajor,
			NoiseClean_VersionMinor,
			NoiseClean_VersionSub,
			NoiseClean_VersionStage,
			NoiseClean_VersionBuild);

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
//		(*pixelFormatSuite->AddSupportedPixelFormat)(in_data->effect_ref, PrPixelFormat_VUYA_4444_8u);
//		(*pixelFormatSuite->AddSupportedPixelFormat)(in_data->effect_ref, PrPixelFormat_VUYA_4444_32f_709);
//		(*pixelFormatSuite->AddSupportedPixelFormat)(in_data->effect_ref, PrPixelFormat_VUYA_4444_32f);
//		(*pixelFormatSuite->AddSupportedPixelFormat)(in_data->effect_ref, PrPixelFormat_BGRA_4444_8u);
//		(*pixelFormatSuite->AddSupportedPixelFormat)(in_data->effect_ref, PrPixelFormat_BGRA_4444_16u);
//		(*pixelFormatSuite->AddSupportedPixelFormat)(in_data->effect_ref, PrPixelFormat_BGRA_4444_32f);
	}

	return PF_Err_NONE;
}

static PF_Err
GlobalSetDown (void)
{
    UnloadMemoryInterfaceProvider();
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
	constexpr PF_ParamFlags   flags    = PF_ParamFlag_SUPERVISE | PF_ParamFlag_CANNOT_TIME_VARY | PF_ParamFlag_CANNOT_INTERP;
	constexpr PF_ParamUIFlags ui_flags = PF_PUI_NONE;
	constexpr PF_ParamUIFlags ui_flags_disables = ui_flags | PF_PUI_DISABLED;

	AEFX_INIT_PARAM_STRUCTURE (def, flags, ui_flags);
	PF_ADD_POPUP(
		strAlgoPopupName,			/* pop-up name			*/
		eNOISE_CLEAN_TOTAL_ALGOS,	/* number of variants	*/
		eNOISE_CLEAN_NONE,			/* default variant		*/
		strAlgoTypes,				/* string for pop-up	*/
		eNOISE_CLEAN_ALGO_POPUP);	/* control ID			*/

	AEFX_INIT_PARAM_STRUCTURE (def, flags, ui_flags_disables);
	PF_ADD_SLIDER(
		strWindowSlider1,
		cBilateralWindowMin,
		cBilateralWindowMax,
		cBilateralWindowMin,
		cBilateralWindowMax,
		cBilateralWindowDefault,
		eNOISE_CLEAN_BILATERAL_WINDOW_SLIDER);

	AEFX_INIT_PARAM_STRUCTURE(def, flags, ui_flags_disables);
	PF_ADD_SLIDER(
		strWindowSlider2,
		cDispersionMin,
		cDispersionMax,
		cDispersionMin,
		cDispersionMax,
		cDispersionDefault,
		eNOISE_CLEAN_ANYSOTROPIC_DISPERSION);

	AEFX_INIT_PARAM_STRUCTURE(def, flags, ui_flags_disables);
	PF_ADD_SLIDER(
		strWindowSlider3,
		cTimeStepMin,
		cTimeStepMax,
		cTimeStepMin,
		cTimeStepMax,
		cTimeStepDefault,
		eNOISE_CLEAN_ANYSOTROPIC_TIMESTEP);

	AEFX_INIT_PARAM_STRUCTURE(def, flags, ui_flags_disables);
	PF_ADD_SLIDER(
		strWindowSlider4,
		cNoiseLevelMin,
		cNoiseLevelMax,
		cNoiseLevelMin,
		cNoiseLevelMax,
		cNoiseLevelDefault,
		eNOISE_CLEAN_ANYSOTROPIC_NOISELEVEL);

	AEFX_INIT_PARAM_STRUCTURE(def, flags, ui_flags_disables);
	PF_ADD_SLIDER(
		strWindowSlider5,
		cNonLocalBayesNoiseStdMin,
		cNonLocalBayesNoiseStdMax,
		cNonLocalBayesNoiseStdMin,
		cNonLocalBayesNoiseStdMax,
		cNonLocalBayesNoiseStdDefault,
		eNOISE_CLEAN_NL_BAYES_SIGMA);

#ifdef _DEBUG
	// for DBG purpose only*
	AEFX_CLR_STRUCT_EX(def);
#endif

	out_data->num_params = eNOISE_CLEAN_TOTAL_PARAMS;
	return PF_Err_NONE;
}


static PF_Err
Render(
	PF_InData		*in_data,
	PF_OutData		*out_data,
	PF_ParamDef		*params[],
	PF_LayerDef		*output)
{
#if !defined __INTEL_COMPILER 
	_MM_SET_FLUSH_ZERO_MODE(_MM_FLUSH_ZERO_ON);
	_MM_SET_DENORMALS_ZERO_MODE(_MM_DENORMALS_ZERO_ON);
#endif
	return (PremierId == in_data->appl_id) ? ProcessImgInPR (in_data, out_data, params, output) : ProcessImgInAE (in_data, out_data, params, output);
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
	A_long updateUI = 0u;

	switch (which_hitP->param_index)
	{
		case eNOISE_CLEAN_ALGO_POPUP:
		{
			/* Algo type - popup chnaged. Let's check what we need to do */
			const auto algoType { static_cast<const eItem>(params[eNOISE_CLEAN_ALGO_POPUP]->u.pd.value - 1) };
			switch (algoType)
			{
				case eNOISE_CLEAN_NONE:
					SwitchToNoAlgo (in_data, out_data, params);
				break;

				case eNOISE_CLEAN_BILATERAL_LUMA:
				case eNOISE_CLEAN_BILATERAL_RGB:
					SwitchToBilateral (in_data, out_data, params);
				break;

				case eNOISE_CLEAN_PERONA_MALIK:
					SwitchToAnysotropic (in_data, out_data, params);
				break;

				case eNOISE_CLEAN_BSDE:
					SwitchToBSDE (in_data, out_data, params);
				break;

				case eNOISE_CLEAN_NONLOCAL_BAYES:
					SwitchToNonLocalBayes (in_data, out_data, params);
				break;

				case eNOISE_CLEAN_ADVANCED_DENOISE:
					SwitchToAdvanced (in_data, out_data, params);
				break;

				default:
					/* normally, we never should enter this case */
					err = PF_Err_INVALID_INDEX;
				break;
			} /* switch (algoType) */
		}
		break;

		default:
			/* nothing to do */
		break;
	} /* switch (which_hitP->param_index) */

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
	PF_Err		err = PF_Err_NONE;
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
                ERR(GlobalSetDown());
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