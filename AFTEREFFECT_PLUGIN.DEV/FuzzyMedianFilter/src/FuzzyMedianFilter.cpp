#include "FuzzyMedianFilter.hpp"
#include "FuzzyMedianFilterEnum.hpp"
#include "PrSDKAESupport.h"
#include "ImageLabMemInterface.hpp"


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
		FuzzyMedian_VersionMajor,
		FuzzyMedian_VersionMinor,
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

    if (false == LoadMemoryInterfaceProvider(in_data))
        return PF_Err_INTERNAL_STRUCT_DAMAGED;

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
			FuzzyMedian_VersionMajor,
			FuzzyMedian_VersionMinor,
			FuzzyMedian_VersionSub,
			FuzzyMedian_VersionStage,
			FuzzyMedian_VersionBuild
		);

	out_data->out_flags  = out_flags1;
	out_data->out_flags2 = out_flags2;

	/* For Premiere - declare supported pixel formats */
	if (PremierId == in_data->appl_id)
	{
		auto const& pixelFormatSuite{ AEFX_SuiteScoper<PF_PixelFormatSuite1>(in_data, kPFPixelFormatSuite, kPFPixelFormatSuiteVersion1, out_data) };

		/*	Add the pixel formats we support in order of preference. */
		(*pixelFormatSuite->ClearSupportedPixelFormats)(in_data->effect_ref);

//		(*pixelFormatSuite->AddSupportedPixelFormat)(in_data->effect_ref, PrPixelFormat_BGRA_4444_8u);
//		(*pixelFormatSuite->AddSupportedPixelFormat)(in_data->effect_ref, PrPixelFormat_BGRA_4444_16u);
//		(*pixelFormatSuite->AddSupportedPixelFormat)(in_data->effect_ref, PrPixelFormat_BGRA_4444_32f);
		(*pixelFormatSuite->AddSupportedPixelFormat)(in_data->effect_ref, PrPixelFormat_VUYA_4444_8u_709);
//		(*pixelFormatSuite->AddSupportedPixelFormat)(in_data->effect_ref, PrPixelFormat_VUYA_4444_8u);
//		(*pixelFormatSuite->AddSupportedPixelFormat)(in_data->effect_ref, PrPixelFormat_VUYA_4444_32f_709);
//		(*pixelFormatSuite->AddSupportedPixelFormat)(in_data->effect_ref, PrPixelFormat_VUYA_4444_32f);
//		(*pixelFormatSuite->AddSupportedPixelFormat)(in_data->effect_ref, PrPixelFormat_RGB_444_10u);
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
    PF_Err		err = PF_Err_NONE;

    constexpr PF_ParamFlags popup_flags = PF_ParamFlag_SUPERVISE | PF_ParamFlag_CANNOT_TIME_VARY | PF_ParamFlag_CANNOT_INTERP;
    constexpr PF_ParamUIFlags popup_ui_flags = PF_PUI_NONE;

    AEFX_INIT_PARAM_STRUCTURE(def, popup_flags, popup_ui_flags);
    PF_ADD_POPUP(
        strPopupName,			            // pop-up name
        eFUZZY_FILTER_TOTAL_VARIANTS,	    // number of variants
        eFUZZY_FILTER_BYPASSED,			    // default variant
        strWindowSizes,			            // string for pop-up
        eFUZZY_MEDIAN_FILTER_KERNEL_SIZE);	// control ID

    AEFX_INIT_PARAM_STRUCTURE(def, popup_flags, popup_ui_flags);
    PF_ADD_FLOAT_SLIDERX(
        strSliderName,
        fSliderValMin,
        fSliderValMax,
        fSliderValMin,
        fSliderValMax,
        fSliderValDefault,
        PF_Precision_TENTHS,
        0,
        0,
        eFUZZY_MEDIAN_FILTER_SIGMA_VALUE);

    out_data->num_params = eFUZZY_MEDIAN_TOTAL_CONTROLS;

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

	return ((PremierId == in_data->appl_id ? ProcessImgInPR(in_data, out_data, params, output) : ProcessImgInAE(in_data, out_data, params, output)));
}



static PF_Err
UpdateParameterUI(
    PF_InData			*in_data,
    PF_OutData			*out_data,
    PF_ParamDef			*params[],
    PF_LayerDef			*output
)
{
    // nothing TODO
    return PF_Err_NONE;
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

    return err;
}


static PF_Err
PreRender(
    PF_InData			*in_data,
    PF_OutData			*out_data,
    PF_PreRenderExtra	*extra
)
{
    PF_Err err = PF_Err_NONE;
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

            case PF_Cmd_USER_CHANGED_PARAM:
                ERR(UserChangedParam(in_data, out_data, params, output, reinterpret_cast<const PF_UserChangedParamExtra*>(extra)));
            break;

                // Handling this selector will ensure that the UI will be properly initialized,
                // even before the user starts changing parameters to trigger PF_Cmd_USER_CHANGED_PARAM
            case PF_Cmd_UPDATE_PARAMS_UI:
                ERR(UpdateParameterUI(in_data, out_data, params, output));
            break;

            case PF_Cmd_SMART_PRE_RENDER:
                ERR(PreRender(in_data, out_data, reinterpret_cast<PF_PreRenderExtra*>(extra)));
            break;

            case PF_Cmd_SMART_RENDER:
                ERR(SmartRender(in_data, out_data, reinterpret_cast<PF_SmartRenderExtra*>(extra)));
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