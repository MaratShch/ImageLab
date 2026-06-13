#include "AutomaticWhiteBalance2.hpp"
#include "AlgorithmEnums.hpp"
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
		AWB2_VersionMajor,
		AWB2_VersionMinor,
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
        PF_OutFlag_PIX_INDEPENDENT       |
        PF_OutFlag_SEND_UPDATE_PARAMS_UI |
        PF_OutFlag_USE_OUTPUT_EXTENT     |
        PF_OutFlag_DEEP_COLOR_AWARE      |
        PF_OutFlag_WIDE_TIME_INPUT;

    constexpr PF_OutFlags out_flags2 =
        PF_OutFlag2_PARAM_GROUP_START_COLLAPSED_FLAG |
        PF_OutFlag2_DOESNT_NEED_EMPTY_PIXELS         |
        PF_OutFlag2_AUTOMATIC_WIDE_TIME_INPUT        |
        PF_OutFlag2_SUPPORTS_SMART_RENDER;

	out_data->my_version =
		PF_VERSION(
			AWB2_VersionMajor,
			AWB2_VersionMinor,
			AWB2_VersionSub,
			AWB2_VersionStage,
			AWB2_VersionBuild
		);

	out_data->out_flags  = out_flags1;
	out_data->out_flags2 = out_flags2;

	/* For Premiere - declare supported pixel formats */
	if (PremierId == in_data->appl_id)
	{
		auto const pixelFormatSuite{ AEFX_SuiteScoper<PF_PixelFormatSuite1>(in_data, kPFPixelFormatSuite, kPFPixelFormatSuiteVersion1, out_data) };

        /*	Add the pixel formats we support in order of preference. */
        (*pixelFormatSuite->ClearSupportedPixelFormats)(in_data->effect_ref);

        (*pixelFormatSuite->AddSupportedPixelFormat)(in_data->effect_ref, PrPixelFormat_BGRA_4444_8u);
//        (*pixelFormatSuite->AddSupportedPixelFormat)(in_data->effect_ref, PrPixelFormat_BGRA_4444_16u);
//        (*pixelFormatSuite->AddSupportedPixelFormat)(in_data->effect_ref, PrPixelFormat_BGRA_4444_32f);
//        (*pixelFormatSuite->AddSupportedPixelFormat)(in_data->effect_ref, PrPixelFormat_BGRA_4444_32f_Linear);
//        (*pixelFormatSuite->AddSupportedPixelFormat)(in_data->effect_ref, PrPixelFormat_BGRP_4444_8u);
//        (*pixelFormatSuite->AddSupportedPixelFormat)(in_data->effect_ref, PrPixelFormat_BGRP_4444_16u);
//        (*pixelFormatSuite->AddSupportedPixelFormat)(in_data->effect_ref, PrPixelFormat_BGRP_4444_32f);
//        (*pixelFormatSuite->AddSupportedPixelFormat)(in_data->effect_ref, PrPixelFormat_BGRP_4444_32f_Linear);
//        (*pixelFormatSuite->AddSupportedPixelFormat)(in_data->effect_ref, PrPixelFormat_BGRX_4444_8u);
//        (*pixelFormatSuite->AddSupportedPixelFormat)(in_data->effect_ref, PrPixelFormat_BGRX_4444_16u);
//        (*pixelFormatSuite->AddSupportedPixelFormat)(in_data->effect_ref, PrPixelFormat_BGRX_4444_32f);
//        (*pixelFormatSuite->AddSupportedPixelFormat)(in_data->effect_ref, PrPixelFormat_BGRX_4444_32f_Linear);
//        (*pixelFormatSuite->AddSupportedPixelFormat)(in_data->effect_ref, PrPixelFormat_VUYA_4444_8u_709);
//        (*pixelFormatSuite->AddSupportedPixelFormat)(in_data->effect_ref, PrPixelFormat_VUYA_4444_8u);
//        (*pixelFormatSuite->AddSupportedPixelFormat)(in_data->effect_ref, PrPixelFormat_VUYA_4444_32f_709);
//        (*pixelFormatSuite->AddSupportedPixelFormat)(in_data->effect_ref, PrPixelFormat_VUYA_4444_32f);
//        (*pixelFormatSuite->AddSupportedPixelFormat)(in_data->effect_ref, PrPixelFormat_VUYP_4444_8u_709);
//        (*pixelFormatSuite->AddSupportedPixelFormat)(in_data->effect_ref, PrPixelFormat_VUYP_4444_8u);
//        (*pixelFormatSuite->AddSupportedPixelFormat)(in_data->effect_ref, PrPixelFormat_VUYP_4444_32f_709);
//        (*pixelFormatSuite->AddSupportedPixelFormat)(in_data->effect_ref, PrPixelFormat_VUYP_4444_32f);
//        (*pixelFormatSuite->AddSupportedPixelFormat)(in_data->effect_ref, PrPixelFormat_VUYX_4444_8u_709);
//        (*pixelFormatSuite->AddSupportedPixelFormat)(in_data->effect_ref, PrPixelFormat_VUYX_4444_8u);
//        (*pixelFormatSuite->AddSupportedPixelFormat)(in_data->effect_ref, PrPixelFormat_VUYX_4444_32f_709);
//        (*pixelFormatSuite->AddSupportedPixelFormat)(in_data->effect_ref, PrPixelFormat_VUYX_4444_32f);
//        (*pixelFormatSuite->AddSupportedPixelFormat)(in_data->effect_ref, PrPixelFormat_RGB_444_10u);
//        (*pixelFormatSuite->AddSupportedPixelFormat)(in_data->effect_ref, PrPixelFormat_ARGB_4444_8u);
//        (*pixelFormatSuite->AddSupportedPixelFormat)(in_data->effect_ref, PrPixelFormat_ARGB_4444_16u);
//        (*pixelFormatSuite->AddSupportedPixelFormat)(in_data->effect_ref, PrPixelFormat_ARGB_4444_32f);
//        (*pixelFormatSuite->AddSupportedPixelFormat)(in_data->effect_ref, PrPixelFormat_PRGB_4444_32f);
//        (*pixelFormatSuite->AddSupportedPixelFormat)(in_data->effect_ref, PrPixelFormat_XRGB_4444_32f);
//        (*pixelFormatSuite->AddSupportedPixelFormat)(in_data->effect_ref, PrPixelFormat_ARGB_4444_32f_Linear);
//        (*pixelFormatSuite->AddSupportedPixelFormat)(in_data->effect_ref, PrPixelFormat_PRGB_4444_32f_Linear);
//        (*pixelFormatSuite->AddSupportedPixelFormat)(in_data->effect_ref, PrPixelFormat_XRGB_4444_32f_Linear);
    }

	return PF_Err_NONE;
}


static PF_Err
GlobalSetdown(
	PF_InData		*in_data,
	PF_OutData		*out_data,
	PF_ParamDef		*params[],
	PF_LayerDef		*output)
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
    CACHE_ALIGN PF_ParamDef	def{};
    PF_Err		err = PF_Err_NONE;

    constexpr PF_ParamFlags popup_flags = PF_ParamFlag_SUPERVISE | PF_ParamFlag_CANNOT_TIME_VARY | PF_ParamFlag_CANNOT_INTERP;
    constexpr PF_ParamUIFlags popup_ui_flags = PF_PUI_NONE;

    AEFX_INIT_PARAM_STRUCTURE(def, popup_flags, popup_ui_flags);
    PF_ADD_POPUP(
        strCtrlNames[0],		                                            // pop-up name
        gTotalNumbersOfColorSpaces,	                                        // number of Color Spaces
        gDefNumberOfColorSpace,		                                        // default color space
        strColorSpace,    		                                            // string for pop-up
        UnderlyingType(eImageLab2AWB_Controls::AWB2_COLOR_SPACE_POPUP));	// control ID

    AEFX_INIT_PARAM_STRUCTURE(def, popup_flags, popup_ui_flags);
    PF_ADD_POPUP(
        strCtrlNames[1],		                                            // pop-up name
        UnderlyingType(eILLUMINATE::TOTAL_ILLUMINANTES),		            // number of Illuminates
        UnderlyingType(eILLUMINATE::DAYLIGHT),                              // default Illumnat
        strIlluminantName,			                                        // string for pop-up
        UnderlyingType(eImageLab2AWB_Controls::AWB2_ILLUMINATE_POPUP));	    // control ID

    AEFX_INIT_PARAM_STRUCTURE(def, popup_flags, popup_ui_flags);
    PF_ADD_POPUP(
        strCtrlNames[2],                                                    // pop-up name
        UnderlyingType(eChromaticAdaptation::TOTAL_CHROMATIC),		        // number of Illuminates
        UnderlyingType(eChromaticAdaptation::CHROMATIC_CAT02),              // default Illumnat
        strChtomaticAdaptation,			                                    // string for pop-up
        UnderlyingType(eImageLab2AWB_Controls::AWB2_CHROMATIC_POPUP));	    // control ID

    AEFX_INIT_PARAM_STRUCTURE(def, popup_flags, popup_ui_flags);
    PF_ADD_FLOAT_SLIDERX(
        strCtrlNames[3],                                                    // pop-up name
        extremePixMin,
        extremePixMax,
        extremePixMin,
        extremePixMax,
        extremePixDef,
        PF_Precision_TENTHS,
        0,
        0,
        UnderlyingType(eImageLab2AWB_Controls::AWB2_EXTERME_PIXELS));

    AEFX_INIT_PARAM_STRUCTURE(def, popup_flags, popup_ui_flags);
    PF_ADD_FLOAT_SLIDERX(
        strCtrlNames[4],                                                    // pop-up name
        saturationThrMin,
        saturationThrMax,
        saturationThrMin,
        saturationThrMax,
        saturationThrDef,
        PF_Precision_HUNDREDTHS,
        0,
        0,
        UnderlyingType(eImageLab2AWB_Controls::AWB2_SATRURATION_THRESHOLD));

    AEFX_INIT_PARAM_STRUCTURE(def, popup_flags, popup_ui_flags);
    PF_ADD_FLOAT_SLIDERX(
        strCtrlNames[5],                                                    // pop-up name
        blackLevelThresholdMin,
        blackLevelThresholdMax,
        blackLevelThresholdMin,
        blackLevelThresholdMax,
        blackLevelThresholdDef,
        PF_Precision_HUNDREDTHS,
        0,
        0,
        UnderlyingType(eImageLab2AWB_Controls::AWB2_BLACK_LEVEL_THRESHOLD));

    out_data->num_params = UnderlyingType(eImageLab2AWB_Controls::AWB2_TOTAL_CONTROLS);

	return err;
}


static PF_Err
Render(
	PF_InData		*in_data,
	PF_OutData		*out_data,
	PF_ParamDef		*params[],
	PF_LayerDef		*output)
{
	return ((PremierId == in_data->appl_id ? ProcessImgInPR(in_data, out_data, params, output) : ProcessImgInAE(in_data, out_data, params, output)));
}



static PF_Err
PreRender(
    PF_InData			*in_data,
    PF_OutData			*out_data,
    PF_PreRenderExtra	*extra
)
{
    return PF_Err_NONE;
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