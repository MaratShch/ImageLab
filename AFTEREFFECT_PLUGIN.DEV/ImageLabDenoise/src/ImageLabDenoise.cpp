#include "ImageLabDenoise.hpp"
#include "ImageLabMemInterface.hpp"
#include "PrSDKAESupport.h"


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
		ImageLabDenoise_VersionMajor,
		ImageLabDenoise_VersionMinor,
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
    PF_Err	err = PF_Err_INTERNAL_STRUCT_DAMAGED;

    if (false == LoadMemoryInterfaceProvider(in_data))
        return err;

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
		PF_VERSION(
			ImageLabDenoise_VersionMajor,
			ImageLabDenoise_VersionMinor,
			ImageLabDenoise_VersionSub,
			ImageLabDenoise_VersionStage,
			ImageLabDenoise_VersionBuild
		);

	out_data->out_flags  = out_flags1;
	out_data->out_flags2 = out_flags2;

	/* For Premiere - declare supported pixel formats */
	if (PremierId == in_data->appl_id)
	{
		auto const pixelFormatSuite{ AEFX_SuiteScoper<PF_PixelFormatSuite1>(in_data, kPFPixelFormatSuite, kPFPixelFormatSuiteVersion1, out_data) };

		/*	Add the pixel formats we support in order of preference. */
		(*pixelFormatSuite->ClearSupportedPixelFormats)(in_data->effect_ref);

//		(*pixelFormatSuite->AddSupportedPixelFormat)(in_data->effect_ref, PrPixelFormat_BGRA_4444_8u);
//		(*pixelFormatSuite->AddSupportedPixelFormat)(in_data->effect_ref, PrPixelFormat_BGRA_4444_16u);
		(*pixelFormatSuite->AddSupportedPixelFormat)(in_data->effect_ref, PrPixelFormat_BGRA_4444_32f);
//		(*pixelFormatSuite->AddSupportedPixelFormat)(in_data->effect_ref, PrPixelFormat_VUYA_4444_8u_709);
//		(*pixelFormatSuite->AddSupportedPixelFormat)(in_data->effect_ref, PrPixelFormat_VUYA_4444_8u);
//		(*pixelFormatSuite->AddSupportedPixelFormat)(in_data->effect_ref, PrPixelFormat_VUYA_4444_32f_709);
//		(*pixelFormatSuite->AddSupportedPixelFormat)(in_data->effect_ref, PrPixelFormat_VUYA_4444_32f);
//		(*pixelFormatSuite->AddSupportedPixelFormat)(in_data->effect_ref, PrPixelFormat_RGB_444_10u);
	}

    err = PF_Err_NONE;
	return err;
}


static PF_Err
GlobalSetdown(
	PF_InData		*in_data,
	PF_OutData		*out_data,
	PF_ParamDef		*params[],
	PF_LayerDef		*output)
{
    // Unload memory management library
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
    return SetupControlElements (in_data, out_data);
}


static PF_Err
SequenceSetup (
    PF_InData		*in_data,
    PF_OutData		*out_data,
    PF_ParamDef		*params[],
    PF_LayerDef		*output)
{
    return ImageLabDenoise_SequenceSetup (in_data, out_data, params, output);
}


static PF_Err
SequenceResetup (
    PF_InData		*in_data,
    PF_OutData		*out_data,
    PF_ParamDef		*params[],
    PF_LayerDef		*output)
{
    return ImageLabDenoise_SequenceReSetup (in_data, out_data, params, output);
}

static PF_Err
SequenceFlatten (
    PF_InData		*in_data,
    PF_OutData		*out_data,
    PF_ParamDef		*params[],
    PF_LayerDef		*output)
{
    return ImageLabDenoise_SequenceFlatten (in_data, out_data, params, output);
}


static PF_Err
SequenceSetdown (
    PF_InData		*in_data,
    PF_OutData		*out_data,
    PF_ParamDef		*params[],
    PF_LayerDef		*output)
{
    return ImageLabDenoise_SequenceSetdown (in_data, out_data, params, output);
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
UserChangedParam(
    PF_InData						*in_data,
    PF_OutData						*out_data,
    PF_ParamDef						*params[],
    PF_LayerDef						*outputP,
    const PF_UserChangedParamExtra	*which_hitP
)
{
    PF_Err errControl = PF_Err_NONE;
    return errControl;
}


static PF_Err
UpdateParameterUI(
    PF_InData			*in_data,
    PF_OutData			*out_data,
    PF_ParamDef			*params[],
    PF_LayerDef			*outputP
) noexcept
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
    return ImageLabDenoise_PreRender (in_data, out_data, extra);
}


PF_Err
SmartRender
(
    PF_InData				*in_data,
    PF_OutData				*out_data,
    PF_SmartRenderExtra		*extra
)
{
    return ImageLabDenoise_SmartRender(in_data, out_data, extra);
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

            case PF_Cmd_SEQUENCE_SETUP:
                ERR(SequenceSetup(in_data, out_data, params, output));
            break;

            case PF_Cmd_SEQUENCE_RESETUP:
                ERR(SequenceResetup(in_data, out_data, params, output));
            break;

            case PF_Cmd_SEQUENCE_FLATTEN:
                ERR(SequenceFlatten(in_data, out_data, params, output));
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