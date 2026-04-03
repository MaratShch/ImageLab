#include "ArtPaint.hpp"
#include "ArtPaintEnums.hpp"
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
		ArtPaint_VersionMajor,
		ArtPaint_VersionMinor,
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
        PF_OutFlag2_AUTOMATIC_WIDE_TIME_INPUT        |
        PF_OutFlag2_FLOAT_COLOR_AWARE                |
        PF_OutFlag2_SUPPORTS_SMART_RENDER;

	out_data->my_version =
		PF_VERSION(
			ArtPaint_VersionMajor,
			ArtPaint_VersionMinor,
			ArtPaint_VersionSub,
			ArtPaint_VersionStage,
			ArtPaint_VersionBuild
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
#if 0
		(*pixelFormatSuite->AddSupportedPixelFormat)(in_data->effect_ref, PrPixelFormat_BGRA_4444_16u);
		(*pixelFormatSuite->AddSupportedPixelFormat)(in_data->effect_ref, PrPixelFormat_BGRA_4444_32f);
        (*pixelFormatSuite->AddSupportedPixelFormat)(in_data->effect_ref, PrPixelFormat_BGRA_4444_32f_Linear);
        (*pixelFormatSuite->AddSupportedPixelFormat)(in_data->effect_ref, PrPixelFormat_BGRP_4444_8u);
        (*pixelFormatSuite->AddSupportedPixelFormat)(in_data->effect_ref, PrPixelFormat_BGRP_4444_16u);
        (*pixelFormatSuite->AddSupportedPixelFormat)(in_data->effect_ref, PrPixelFormat_BGRP_4444_32f);
        (*pixelFormatSuite->AddSupportedPixelFormat)(in_data->effect_ref, PrPixelFormat_BGRP_4444_32f_Linear);
        (*pixelFormatSuite->AddSupportedPixelFormat)(in_data->effect_ref, PrPixelFormat_BGRX_4444_8u);
        (*pixelFormatSuite->AddSupportedPixelFormat)(in_data->effect_ref, PrPixelFormat_BGRX_4444_16u);
        (*pixelFormatSuite->AddSupportedPixelFormat)(in_data->effect_ref, PrPixelFormat_BGRX_4444_32f);
        (*pixelFormatSuite->AddSupportedPixelFormat)(in_data->effect_ref, PrPixelFormat_BGRX_4444_32f_Linear);

        (*pixelFormatSuite->AddSupportedPixelFormat)(in_data->effect_ref, PrPixelFormat_VUYA_4444_8u_709);
		(*pixelFormatSuite->AddSupportedPixelFormat)(in_data->effect_ref, PrPixelFormat_VUYA_4444_8u);
		(*pixelFormatSuite->AddSupportedPixelFormat)(in_data->effect_ref, PrPixelFormat_VUYA_4444_32f_709);
		(*pixelFormatSuite->AddSupportedPixelFormat)(in_data->effect_ref, PrPixelFormat_VUYA_4444_32f);
        (*pixelFormatSuite->AddSupportedPixelFormat)(in_data->effect_ref, PrPixelFormat_VUYX_4444_8u_709);
        (*pixelFormatSuite->AddSupportedPixelFormat)(in_data->effect_ref, PrPixelFormat_VUYX_4444_8u);
        (*pixelFormatSuite->AddSupportedPixelFormat)(in_data->effect_ref, PrPixelFormat_VUYX_4444_32f_709);
        (*pixelFormatSuite->AddSupportedPixelFormat)(in_data->effect_ref, PrPixelFormat_VUYX_4444_32f);

        (*pixelFormatSuite->AddSupportedPixelFormat)(in_data->effect_ref, PrPixelFormat_RGB_444_10u);

        (*pixelFormatSuite->AddSupportedPixelFormat)(in_data->effect_ref, PrPixelFormat_ARGB_4444_8u);
        (*pixelFormatSuite->AddSupportedPixelFormat)(in_data->effect_ref, PrPixelFormat_ARGB_4444_16u);
        (*pixelFormatSuite->AddSupportedPixelFormat)(in_data->effect_ref, PrPixelFormat_ARGB_4444_32f);
        (*pixelFormatSuite->AddSupportedPixelFormat)(in_data->effect_ref, PrPixelFormat_ARGB_4444_32f_Linear);
        (*pixelFormatSuite->AddSupportedPixelFormat)(in_data->effect_ref, PrPixelFormat_PRGB_4444_8u);
        (*pixelFormatSuite->AddSupportedPixelFormat)(in_data->effect_ref, PrPixelFormat_PRGB_4444_16u);
        (*pixelFormatSuite->AddSupportedPixelFormat)(in_data->effect_ref, PrPixelFormat_PRGB_4444_32f);
        (*pixelFormatSuite->AddSupportedPixelFormat)(in_data->effect_ref, PrPixelFormat_PRGB_4444_32f_Linear);
        (*pixelFormatSuite->AddSupportedPixelFormat)(in_data->effect_ref, PrPixelFormat_XRGB_4444_8u);
        (*pixelFormatSuite->AddSupportedPixelFormat)(in_data->effect_ref, PrPixelFormat_XRGB_4444_16u);
        (*pixelFormatSuite->AddSupportedPixelFormat)(in_data->effect_ref, PrPixelFormat_XRGB_4444_32f);
        (*pixelFormatSuite->AddSupportedPixelFormat)(in_data->effect_ref, PrPixelFormat_XRGB_4444_32f_Linear);
#endif
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

    constexpr PF_ParamFlags     flags = PF_ParamFlag_SUPERVISE | PF_ParamFlag_CANNOT_TIME_VARY | PF_ParamFlag_CANNOT_INTERP;
    constexpr PF_ParamUIFlags   ui_flags = PF_PUI_NONE;
 
    AEFX_INIT_PARAM_STRUCTURE(def, flags, ui_flags);
    PF_ADD_POPUP(
        ArtPaintControlsStr[0],
        UnderlyingType(StrokeBias::TotalStrokeBias),
        UnderlyingType(StrokeBias::DarkBias_Open),
        StrokeBiasStr,
        UnderlyingType(ArtPaintControls::ART_PAINT_STYLE));

    AEFX_INIT_PARAM_STRUCTURE(def, flags, ui_flags);
    PF_ADD_FLOAT_SLIDERX(
        ArtPaintControlsStr[1],
        sigmaMin,
        sigmaMax,
        sigmaMin,
        sigmaMax,
        sigmaDef,
        PF_Precision_TENTHS,
        0,
        0,
        UnderlyingType(ArtPaintControls::ART_PAINT_BRUSH_WIDTH));

    AEFX_INIT_PARAM_STRUCTURE(def, flags, ui_flags);
    PF_ADD_FLOAT_SLIDERX(
        ArtPaintControlsStr[2],
        angularMin,
        angularMax,
        angularMin,
        angularMax,
        angularDef,
        PF_Precision_INTEGER,
        0,
        0,
        UnderlyingType(ArtPaintControls::ART_PAINT_BRUSH_LENGTH));

    AEFX_INIT_PARAM_STRUCTURE(def, flags, ui_flags);
    PF_ADD_FLOAT_SLIDERX(
        ArtPaintControlsStr[3],
        angleMin,
        angleMax,
        angleMin,
        angleMax,
        angleDef,
        PF_Precision_INTEGER,
        0,
        0,
        UnderlyingType(ArtPaintControls::ART_PAINT_STROKE_CURVATIVE));

    AEFX_INIT_PARAM_STRUCTURE(def, flags, ui_flags);
    PF_ADD_SLIDER(
        ArtPaintControlsStr[4],
        iterMin,
        iterMax,
        iterMin,
        iterMax,
        iterDef,
        UnderlyingType(ArtPaintControls::ART_PAINT_STROKE_SPREADING));

    out_data->num_params = UnderlyingType(ArtPaintControls::ART_PAINT_TOTAL_PARAMS);

    return PF_Err_NONE;
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
    PF_InData				*in_data,
    PF_OutData				*out_data,
    PF_PreRenderExtra		*extraP
)
{
    return ArtPaint_PreRender(in_data, out_data, extraP);
}


static PF_Err
SmartRender(
    PF_InData				*in_data,
    PF_OutData				*out_data,
    PF_SmartRenderExtra		*extraP
)
{
    return ArtPaint_SmartRender(in_data, out_data, extraP);
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
    return PF_Err_NONE;
}


static PF_Err
UpdateParameterUI(
    PF_InData		*in_data,
    PF_OutData		*out_data,
    PF_ParamDef		*params[],
    PF_LayerDef		*output)
{
    return PF_Err_NONE;
}


inline PF_Err
HandleEvent
(
    PF_InData		*in_data,
    PF_OutData		*out_data,
    PF_ParamDef		*params[],
    PF_LayerDef		*output,
    PF_EventExtra	*extra)
{
    PF_Err		err = PF_Err_NONE;

    switch (extra->e_type)
    {
    case PF_Event_DRAW:
        //        err = DrawEvent(in_data, out_data, params, output, extra);
        break;

    default:
        break;
    }
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

//          case PF_Cmd_EVENT:
//              ERR(HandleEvent(in_data, out_data, params, output, reinterpret_cast<PF_EventExtra*>(extra)));
//          break;

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