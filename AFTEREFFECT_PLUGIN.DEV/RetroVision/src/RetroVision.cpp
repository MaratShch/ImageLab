#include "RetroVision.hpp"
#include "RetroVisionEnum.hpp"
#include "RetroVisionGui.hpp"
#include "RetroVisionResource.hpp"
#include "Param_Utils.h"
#include "PrSDKAESupport.h"
#include "ImageLabMemInterface.hpp"


static PF_Err
About
(
	PF_InData		*in_data,
	PF_OutData		*out_data,
	PF_ParamDef		*params[],
	PF_LayerDef		*output
)
{
	PF_SPRINTF(
		out_data->return_msg,
		"%s, v%d.%d\r%s",
		strName,
		RetroVision_VersionMajor,
		RetroVision_VersionMinor,
		strCopyright);

	return PF_Err_NONE;
}


static PF_Err
GlobalSetup
(
	PF_InData		*in_data,
	PF_OutData		*out_data,
	PF_ParamDef		*params[],
	PF_LayerDef		*output
)
{
    PF_Err	err = PF_Err_INTERNAL_STRUCT_DAMAGED;

    if (false == LoadMemoryInterfaceProvider(in_data))
        return err;

    constexpr PF_OutFlags out_flags1 =
        PF_OutFlag_WIDE_TIME_INPUT                |
        PF_OutFlag_SEQUENCE_DATA_NEEDS_FLATTENING |
        PF_OutFlag_USE_OUTPUT_EXTENT              |
        PF_OutFlag_PIX_INDEPENDENT                |
        PF_OutFlag_DEEP_COLOR_AWARE               |
        PF_OutFlag_CUSTOM_UI                      |
        PF_OutFlag_SEND_UPDATE_PARAMS_UI;

    constexpr PF_OutFlags out_flags2 =
        PF_OutFlag2_PARAM_GROUP_START_COLLAPSED_FLAG |
        PF_OutFlag2_DOESNT_NEED_EMPTY_PIXELS         |
        PF_OutFlag2_AUTOMATIC_WIDE_TIME_INPUT        |
        PF_OutFlag2_SUPPORTS_GET_FLATTENED_SEQUENCE_DATA;

	out_data->my_version =
		PF_VERSION(
			RetroVision_VersionMajor,
			RetroVision_VersionMinor,
			RetroVision_VersionSub,
			RetroVision_VersionStage,
			RetroVision_VersionBuild
		);

	out_data->out_flags  = out_flags1;
	out_data->out_flags2 = out_flags2;

	/* For Premiere - declare supported pixel formats */
	if (PremierId == in_data->appl_id)
	{
		auto const& pixelFormatSuite{ AEFX_SuiteScoper<PF_PixelFormatSuite1>(in_data, kPFPixelFormatSuite, kPFPixelFormatSuiteVersion1, out_data) };

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

    err = PF_Err_NONE;// (true == LoadBitmaps() ? PF_Err_NONE : PF_Err_INTERNAL_STRUCT_DAMAGED);

	return err;
}


static PF_Err
GlobalSetdown
(
	PF_InData		*in_data,
	PF_OutData		*out_data,
	PF_ParamDef		*params[],
	PF_LayerDef		*output
)
{
	/* nothing to do */
	return PF_Err_NONE;
}



static PF_Err
ParamsSetup
(
	PF_InData		*in_data,
	PF_OutData		*out_data,
	PF_ParamDef		*params[],
	PF_LayerDef		*output
)
{
    PF_ParamDef	def{};
    PF_Err		err = PF_Err_NONE;

    constexpr PF_ParamFlags   flags    = PF_ParamFlag_SUPERVISE;
    constexpr PF_ParamUIFlags ui_flags = PF_PUI_NONE;
    constexpr PF_ParamUIFlags ui_disabled_flags = ui_flags | PF_PUI_DISABLED;

    // SetUp 'Enable' checkbox. Default state - non selected
    AEFX_INIT_PARAM_STRUCTURE(def, flags, ui_flags);
    PF_ADD_CHECKBOXX(
        controlItemName[0],
        FALSE,
        flags,
        UnderlyingType(RetroVision::eRETRO_VISION_ENABLE));

    // add Display Type Logo (GUI)
    AEFX_CLR_STRUCT_EX(def);
    def.flags = flags;
    def.ui_flags = ui_disabled_flags;
    def.ui_width  = guiBarWidth;
    def.ui_height = guiBarHeight;
    if (PremierId != in_data->appl_id)
    {
        PF_ADD_COLOR(
            controlItemName[1],
            0,
            0,
            0,
            UnderlyingType(RetroVision::eRETRO_VISION_GUI));
    }
    else
    {
        PF_ADD_ARBITRARY2(
            controlItemName[1],
            guiBarWidth,
            guiBarHeight,
            0,
            PF_PUI_CONTROL,
            0,
            UnderlyingType(RetroVision::eRETRO_VISION_GUI),
            0);
    }
    if (PF_Err_NONE == err)
    {
        PF_CustomUIInfo	ui;
        AEFX_CLR_STRUCT_EX(ui);

        ui.events = PF_CustomEFlag_EFFECT;

        ui.comp_ui_width = 0;
        ui.comp_ui_height = 0;
        ui.comp_ui_alignment = PF_UIAlignment_NONE;

        ui.layer_ui_width = 0;
        ui.layer_ui_height = 0;
        ui.layer_ui_alignment = PF_UIAlignment_NONE;

        ui.preview_ui_width = 0;
        ui.preview_ui_height = 0;
        ui.layer_ui_alignment = PF_UIAlignment_NONE;

        err = (*(in_data->inter.register_ui))(in_data->effect_ref, &ui);
    } // if (PF_Err_NONE == err)

    // Setup 'Retro Monitor' popup - default value "CGA"
    AEFX_INIT_PARAM_STRUCTURE(def, flags, ui_disabled_flags);
    PF_ADD_POPUP(
        controlItemName[2],                                 // pop-up name
        UnderlyingType(RetroMonitor::eRETRO_BITMAP_TOTALS), // number of variants
        UnderlyingType(RetroMonitor::eRETRO_BITMAP_CGA),    // default variant
        retroMonitorName,                                   // string for pop-up
        UnderlyingType(RetroVision::eRETRO_VISION_DISPLAY));// control ID

    // Setup 'CGA Palette' popup - default value "CGA-1"
    AEFX_INIT_PARAM_STRUCTURE(def, flags, ui_disabled_flags);
    PF_ADD_POPUP(
        controlItemName[3],                                     // pop-up name
        UnderlyingType(PaletteCGA::eRETRO_PALETTE_CGA_TOTAL),   // number of variants
        UnderlyingType(PaletteCGA::eRETRO_PALETTE_CGA1),        // default variant
        cgaPaletteName,                                         // string for pop-up
        UnderlyingType(RetroVision::eRETRO_VISION_CGA_PALETTE));// control ID

    // Setup 'CGA Intencity Bit' checkbox. Default state - non selected
    AEFX_INIT_PARAM_STRUCTURE(def, flags, ui_disabled_flags);
    PF_ADD_CHECKBOXX(
        controlItemName[4],
        FALSE,
        flags,
        UnderlyingType(RetroVision::eRETRO_VISION_CGA_INTTENCITY_BIT));

    // Setup 'EGA Palette' popup - default value "Standard"
    AEFX_INIT_PARAM_STRUCTURE(def, flags, ui_disabled_flags);
    PF_ADD_POPUP(
        controlItemName[5],                                     // pop-up name
        UnderlyingType(PaletteEGA::eRETRO_PALETTE_EGA_TOTAL),   // number of variants
        UnderlyingType(PaletteEGA::eRETRO_PALETTE_EGA_STANDARD),// default variant
        egaPaletteName,                                         // string for pop-up
        UnderlyingType(RetroVision::eRETRO_VISION_EGA_PALETTE));// control ID

    // Setup 'VGA Palette' popup - default value "VGA 16 colors"
    AEFX_INIT_PARAM_STRUCTURE(def, flags, ui_disabled_flags);
    PF_ADD_POPUP(
        controlItemName[6],                                     // pop-up name
        UnderlyingType(PaletteVGA::eRETRO_PALETTE_VGA_TOTAL),   // number of variants
        UnderlyingType(PaletteVGA::eRETRO_PALETTE_VGA_16_BITS), // default variant
        vgaPaletteName,                                         // string for pop-up
        UnderlyingType(RetroVision::eRETRO_VISION_EGA_PALETTE));// control ID

    out_data->num_params = UnderlyingType(RetroVision::eRETRO_VISION_TOTAL_CONTROLS);

	return PF_Err_NONE;
}


static PF_Err
Render
(
	PF_InData		*in_data,
	PF_OutData		*out_data,
	PF_ParamDef		*params[],
	PF_LayerDef		*output
)
{
	return ((PremierId == in_data->appl_id ? ProcessImgInPR(in_data, out_data, params, output) : ProcessImgInAE(in_data, out_data, params, output)));
}


static PF_Err
PreRender
(
    PF_InData			*in_data,
    PF_OutData			*out_data,
    PF_PreRenderExtra	*extra
)
{
    return RetroVision_PreRender(in_data, out_data, extra);
}


static PF_Err
SmartRender
(
    PF_InData				*in_data,
    PF_OutData				*out_data,
    PF_SmartRenderExtra		*extraP
)
{
    return RetroVision_SmartRender(in_data, out_data, extraP);
}


static PF_Err
UserChangedParam
(
    PF_InData						*in_data,
    PF_OutData						*out_data,
    PF_ParamDef						*params[],
    PF_LayerDef						*outputP,
    const PF_UserChangedParamExtra	*which_hitP
)
{
    return RetroVision_UserChangedParam (in_data, out_data, params, outputP, which_hitP);
}


static PF_Err
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
        //    err = DrawEvent(in_data, out_data, params, output, extra);
        break;

        //		case PF_Event_ADJUST_CURSOR:
        //		break;

        default:
        break;
    }
    return err;
}


static PF_Err
UpdateParameterUI
(
    PF_InData			*in_data,
    PF_OutData			*out_data,
    PF_ParamDef			*params[],
    PF_LayerDef			*outputP
) noexcept
{
    PF_Err err = PF_Err_NONE;
    return err;
}



PLUGIN_ENTRY_POINT_CALL PF_Err
EffectMain
(
	PF_Cmd			cmd,
	PF_InData		*in_data,
	PF_OutData		*out_data,
	PF_ParamDef		*params[],
	PF_LayerDef		*output,
	void			*extra
)
{
	PF_Err err{ PF_Err_NONE };

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

            case PF_Cmd_UPDATE_PARAMS_UI:
                ERR(UpdateParameterUI(in_data, out_data, params, output));
            break;

            case PF_Cmd_EVENT:
                ERR(HandleEvent(in_data, out_data, params, output, reinterpret_cast<PF_EventExtra*>(extra)));
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