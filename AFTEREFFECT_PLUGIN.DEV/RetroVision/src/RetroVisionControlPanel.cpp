#include "RetroVision.hpp"
#include "RetroVisionEnum.hpp"
#include "RetroVisionGui.hpp"
#include "Param_Utils.h"

PF_Err
SetupControlElement
(
    const PF_InData*  in_data,
          PF_OutData* out_data
)
{
    CACHE_ALIGN PF_ParamDef	def{};
    PF_Err		err = PF_Err_NONE;

    constexpr PF_ParamFlags   flags = PF_ParamFlag_SUPERVISE;
    constexpr PF_ParamUIFlags ui_flags = PF_PUI_NONE;
    constexpr PF_ParamUIFlags ui_disabled_flags = ui_flags | PF_PUI_DISABLED;

    // Setup 'Gamma Adjust' slider
    AEFX_INIT_PARAM_STRUCTURE(def, flags, ui_flags);
    PF_ADD_FLOAT_SLIDERX(
        controlItemName[0],
        gammaSliderMin,
        gammaSliderMax,
        gammaSliderMin,
        gammaSliderMax,
        gammaSliderDef,
        PF_Precision_TENTHS,
        0,
        0,
        UnderlyingType(RetroVision::eRETRO_VISION_GAMMA_ADJUST));

    // SetUp 'Enable' checkbox. Default state - non selected
    AEFX_INIT_PARAM_STRUCTURE(def, flags, ui_flags);
    PF_ADD_CHECKBOXX(
        controlItemName[1],
        FALSE,
        flags,
        UnderlyingType(RetroVision::eRETRO_VISION_ENABLE));

    AEFX_CLR_STRUCT(def);
    PF_ADD_TOPICX(
        controlItemName[2],
        ui_disabled_flags,
        UnderlyingType(RetroVision::eRETRO_VISION_MONITOR_TYPE_START));

    // add Display Type Logo (GUI)
    AEFX_CLR_STRUCT_EX(def);
    def.flags = flags;
    def.ui_flags = ui_disabled_flags;
    def.ui_width = guiBarWidth;
    def.ui_height = guiBarHeight;
    if (PremierId != in_data->appl_id)
    {
        PF_ADD_COLOR(
            controlItemName[3],
            0,
            0,
            0,
            UnderlyingType(RetroVision::eRETRO_VISION_GUI));
    }
    else
    {
        PF_ADD_ARBITRARY2(
            controlItemName[3],
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
        controlItemName[4],                                 // pop-up name
        UnderlyingType(RetroMonitor::eRETRO_BITMAP_TOTALS), // number of variants
        UnderlyingType(RetroMonitor::eRETRO_BITMAP_CGA),    // default variant
        retroMonitorName,                                   // string for pop-up
        UnderlyingType(RetroVision::eRETRO_VISION_DISPLAY));// control ID

                                                            // Setup 'CGA Palette' popup - default value "CGA-1"
    AEFX_INIT_PARAM_STRUCTURE(def, flags, ui_disabled_flags);
    PF_ADD_POPUP(
        controlItemName[5],                                     // pop-up name
        UnderlyingType(PaletteCGA::eRETRO_PALETTE_CGA_TOTAL),   // number of variants
        UnderlyingType(PaletteCGA::eRETRO_PALETTE_CGA1),        // default variant
        cgaPaletteName,                                         // string for pop-up
        UnderlyingType(RetroVision::eRETRO_VISION_CGA_PALETTE));// control ID

                                                                // Setup 'CGA Intencity Bit' checkbox. Default state - non selected
    AEFX_INIT_PARAM_STRUCTURE(def, flags, ui_disabled_flags);
    PF_ADD_CHECKBOXX(
        controlItemName[6],
        FALSE,
        flags,
        UnderlyingType(RetroVision::eRETRO_VISION_CGA_INTTENCITY_BIT));

    // Setup 'EGA Palette' popup - default value "Standard"
    AEFX_INIT_PARAM_STRUCTURE(def, flags, ui_disabled_flags);
    PF_ADD_POPUP(
        controlItemName[7],                                     // pop-up name
        UnderlyingType(PaletteEGA::eRETRO_PALETTE_EGA_TOTAL),   // number of variants
        UnderlyingType(PaletteEGA::eRETRO_PALETTE_EGA_STANDARD),// default variant
        egaPaletteName,                                         // string for pop-up
        UnderlyingType(RetroVision::eRETRO_VISION_EGA_PALETTE));// control ID

                                                                // Setup 'VGA Palette' popup - default value "VGA 16 colors"
    AEFX_INIT_PARAM_STRUCTURE(def, flags, ui_disabled_flags);
    PF_ADD_POPUP(
        controlItemName[8],                                     // pop-up name
        UnderlyingType(PaletteVGA::eRETRO_PALETTE_VGA_TOTAL),   // number of variants
        UnderlyingType(PaletteVGA::eRETRO_PALETTE_VGA_16_BITS), // default variant
        vgaPaletteName,                                         // string for pop-up
        UnderlyingType(RetroVision::eRETRO_VISION_EGA_PALETTE));// control ID

    AEFX_CLR_STRUCT_EX(def);
    PF_END_TOPIC(UnderlyingType(RetroVision::eRETRO_VISION_MONITOR_TYPE_STOP));

    AEFX_CLR_STRUCT_EX(def);
    PF_ADD_TOPICX(
        controlItemName[9],
        ui_disabled_flags,
        UnderlyingType(RetroVision::eRETRO_VISION_CRT_ARTIFACTS_START));

    AEFX_INIT_PARAM_STRUCTURE(def, flags, ui_disabled_flags);
    PF_ADD_CHECKBOXX(
        controlItemName[10],
        FALSE,
        flags,
        UnderlyingType(RetroVision::eRETRO_VISION_CRT_ARTIFACTS_SCANLINES));

    AEFX_INIT_PARAM_STRUCTURE(def, flags, ui_disabled_flags);
    PF_ADD_CHECKBOXX(
        controlItemName[11],
        FALSE,
        flags,
        UnderlyingType(RetroVision::eRETRO_VISION_CRT_ARTIFACTS_SMOOTH_SCANLINES));

    AEFX_INIT_PARAM_STRUCTURE(def, flags, ui_disabled_flags);
    PF_ADD_SLIDER(
        controlItemName[12],
        scanLinesSliderMin,
        scanLinesSliderMax,
        scanLinesSliderMin,
        scanLinesSliderMax,
        scanLinesSliderDef,
        UnderlyingType(RetroVision::eRETRO_VISION_CRT_ARTIFACTS_SCANLINES_INTERVAL));

    AEFX_INIT_PARAM_STRUCTURE(def, flags, ui_disabled_flags);
    PF_ADD_FLOAT_SLIDERX(
        controlItemName[13],
        scanLinesDarknessMin,
        scanLinesDarknessMax,
        scanLinesDarknessMin,
        scanLinesDarknessMax,
        scanLinesDarknessDef,
        PF_Precision_HUNDREDTHS,
        0,
        0,
        UnderlyingType(RetroVision::eRETRO_VISION_CRT_ARTIFACTS_SCANLINES_DARKNESS));

    AEFX_INIT_PARAM_STRUCTURE(def, flags, ui_disabled_flags);
    PF_ADD_CHECKBOXX(
        controlItemName[14],
        FALSE,
        flags,
        UnderlyingType(RetroVision::eRETRO_VISION_CRT_ARTIFACTS_PHOSPHOR_GLOW));

    AEFX_INIT_PARAM_STRUCTURE(def, flags, ui_disabled_flags);
    PF_ADD_FLOAT_SLIDERX(
        controlItemName[15],
        phosphorGlowStrengthMin,
        phosphorGlowStrengthMax,
        phosphorGlowStrengthMin,
        phosphorGlowStrengthMax,
        phosphorGlowStrengthDef,
        PF_Precision_HUNDREDTHS,
        0,
        0,
        UnderlyingType(RetroVision::eRETRO_VISION_CRT_ARTIFACTS_PHOSPHOR_GLOW_STRENGHT));

    AEFX_INIT_PARAM_STRUCTURE(def, flags, ui_disabled_flags);
    PF_ADD_FLOAT_SLIDERX(
        controlItemName[16],
        phosphorGlowOpacityMin,
        phosphorGlowOpacityMax,
        phosphorGlowOpacityMin,
        phosphorGlowOpacityMax,
        phosphorGlowOpacityDef,
        PF_Precision_HUNDREDTHS,
        0,
        0,
        UnderlyingType(RetroVision::eRETRO_VISION_CRT_ARTIFACTS_PHOSPHOR_GLOW_OPACITY));

    AEFX_INIT_PARAM_STRUCTURE(def, flags, ui_disabled_flags);
    PF_ADD_CHECKBOXX(
        controlItemName[17],
        FALSE,
        flags,
        UnderlyingType(RetroVision::eRETRO_VISION_CRT_ARTIFACTS_APPERTURE_GRILL));

    AEFX_INIT_PARAM_STRUCTURE(def, flags, ui_disabled_flags);
    PF_ADD_POPUP(
        controlItemName[18],
        UnderlyingType(AppertureGtrill::eRETRO_APPERTURE_TOTALS),
        UnderlyingType(AppertureGtrill::eRETRO_APPERTURE_NONE),
        appertureMaskType,
        UnderlyingType(RetroVision::eRETRO_VISION_CRT_ARTIFACTS_APPERTURE_GRILL_POPUP));

    AEFX_INIT_PARAM_STRUCTURE(def, flags, ui_disabled_flags);
    PF_ADD_SLIDER(
        controlItemName[19],
        appertureMaskMin,
        appertureMaskMax,
        appertureMaskMin,
        appertureMaskMax,
        appertureMaskDef,
        UnderlyingType(RetroVision::eRETRO_VISION_CRT_ARTIFACTS_APPERTURE_GRILL_INTERVAL));

    AEFX_INIT_PARAM_STRUCTURE(def, flags, ui_disabled_flags);
    PF_ADD_FLOAT_SLIDERX(
        controlItemName[20],
        appertureMaskDarkMin,
        appertureMaskDarkMax,
        appertureMaskDarkMin,
        appertureMaskDarkMax,
        appertureMaskDarkDef,
        PF_Precision_HUNDREDTHS,
        0,
        0,
        UnderlyingType(RetroVision::eRETRO_VISION_CRT_ARTIFACTS_APPERTURE_GRILL_DARKNESS));

    AEFX_INIT_PARAM_STRUCTURE(def, flags, ui_disabled_flags);
    PF_ADD_SLIDER(
        controlItemName[21],
        appertureMaskColorMin,
        appertureMaskColorMax,
        appertureMaskColorMin,
        appertureMaskColorMax,
        appertureMaskColorDef,
        UnderlyingType(RetroVision::eRETRO_VISION_CRT_ARTIFACTS_APPERTURE_GRILL_COLOR));

    AEFX_CLR_STRUCT_EX(def);
    PF_END_TOPIC(UnderlyingType(RetroVision::eRETRO_VISION_CRT_ARTIFACTS_STOP));

    out_data->num_params = UnderlyingType(RetroVision::eRETRO_VISION_TOTAL_CONTROLS);

    return err;
}