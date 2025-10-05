#ifndef __IMAGE_LAB_RETRO_VISION_FILTER_CONTROLS__
#define __IMAGE_LAB_RETRO_VISION_FILTER_CONTROLS__

#include <cstdint>
#include "RetroVisionEnum.hpp"

struct RVControls
{
    RetroMonitor monitor;
    PaletteCGA   cga_palette;
    int32_t      cga_intencity_bit;
    PaletteEGA   ega_palette;
    PaletteVGA   vga_palette;
    int32_t hercules_threshold;
    int32_t scan_lines_enable;
    int32_t scan_lines_smooth;
    int32_t scan_lines_interval;
    float   scan_lines_darkness;
    int32_t phosphor_glow_enable;
    float   phosphor_glow_strength;
    float   phosphor_glow_opacity;
    int32_t apperture_grill_enable;
    AppertureGtrill mask_type;
    int32_t mask_interval;
    float   mask_darkness;
    int32_t mask_color;
    HerculesWhiteColor white_color_hercules;
};

constexpr size_t RVControlsSize = sizeof(RVControls);


inline RVControls GetControlParametersStruct
(
    PF_ParamDef* __restrict params[]
) noexcept
{
    RVControls rvControls{};

    rvControls.monitor           = static_cast<RetroMonitor>(params[UnderlyingType(RetroVision::eRETRO_VISION_DISPLAY)]->u.pd.value - 1);
    rvControls.cga_palette       = static_cast<PaletteCGA>(params[UnderlyingType(RetroVision::eRETRO_VISION_CGA_PALETTE)]->u.pd.value - 1);
    rvControls.cga_intencity_bit = params[UnderlyingType(RetroVision::eRETRO_VISION_CGA_INTTENCITY_BIT)]->u.bd.value;
    rvControls.ega_palette       = static_cast<PaletteEGA>(params[UnderlyingType(RetroVision::eRETRO_VISION_EGA_PALETTE)]->u.pd.value - 1);
    rvControls.vga_palette       = static_cast<PaletteVGA>(params[UnderlyingType(RetroVision::eRETRO_VISION_VGA_PALETTE)]->u.pd.value - 1);
    rvControls.hercules_threshold = params[UnderlyingType(RetroVision::eRETRO_VISION_HERCULES_THRESHOLD)]->u.sd.value;

    rvControls.scan_lines_enable   = params[UnderlyingType(RetroVision::eRETRO_VISION_CRT_ARTIFACTS_SCANLINES)]->u.bd.value;
    rvControls.scan_lines_smooth   = params[UnderlyingType(RetroVision::eRETRO_VISION_CRT_ARTIFACTS_SMOOTH_SCANLINES)]->u.bd.value;
    rvControls.scan_lines_interval = params[UnderlyingType(RetroVision::eRETRO_VISION_CRT_ARTIFACTS_SCANLINES_INTERVAL)]->u.sd.value;
    rvControls.scan_lines_darkness = static_cast<float>(params[UnderlyingType(RetroVision::eRETRO_VISION_CRT_ARTIFACTS_SCANLINES_DARKNESS)]->u.fs_d.value);

    rvControls.phosphor_glow_enable   = params[UnderlyingType(RetroVision::eRETRO_VISION_CRT_ARTIFACTS_PHOSPHOR_GLOW)]->u.bd.value;
    rvControls.phosphor_glow_strength = static_cast<float>(params[UnderlyingType(RetroVision::eRETRO_VISION_CRT_ARTIFACTS_PHOSPHOR_GLOW_STRENGHT)]->u.fs_d.value);
    rvControls.phosphor_glow_opacity  = static_cast<float>(params[UnderlyingType(RetroVision::eRETRO_VISION_CRT_ARTIFACTS_PHOSPHOR_GLOW_OPACITY)]->u.fs_d.value);

    rvControls.apperture_grill_enable = params[UnderlyingType(RetroVision::eRETRO_VISION_CRT_ARTIFACTS_APPERTURE_GRILL)]->u.bd.value;
    rvControls.mask_type              = static_cast<AppertureGtrill>(params[UnderlyingType(RetroVision::eRETRO_VISION_CRT_ARTIFACTS_APPERTURE_GRILL_POPUP)]->u.pd.value - 1);
    rvControls.mask_interval          = params[UnderlyingType(RetroVision::eRETRO_VISION_CRT_ARTIFACTS_APPERTURE_GRILL_INTERVAL)]->u.sd.value;
    rvControls.mask_darkness          = static_cast<float>(params[UnderlyingType(RetroVision::eRETRO_VISION_CRT_ARTIFACTS_APPERTURE_GRILL_DARKNESS)]->u.fs_d.value);

    rvControls.white_color_hercules = static_cast<HerculesWhiteColor>(params[UnderlyingType(RetroVision::eRETRO_VISION_CRT_ARTIFACTS_HERCULES_WHITE_COLOR)]->u.pd.value - 1);

    return rvControls;
}

#endif // __IMAGE_LAB_RETRO_VISION_FILTER_CONTROLS__