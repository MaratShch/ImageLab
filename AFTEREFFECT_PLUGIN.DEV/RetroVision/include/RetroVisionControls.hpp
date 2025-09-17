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
    int32_t scan_lines_interval;
    float   scan_lines_darkness;
    int32_t phosphor_glow_enable;
    float   phosphor_glow_strength;
    float   phosphor_glow_opacity;
    int32_t apperture_grill_enable;
    int32_t mask_type;
    int32_t mask_interval;
    float   mask_darkness;
    int32_t mask_color;
    int32_t white_color_hercules;
};

constexpr size_t RVControlsSize = sizeof(RVControls);


inline RVControls GetControlParametersStruct
(
    PF_ParamDef* __restrict params[]
) noexcept
{
    RVControls rvControls{};

    rvControls.monitor           = static_cast<RetroMonitor>(params[UnderlyingType(RetroVision::eRETRO_VISION_DISPLAY)]->u.pd.value);
    rvControls.cga_palette       = static_cast<PaletteCGA>(params[UnderlyingType(RetroVision::eRETRO_VISION_CGA_PALETTE)]->u.pd.value);
    rvControls.cga_intencity_bit = params[UnderlyingType(RetroVision::eRETRO_VISION_CGA_INTTENCITY_BIT)]->u.bd.value;
    rvControls.ega_palette       = static_cast<PaletteEGA>(params[UnderlyingType(RetroVision::eRETRO_VISION_EGA_PALETTE)]->u.pd.value);
    rvControls.vga_palette       = static_cast<PaletteVGA>(params[UnderlyingType(RetroVision::eRETRO_VISION_VGA_PALETTE)]->u.pd.value);

    rvControls.hercules_threshold = params[UnderlyingType(RetroVision::eRETRO_VISION_HERCULES_THRESHOLD)]->u.sd.value;


    return rvControls;
}

#endif // __IMAGE_LAB_RETRO_VISION_FILTER_CONTROLS__