#ifndef __IMAGE_LAB_RETRO_VISION_FILTER_ENUMERATORS__
#define __IMAGE_LAB_RETRO_VISION_FILTER_ENUMERATORS__

#include <cstdint>
#include "AE_Effect.h"
#include "CompileTimeUtils.hpp"

enum class RetroVision : int32_t
{
    eRETRO_VISION_INPUT,
    eRETRO_VISION_GAMMA_ADJUST,
    eRETRO_VISION_ENABLE,
    eRETRO_VISION_MONITOR_TYPE_START,
    eRETRO_VISION_GUI,
    eRETRO_VISION_DISPLAY,
    eRETRO_VISION_CGA_PALETTE,
    eRETRO_VISION_CGA_INTTENCITY_BIT,
    eRETRO_VISION_EGA_PALETTE,
    eRETRO_VISION_VGA_PALETTE,
    eRETRO_VISION_HERCULES_THRESHOLD,
    eRETRO_VISION_MONITOR_TYPE_STOP,
    eRETRO_VISION_CRT_ARTIFACTS_START,
    eRETRO_VISION_CRT_ARTIFACTS_SCANLINES,
    eRETRO_VISION_CRT_ARTIFACTS_SMOOTH_SCANLINES,
    eRETRO_VISION_CRT_ARTIFACTS_SCANLINES_INTERVAL,
    eRETRO_VISION_CRT_ARTIFACTS_SCANLINES_DARKNESS,
    eRETRO_VISION_CRT_ARTIFACTS_PHOSPHOR_GLOW,
    eRETRO_VISION_CRT_ARTIFACTS_PHOSPHOR_GLOW_STRENGHT,
    eRETRO_VISION_CRT_ARTIFACTS_PHOSPHOR_GLOW_OPACITY,
    eRETRO_VISION_CRT_ARTIFACTS_APPERTURE_GRILL,
    eRETRO_VISION_CRT_ARTIFACTS_APPERTURE_GRILL_POPUP,
    eRETRO_VISION_CRT_ARTIFACTS_APPERTURE_GRILL_INTERVAL,
    eRETRO_VISION_CRT_ARTIFACTS_APPERTURE_GRILL_DARKNESS,
    eRETRO_VISION_CRT_ARTIFACTS_APPERTURE_GRILL_COLOR,
    eRETRO_VISION_CRT_ARTIFACTS_HERCULES_WHITE_COLOR,
    eRETRO_VISION_CRT_ARTIFACTS_STOP,
    eRETRO_VISION_TOTAL_CONTROLS
};

constexpr double gammaSliderMin = 0.5;
constexpr double gammaSliderMax = 2.5;
constexpr double gammaSliderDef = 1.0;

enum class RetroMonitor : int32_t
{
    eRETRO_BITMAP_CGA,
    eRETRO_BITMAP_EGA,
    eRETRO_BITMAP_VGA,
    eRETRO_BITMAP_HERCULES,
    eRETRO_BITMAP_TOTALS
};

constexpr char retroMonitorName[] =
{
    "CGA |"
    "EGA |"
    "VGA |"
    "HERCULES"
};


enum class PaletteCGA : int32_t
{
    eRETRO_PALETTE_CGA1,
    eRETRO_PALETTE_CGA2,
    eRETRO_PALETTE_CGA_TOTAL
};

constexpr char cgaPaletteName[] =
{
    "CGA-1 |"
    "CGA-2 "
};


enum class PaletteEGA : int32_t
{
    eRETRO_PALETTE_EGA_STANDARD,
    eRETRO_PALETTE_EGA_KING_QUESTS,
    eRETRO_PALETTE_EGA_KYRANDIA,
    eRETRO_PALETTE_EGA_THEXDER,
    eRETRO_PALETTE_EGA_DUNE,
    eRETRO_PALETTE_EGA_DOOM,
    eRETRO_PALETTE_EGA_METAL_MUTANT,
    eRETRO_PALETTE_EGA_WOLFENSTEIN,
    eRETRO_PALETTE_EGA_TOTAL
};

constexpr char egaPaletteName[] =
{
    "Standard |"
    "King Quest |"
    "Legend of Kyrandia |"
    "Thexder |"
    "Dune |"
    "Doom |"
    "Metal Mutant |"
    "Wolfenstein 3D"
};


enum class PaletteVGA : int32_t
{
    eRETRO_PALETTE_VGA_16_BITS,
    eRETRO_PALETTE_VGA_256_BITS,
    eRETRO_PALETTE_VGA_TOTAL
};

constexpr char vgaPaletteName[] =
{
    "16 bits |"
    "256 bits "
};

constexpr int32_t herculesThresholdMin = 32;
constexpr int32_t herculesThresholdMax = 192;
constexpr int32_t herculesThresholdDef = 128;

constexpr int32_t scanLinesSliderMin = 1;
constexpr int32_t scanLinesSliderMax = 6;
constexpr int32_t scanLinesSliderDef = 2;
constexpr int32_t scanLinesSliderHerculesDef = 3;

constexpr double scanLinesDarknessMin = 0.0;
constexpr double scanLinesDarknessMax = 1.0;
constexpr double scanLinesDarknessDef = 0.2;

constexpr double phosphorGlowStrengthMin = 0.0;
constexpr double phosphorGlowStrengthMax = 3.0;
constexpr double phosphorGlowStrengthDef = 2.0;

constexpr double phosphorGlowOpacityMin = 0.0;
constexpr double phosphorGlowOpacityMax = 1.0;
constexpr double phosphorGlowOpacityDef = 0.3;

enum class AppertureGtrill : int32_t
{
    eRETRO_APPERTURE_NONE,
    eRETRO_APPERTURE_SHADOW_MASK,
    eRETRO_APPERTURE_APPERTURE_GRILL,
    eRETRO_APPERTURE_TOTALS
};

constexpr char appertureMaskType[] =
{
    "None|"
    "ShadowMask|"
    "Apperture Grill"
};

constexpr int32_t appertureMaskMin = 2;
constexpr int32_t appertureMaskMax = 8;
constexpr int32_t appertureMaskDef = 2;
constexpr int32_t appertureMaskDefHercules = 3;

constexpr double appertureMaskDarkMin = 0.0;
constexpr double appertureMaskDarkMax = 1.0;
constexpr double appertureMaskDarkDef = 0.25;

constexpr int32_t appertureMaskColorMin = 0;
constexpr int32_t appertureMaskColorMax = 64;
constexpr int32_t appertureMaskColorDef = 20;

constexpr char herculesWhiteName[] =
{
    "Pure White|"
    "Little Bluish|"
    "Bluish"
};

enum class HerculesWhiteColor : int32_t
{
    eHERCULES_PURE_WHITE,
    eHERCULES_LITTLE_BLUISH,
    eHERCULES_BLUISH,
    eHERCULES_WHITE_TOTALS
};

constexpr char controlItemName[][24] =
{
    "Gamma Adjust",		    // float slider
    "Enable",				// check box
    "Monitor/Palette",      // New Chapter 
    "Palette",  			// UI color bar
    "Monitor",				// popup
    "CGA Palette",   		// popup
    "Intencity bit",   		// check box
    "EGA Palette",   		// popup
    "VGA Palette",          // popup
    "Hercules Threshold",   // Hercules B/W threshold
    "CRT Artifacts",        // New Chapter
    "Scanlines",            // check box
    "Smooth Scanlines",     // check box
    "Scanlines Interval",   // int slider
    "Scanlines Darkness",   // float slider
    "Phosphor Glow",        // check box
    "Glow Strength",        // slider
    "Glow Opacity",         // slider
    "Aperture Grill",       // check box
    "Mask Type",            // popup
    "Mask Interval",        // slider
    "Mask Darkness",        // slider
    "Mask Color",           // color
    "White Color & Tint"    // white color on Hercules (popup)
};


#endif // __IMAGE_LAB_RETRO_VISION_FILTER_ENUMERATORS__