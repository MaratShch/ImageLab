#ifndef __IMAGE_LAB_RETRO_VISION_FILTER_ENUMERATORS__
#define __IMAGE_LAB_RETRO_VISION_FILTER_ENUMERATORS__

#include <cstdint>
#include "AE_Effect.h"
#include "CompileTimeUtils.hpp"

enum class RetroVision : int32_t
{
    eRETRO_VISION_INPUT,
    eRETRO_VISION_ENABLE,
    eRETRO_VISION_GUI,
    eRETRO_VISION_DISPLAY,
    eRETRO_VISION_DITHERING,
    eRETRO_VISION_TOTAL_CONTROLS
};

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


constexpr uint32_t guiBarWidth  = 48u;
constexpr uint32_t guiBarHeight = 48u;

constexpr char controlItemName[][16] =
{
    "Enable",				// check box
    "Display",  			// UI color bar
    "Display",				// popup
    "CGA Palette",   		// popup
    "Intencity bit",   		// check box
    "EGA Palette",   		// popup
    "VGA Palette",          // popup
    "Scanlines",            // slider
    "Phosphor Glow",        // check box
    "Aperture Grill"       
};


#endif // __IMAGE_LAB_RETRO_VISION_FILTER_ENUMERATORS__