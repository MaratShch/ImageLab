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


enum class RetroBitmap : int32_t
{
    eRETRO_BITMAP_CGA1,
    eRETRO_BITMAP_CGA2,
    eRETRO_BITMAP_EGA,
    eRETRO_BITMAP_VGA16,
    eRETRO_BITMAP_VGA256,
    eRETRO_BITMAP_HERCULES,
    eRETRO_BITMAP_TOTALS
};

constexpr uint32_t guiBarWidth  = 48u;
constexpr uint32_t guiBarHeight = 48u;

constexpr char controlItemName[][16] =
{
    "Enable",				// check box
    "Display",  			// UI color bar
    "Display",				// popup
    "Dithering"             // checkbox
};

constexpr char retroDisplayName[] =
{
    "CGA-1   |"
    "CGA-2   |"
    "EGA     |"
    "VGA-16  |"
    "VGA-256 |"
    "HERCULES"
};


#endif // __IMAGE_LAB_RETRO_VISION_FILTER_ENUMERATORS__