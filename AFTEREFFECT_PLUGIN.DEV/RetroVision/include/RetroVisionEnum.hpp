#ifndef __IMAGE_LAB_RETRO_VISION_FILTER_ENUMERATORS__
#define __IMAGE_LAB_RETRO_VISION_FILTER_ENUMERATORS__

#include "AE_Effect.h"
#include "CompileTimeUtils.hpp"

enum class RetroVision : int32_t
{
    eRETRO_VISION_INPUT,
    eRETRO_VISION_TOTAL_CONTROLS
};


enum class RetroBitmap : int32_t
{
    eRETRO_BITMAP_CGA = 100,
    eRETRO_BITMAP_EGA,
    eRETRO_BITMAP_VGA,
    eRETRO_BITMAP_HERCULES
};


#endif // __IMAGE_LAB_RETRO_VISION_FILTER_ENUMERATORS__