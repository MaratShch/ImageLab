#ifndef __IMAGE_LAB_IMAGE_ART_MOSAIC_ENUMERATORS__
#define __IMAGE_LAB_IMAGE_ART_MOSAIC_ENUMERATORS__

#include <cstdint>

enum class eART_MOSAIC_ITEMS : int32_t
{
    eIMAGE_ART_MOSAIC_INPUT,
    eIMAGE_ART_MOSAIC_CELLS_SLIDER,
    eIMAGE_ART_MOSAIC_TOTAL_CONTROLS
};


constexpr char sliderName[] = { "Cells number" };
constexpr int32_t cellMin = 500;
constexpr int32_t cellMax = 2000;
constexpr int32_t cellDef = 1000;

#endif // __IMAGE_LAB_IMAGE_ART_MOSAIC_ENUMERATORS__