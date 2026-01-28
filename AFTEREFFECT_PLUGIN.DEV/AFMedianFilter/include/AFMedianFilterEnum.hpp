#ifndef __IMAGE_LAB_IMAGE_GEOMETRY_ENUMERATORS_FILTER__
#define __IMAGE_LAB_IMAGE_GEOMETRY_ENUMERATORS_FILTER__

#include <cstdint>
#include "AE_Effect.h"

enum class AFMF : int32_t
{
    eIMAGE_AFMEDIAN_INPUT,
    eIMAGE_AFMEDIAN_TOTAL_CONTROLS
};

constexpr int32_t kerenlSizeMin = 3;
constexpr int32_t kernelSizeMax = 31;


#endif // __IMAGE_LAB_IMAGE_GEOMETRY_ENUMERATORS_FILTER__