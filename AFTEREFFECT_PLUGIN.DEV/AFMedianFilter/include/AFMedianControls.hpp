#ifndef __IMAGE_LAB_ADAPTIVE_FREQUENCY_MEDIAN_FILTER_CONTROLS__
#define __IMAGE_LAB_ADAPTIVE_FREQUENCY_MEDIAN_FILTER_CONTROLS__

#include <cstdint>
#include "AE_Effect.h"
#include "AFMedianFilterEnum.hpp"

struct AfmfControls
{
    int32_t afmfRadius;
    float   afmfTolerance;
    int32_t afmfIterations;

    constexpr AfmfControls(void) : afmfRadius(kerenlRadiusMin), afmfTolerance(noiseToleranceDef), afmfIterations(iterPassMin) {};
};


const AfmfControls getAlgoControls (PF_ParamDef* params[]);


#endif // __IMAGE_LAB_ADAPTIVE_FREQUENCY_MEDIAN_FILTER_CONTROLS__