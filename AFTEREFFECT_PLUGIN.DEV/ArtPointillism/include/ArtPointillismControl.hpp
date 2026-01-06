#ifndef __IMAGE_LAB_ART_POINTILISM_CONTROLS_PARAMETERS_DEFINITIONS__
#define __IMAGE_LAB_ART_POINTILISM_CONTROLS_PARAMETERS_DEFINITIONS__

#include "Common.hpp"
#include "ArtPointillismEnums.hpp"

struct PontillismControls
{
    ArtPointillismPainter   PainterStyle;
    int32_t                 DotDencity;
    int32_t                 DotSize;
    int32_t                 EdgeSensitivity;
    int32_t                 Vibrancy;
    StrokeShape             Shape;
    BackgroundArt           Background;
    int32_t                 Opacity;
    int32_t                 RandomSeed;
};

constexpr size_t PontillismControlsSize = sizeof(PontillismControls);


#endif // __IMAGE_LAB_ART_POINTILISM_CONTROLS_PARAMETERS_DEFINITIONS__