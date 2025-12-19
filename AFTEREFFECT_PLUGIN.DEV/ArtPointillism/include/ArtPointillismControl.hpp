#ifndef __IMAGE_LAB_ART_POINTILISM_CONTROLS_PARAMETERS_DEFINITIONS__
#define __IMAGE_LAB_ART_POINTILISM_CONTROLS_PARAMETERS_DEFINITIONS__

#include "Common.hpp"
#include "CommonAdobeAE.hpp"
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


PF_Err
SetupControlElements
(
    const PF_InData* RESTRICT in_data,
    PF_OutData* RESTRICT out_data
);

PontillismControls GetControlParametersStruct
(
    PF_ParamDef* RESTRICT params[]
) noexcept;


#endif // __IMAGE_LAB_ART_POINTILISM_CONTROLS_PARAMETERS_DEFINITIONS__