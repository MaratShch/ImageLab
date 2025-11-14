#ifndef __IMAGE_LAB_ART_POINTILISM_CONTROLS_PARAMETERS_DEFINITIONS__
#define __IMAGE_LAB_ART_POINTILISM_CONTROLS_PARAMETERS_DEFINITIONS__

#include "CommonAdobeAE.hpp"
#include "ArtPointillismEnums.hpp"

struct PontillismControls
{
    ArtPointilismPainter    ctrlPainter;
    uint32_t                ctrlDotSize;
    float                   ctrlDotDensity;
    float                   ctrlColorFidelity;
    ArtPointilismBlending   ctrlBlending;
    ArtPointilismStroke     ctrlStroke;
    float                   ctrlEdgeSensitivity;
    float                   ctrlBackgroundColor;
};

constexpr size_t PontillismControlsSize = sizeof(PontillismControls);


PF_Err
SetupControlElements
(
    const PF_InData*  in_data,
    PF_OutData* out_data
);

PontillismControls GetControlParametersStruct
(
    PF_ParamDef* __restrict params[]
) noexcept;


#endif // __IMAGE_LAB_ART_POINTILISM_CONTROLS_PARAMETERS_DEFINITIONS__