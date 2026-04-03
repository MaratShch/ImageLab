#ifndef __IMAGE_LAB_CHROMATIC_ABERRATION_FILTER_ENUMERATORS__
#define __IMAGE_LAB_CHROMATIC_ABERRATION_FILTER_ENUMERATORS__

#include "AE_Effect.h"
#include "CompileTimeUtils.hpp"
#include "PaintAlgoContols.hpp"

enum class ArtPaintControls : int32_t
{
    ART_PAINT_INPUT,
    ART_PAINT_STYLE,
    ART_PAINT_BRUSH_WIDTH,
    ART_PAINT_BRUSH_LENGTH,
    ART_PAINT_STROKE_CURVATIVE,
    ART_PAINT_STROKE_SPREADING,
    ART_PAINT_TOTAL_PARAMS
};

constexpr A_char ArtPaintControlsStr[5][24] =
{
    "Paint Style",
    "Brush Width",
    "Brush Length",
    "Stroke Curvature",
    "Stroke Spreading"
};

constexpr char StrokeBiasStr[] =
{
    "Dark|"
    "Light|"
    "Balanced"
};


#endif // __IMAGE_LAB_CHROMATIC_ABERRATION_FILTER_ENUMERATORS__