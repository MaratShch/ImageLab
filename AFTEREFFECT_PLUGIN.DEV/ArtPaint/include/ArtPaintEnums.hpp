#ifndef __IMAGE_LAB_CHROMATIC_ABERRATION_FILTER_ENUMERATORS__
#define __IMAGE_LAB_CHROMATIC_ABERRATION_FILTER_ENUMERATORS__

#include "AE_Effect.h"
#include "CompileTimeUtils.hpp"
#include "PaintAlgoContols.hpp"

enum class ArtPaintControls : int32_t
{
    ART_PAINT_INPUT,
    ART_PAINT_RENDER_QUALITY,
    ART_PAINT_STYLE,
    ART_PAINT_BRUSH_WIDTH,
    ART_PAINT_BRUSH_LENGTH,
    ART_PAINT_STROKE_CURVATIVE,
    ART_PAINT_STROKE_SPREADING,
    ART_PAINT_TOTAL_PARAMS
};

constexpr A_char ArtPaintControlsStr[][24] =
{
    "Render Quality",
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

constexpr char RenderQualityStr[] = 
{
    "Fast|"
    "Accurate"
};

#endif // __IMAGE_LAB_CHROMATIC_ABERRATION_FILTER_ENUMERATORS__