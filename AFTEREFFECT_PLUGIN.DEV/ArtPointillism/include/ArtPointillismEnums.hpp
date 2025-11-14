#ifndef __IMAGE_LAB_ART_POINTILISM_ENUMERATORS__
#define __IMAGE_LAB_ART_POINTILISM_ENUMERATORS__

#include <cstdint>
#include "AE_Effect.h"
#include "CompileTimeUtils.hpp"

enum class ArtPointilismControls : int32_t
{
    ART_POINTILISM_INPUT,
    ART_POINTILISM_LIST_PAINTERS,
    ART_POINTILISM_SLIDER_DOT_SIZE,
    ART_POINTILISM_FSLIDER_DOT_DENCITY,
    ART_POINTILISM_FSLIDER_COLOR_FIDELITY,
    ART_POINTILISM_BLENDING_MODE_LIST,
    ART_POINTILISM_STROKE_SHAPE_LIST,
    ART_POINTILISM_FSLIDER_EDGE_SENSITIVITY,
    ART_POINTILISM_SLIDER_BACKGROUND_COLOR,
    ART_POINTILISM_TOTAL_PARAMS
};


enum class ArtPointilismPainter : int32_t
{
    ART_POINTILISM_PAINTER_SEURAT,
    ART_POINTILISM_PAINTER_SIGNAC,
    ART_POINTILISM_PAINTER_CROSS,
    ART_POINTILISM_PAINTER_PISSARO,
    ART_POINTILISM_PAINTER_VAN_GOGH,
    ART_POINTILISM_PAINTER_MATISSE,
    ART_POINTILISM_PAINTER_RYSSELBERGHE,
    ART_POINTILISM_PAINTER_LUCE,
    ART_POINTILISM_PAINTER_TOTAL_NUMBER
};

constexpr char PainterNameStr [] = 
{
    "Georges Seurat|"
    "Paul Signac|"
    "Henri-Edmond Cross|"
    "Camille Pissarro|"
    "Vincent van Gogh|"
    "Henri Matisse|"
    "Theo van Rysselberghe|"
    "Maximilien Luce"
};

constexpr uint32_t dotSizeMin = 1u;
constexpr uint32_t dotSizeMax = 10u;
constexpr uint32_t dotSizeDef = 3u;

constexpr double fDotDencityMin = 0.0;
constexpr double fDotDencityMax = 1.0;
constexpr double fDotDencityDef = 0.6;

constexpr double fColorFidelityMin = 0.0;
constexpr double fColorFidelityMax = 1.0;
constexpr double fColorFidelityDef = 0.8;

enum class ArtPointilismBlending : int32_t
{
    ART_POINTILISM_BLEND_NONE,
    ART_POINTILISM_BLEND_ALPHA,
    ART_POINTILISM_BLEND_ADDITIVE,
    ART_POINTILISM_BLEND_TOTAL_NUMBER
};

constexpr char PointillismBlendModeStr[] =
{
    "None|"
    "Alpha|"
    "Additive"
};

enum class ArtPointilismStroke : int32_t
{
    ART_POINTILISM_STROKE_CIRCLE,
    ART_POINTILISM_STROKE_ELLIPSE,
    ART_POINTILISM_STROKE_STROKE,
    ART_POINTILISM_STROKE_TOTAL_NUMBER
};

constexpr char PointillismStrokeStr[] =
{
    "Circle|"
    "Ellipse|"
    "Stroke"
};

constexpr double fEdgeSensitiveMin = 0.0;
constexpr double fEdgeSensitiveMax = 1.0;
constexpr double fEdgeSensitiveDef = 0.5;

constexpr double fBackgroundColorMin = -1.0;
constexpr double fBackgroundColorMax = 1.0;
constexpr double fBackgroundColorDef = 0.0;

constexpr char controlItemName[][24] =
{
    "Painter Name",
    "Dot size",
    "Dot Dencity",
    "Color Fidelity",
    "Blending Mode",
    "Stroke Shape",
    "Edge Sensitivity",
    "Background Color"
};

#endif // __IMAGE_LAB_ART_POINTILISM_ENUMERATORS__