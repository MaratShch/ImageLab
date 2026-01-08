#ifndef __IMAGE_LAB_ART_POINTILISM_ENUMERATORS__
#define __IMAGE_LAB_ART_POINTILISM_ENUMERATORS__

#include <cstdint>
#include "CompileTimeUtils.hpp"

enum class ArtPointillismControls : int32_t
{
    ART_POINTILLISM_INPUT,
    ART_POINTILLISM_PAINTER_STYLE,
    ART_POINTILLISM_SLIDER_DOT_DENCITY,
    ART_POINTILLISM_SLIDER_DOT_SIZE,
    ART_POINTILLISM_SLIDER_EDGE_SENSITIVITY,
    ART_POINTILLISM_SLIDER_COLOR_VIBRANCE,
    ART_POINTILLISM_STROKE_STROKE_SHAPE,
    ART_POINTILLISM_BACKGROUND_ART,
    ART_POINTILLISM_OPACITY,
    ART_POINTILLISM_RANDOM_SEED,
    ART_POINTILLISM_TOTAL_PARAMS
};

constexpr char controlItemName[][24] =
{
    "Painter Style",
    "Dot Dencity",
    "Dot Size",
    "Edge Sensitivity",
    "Vibrancy",
    "Stroke Shape",
    "Background",
    "Background Opacity",
    "Random Seed"
};

enum class ArtPointillismPainter : int32_t
{
    ART_POINTILLISM_PAINTER_SEURAT,
    ART_POINTILLISM_PAINTER_SIGNAC,
    ART_POINTILLISM_PAINTER_CROSS,
    ART_POINTILLISM_PAINTER_PISSARRO,
    ART_POINTILLISM_PAINTER_VAN_GOGH,
    ART_POINTILLISM_PAINTER_MATISSE,
    ART_POINTILLISM_PAINTER_RYSSELBERGHE,
    ART_POINTILLISM_PAINTER_LUCE,
    ART_POINTILLISM_PAINTER_TOTAL_NUMBER
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

constexpr int32_t DotDencityMin = 0;
constexpr int32_t DotDencityMax = 250;
constexpr int32_t DotDencityDef = AverageValue(DotDencityMin, DotDencityMax);

constexpr int32_t DotSizeMin = 0;
constexpr int32_t DotSizeMax = 100;
constexpr int32_t DotSizeDef = 15;

constexpr int32_t EdgeSensitivityMin = 0;
constexpr int32_t EdgeSensitivityMax = 100;
constexpr int32_t EdgeSensitivityDef = 40;

constexpr int32_t ColorVibrancyMin = -100;
constexpr int32_t ColorVibrancyMax = 100;
constexpr int32_t ColorVibrancyDef = AverageValue(ColorVibrancyMin, ColorVibrancyMax);


enum class StrokeShape : int32_t
{
    ART_POINTILLISM_SHAPE_CIRCLE,
    ART_POINTILLISM_SHAPE_ELLIPSE,
    ART_POINTILLISM_SHAPE_SQUARE,
    ART_POINTILLISM_SHAPE_TOTALS
};

constexpr char StrokeShapeStr [] =
{
    "Circle|"
    "Ellipse|"
    "Square"
};

enum class BackgroundArt : int32_t
{
    ART_POINTILLISM_BACKGROUND_CANVAS,
    ART_POINTILLISM_BACKGROUND_WHITE,
    ART_POINTILLISM_BACKGROUND_SOURCE_IMAGE,
    ART_POINTILLISM_BACKGROUND_TOTALS
};

constexpr char BackgroundStr [] =
{
    "Canvas|"
    "White|"
    "Source Image"
};

constexpr int32_t OpacityMin = 0;
constexpr int32_t OpacityMax = 100;
constexpr int32_t OpacityDef = OpacityMin;

constexpr int32_t RandomSeedMin = 0;
constexpr int32_t RandomSeedMax = 32767;
constexpr int32_t RandomSeedDef = 0;


enum class ColorMode : int32_t
{
    Scientific,     // Strict Decomposition
    Expressive      // Saturation Boost / Modulation
};


#endif // __IMAGE_LAB_ART_POINTILISM_ENUMERATORS__