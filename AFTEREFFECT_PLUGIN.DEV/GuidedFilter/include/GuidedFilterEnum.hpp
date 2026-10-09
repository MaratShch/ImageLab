#ifndef __IMAGE_LAB_VIDEO_STABILIZATION_ITEMS_ENUMERATORS__
#define __IMAGE_LAB_VIDEO_STABILIZATION_ITEMS_ENUMERATORS__

#include <cstdint>
#include "CompileTimeUtils.hpp"

enum class CtrlItems : uint32_t
{
    gFILTER_INPUT = 0,
    gFILTER_MODE,
    gFILTER_RADIUS,
    gFILTER_EPSILON,
    gFILTER_SHOW_MAP,
    gFILTER_SKIN_TOLERANCE,
    gFILTER_SKIN_SOFTNESS,
    gFILTER_TOTAL_PARAMETERS
};

constexpr char ctrlItemNames[][32] = 
{
    "Filter Mode",
    "Radius",
    "Epsilon",
    "Show Output",
    "Skin Tolerance",
    "Skin Softness"
};


// ============================================================================
// Processing mode (Render control: ListBox / popup).
// The numeric values are part of the contract between the Render controls and
// the algorithm: ListBox item index == static_cast<uint32_t>(FilterMode).
//   Item 0: "Entire Image"   Item 1: "Human Skin"   Item 2: "Background"
// ============================================================================
enum class FilterMode : uint32_t
{
    Entire_Image = 0,
    Skin,
    Background,
    TotalModes
};

constexpr char gFilterModeStr[] =
{
    "Entire Image|"
    "Skin|"
    "Background"
};


// ============================================================================
// Control ranges and defaults (for the Render-control definitions and for the
// algorithm-side validation in Algorithm_Main). Values outside a range are
// clamped by the algorithm; non-finite floats are replaced by the default.
// ============================================================================
constexpr int32_t AlgoRadiusMin = 0;        // 0 = identity
constexpr int32_t AlgoRadiusMax = 255;
constexpr int32_t AlgoRadiusDef = 8;

constexpr float   AlgoEpsilonMin = 0.00001f;
constexpr float   AlgoEpsilonMax = 1.0f;
constexpr float   AlgoEpsilonDef = 0.01f;

constexpr float   AlgoSkinToleranceMin = 0.25f;
constexpr float   AlgoSkinToleranceMax = 2.0f;
constexpr float   AlgoSkinToleranceDef = 1.0f;
constexpr float   AlgoSkinToleranceStep = 0.05f;

constexpr float   AlgoSkinSoftnessMin = 0.0f;
constexpr float   AlgoSkinSoftnessMax = 1.0f;
constexpr float   AlgoSkinSoftnessDef = 0.25f;
constexpr float   AlgoSkinSoftnessStep = 0.05f;


#endif // __IMAGE_LAB_VIDEO_STABILIZATION_ITEMS_ENUMERATORS__
