#ifndef __IMAGE_LAB_AUTHOMATIC_WB2_ALGO_ENUMS__
#define __IMAGE_LAB_AUTHOMATIC_WB2_ALGO_ENUMS__

#include <cstdint>
#include "CompileTimeUtils.hpp"
#include "ColorTransformMatrix.hpp"

enum class eImageLab2AWB_Controls : int32_t
{
    AWB2_INPUT = 0,
    AWB2_COLOR_SPACE_POPUP,
    AWB2_ILLUMINATE_POPUP,
    AWB2_CHROMATIC_POPUP,
    AWB2_EXTERME_PIXELS,
    AWB2_SATRURATION_THRESHOLD,
    AWB2_BLACK_LEVEL_THRESHOLD,
    AWB2_TOTAL_CONTROLS
};

constexpr char strColorSpace[] =
{
    "BT.601|"
    "BT.709|"
    "BT.2020|"
    "SMPTE240M"
};

constexpr int32_t  gTotalNumbersOfColorSpaces = (static_cast<int32_t>(SMPTE240M) + 1);
constexpr int32_t  gDefNumberOfColorSpace = UnderlyingType(BT601);

constexpr char strCtrlNames[][24] = 
{
    "Color Space",
    "Illuminant",
    "Chromatic Adaptation",
    "Extreme Pixels %",
    "Saturation Threshold",
    "Black Level Threshold"
};

enum class eILLUMINATE : int32_t
{
    DAYLIGHT = 0,
    OLD_DAYLIGHT,
    OLD_DIRECT_SUNLIGHT_AT_NOON,
    MID_MORNING_DAYLIGHT,
    NORTH_SKY_DAYLIGHT,
    DAYLIGHT_FLUORESCENT_F1,
    COOL_FLUERESCENT,
    WHITE_FLUORESCENT,
    WARM_WHITE_FLUORESCENT,
    DAYLIGHT_FLUORESCENT_F5,
    COOL_WHITE_FLUORESCENT,
    TOTAL_ILLUMINANTES
};

constexpr char strIlluminantName[] = 
{
    "Daylight (D65)|"
    "Old Daylight|"
    "Old Direct Sunlight at Noon|"
    "Mid Morning Daylight|"
    "Nort Sky Dayligth|"
    "Dayligth Fluorescent F1|"
    "Cool Fluorescent|"
    "White Fluorescent|"
    "Warm White Fluorescent|"
    "Daylight Fluorescent F5|"
    "Cool White Fluorescent"
};


enum class eChromaticAdaptation : int32_t
{
    CHROMATIC_CAT02 = 0,
    CHROMATIC_VON_KRIES,
    CHROMATIC_BRADFORD,
    CHROMATIC_SHARP,
    CHROMATIC_CMCCAT2000,
    TOTAL_CHROMATIC
};


constexpr char strChtomaticAdaptation[] = 
{
    "CAT02|"
    "Von Kries|"
    "Bradford|"
    "Sharp|"
    "Cmccat2000"
};

constexpr float extremePixMin = 1.0f;
constexpr float extremePixMax = 10.0f;
constexpr float extremePixDef = 3.5f;

constexpr float saturationThrMin = 0.80f;
constexpr float saturationThrMax = 1.00f;
constexpr float saturationThrDef = 0.95f;

constexpr float blackLevelThresholdMin = 0.00f;
constexpr float blackLevelThresholdMax = 0.10f;
constexpr float blackLevelThresholdDef = 0.02f;



#endif // __IMAGE_LAB_AUTHOMATIC_WB2_ALGO_ENUMS__