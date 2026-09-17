#ifndef __IMAGE_LAB2_ALGO_ADOBE_CONTROL_ENUMERATORS__
#define __IMAGE_LAB2_ALGO_ADOBE_CONTROL_ENUMERATORS__

#include "CompileTimeUtils.hpp"

enum class FilmSimulationCtrl : int32_t
{
    VIDEO_INPUT,
    SETUP_BUTTON, // button for run dialog control
    GROUP_START_FILM_PROPERTIES,
    FILM_STOCK,
    FILM_FORMAT,
    PROCESS_VARIANT,
    GROUP_STOP_FILM_PROPERTIES,
    GROUP_START_EXPOSURE_AND_TONE,
    EXPOSURE,
    EXPOSURE_TIME,
    MID_GRAY_TARGET,
    BLACK_POINT_STRETCH,
    GROUP_STOP_EXPOSURE_AND_TONE,
    GROUP_START_DEVELOPMENT,
    DEVELOPMENT_ENABLE,
    DEVELOPMENT_TIME,
    DEVELOPMENT_TEMPERATURE,
    YEARS_OF_DARK_STORAGE,
    GROUP_STOP_DEVELOPMENT,
    GROUP_START_COLOR_WHITE_BALANCE,
    SCENE_COLOUR_TEMPERATURE,
    WHITE_BALANCE_STRENGTH,
    GROUP_STOP_COLOR_WHITE_BALANCE,
    TOTAL_PARAMS
};

constexpr char itemNames[][32] = {
    " ",
    "Film Stock",
    "Film Stock",
    "Film Format",
    "Process Variant",
    "Exposure & Tone",
    "Exposure",
    "Exposure Time",
    "Mid-Gray Target",
    "Black Point Stretch",
    "Development And Storage",
    "Development Enable",
    "Development Time",
    "Development Temperature",
    "Years of Dark Storage",
    "Colour & White Balance",
    "Scene Colour Temperature",
    "White Balance Strength"
};

#endif // __IMAGE_LAB2_ALGO_ADOBE_CONTROL_ENUMERATORS__