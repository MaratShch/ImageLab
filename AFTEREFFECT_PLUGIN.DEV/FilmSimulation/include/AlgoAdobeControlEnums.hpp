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
    GROUP_STOP_DEVELOPMENT,
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
    "Development",
    "Development Enable",
    "Development Time",
    "Development Temperature"
};

#endif // __IMAGE_LAB2_ALGO_ADOBE_CONTROL_ENUMERATORS__