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
    GENERATIONS,
    TOTAL_PARAMS
};

constexpr char itemNames[][32] = {
    " ",
    "Film Stock",
    "Film Stock",
    "Film Format",
    "Process Variant"
};

#endif // __IMAGE_LAB2_ALGO_ADOBE_CONTROL_ENUMERATORS__