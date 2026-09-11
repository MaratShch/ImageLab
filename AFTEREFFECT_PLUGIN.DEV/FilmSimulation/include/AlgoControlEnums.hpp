#pragma once

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


// filmFormat in AlgoControl structure
enum class FilmFormatCtrl : int32_t
{
    eFILM_FORMAT_8_MM,
    eFILM_FORMAT_SUPER_8,
    eFILM_FORMAT_16_MM,
    eFILM_FORMAT_SUPER_16,
    eFILM_FORMAT_ACADEMY_35,
    eFILM_FORMAT_ANAMORPHIC_35,
    eFILM_FORMAT_SUPER_35,
    eFILM_FORMAT_TECHNI_35,
    eFILM_FORMAT_FF_35,
    eFILM_FORMAT_MEDIUM_645,
    eFILM_FORMAT_IMAX_15,
    eFILM_FORMAT_POLAROID_SX_70,
    eFILM_FORMAT_POLAROID_PACK,
    eFILM_FORMAT_LARGE_4x5,
    eFILM_FORMAT_TOTAL_FORMATS
};
// string representation for filmFormat in AlgoControl structure
constexpr char FilmFormatCtrlStr[] = 
{
    "8 mm|"
    "super 8|"
    "16 mm|"
    "super 16|"
    "academy 35|"
    "anamorphic 35|"
    "super 35|"
    "techni 35|"
    "ff 35|"
    "medium 645|"
    "imax 15|"
    "polaroid sx70|"
    "polaroid pack|"
    "large 4x5"
};

// processVariant in AlgoControl structure
enum class FilmProcessVariant : int32_t
{
    eFILM_PROCESS_EI_1600 = 0,
    eFILM_PROCESS_CS2_TWO_BATH_KIT,
    eFILM_PROCESS_TOTAL
};
// string representation for processVariant in AlgoControl structure
constexpr char FilmProcessVariant[] =
{
    "EI 1600|"
    "CS2 TWO BATHS KIT" 
};

