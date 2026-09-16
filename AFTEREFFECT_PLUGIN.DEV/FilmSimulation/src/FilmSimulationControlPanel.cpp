#include "Common.hpp"
#include "AlgoControlEnums.hpp"
#include "AlgoAdobeControlEnums.hpp"
#include "FilmSimulation.hpp"
#include "film_enum.hpp"

CACHE_ALIGN constexpr char filmList[] = {
    #include "film_names.txt"
};



PF_Err SetupControlElements (PF_InData* in_data, PF_OutData* out_data)
{
    CACHE_ALIGN PF_ParamDef	def;

    constexpr PF_ParamFlags   flags = PF_ParamFlag_SUPERVISE | PF_ParamFlag_CANNOT_TIME_VARY | PF_ParamFlag_CANNOT_INTERP;
    constexpr PF_ParamUIFlags ui_flags = PF_PUI_CONTROL;
    constexpr PF_ParamUIFlags ui_disabled_flags = ui_flags | PF_PUI_DISABLED;

    constexpr char ButtonTitle[] = "ImageLab2 Interface";

    A_long totalParams = 0;

    // Setup Button
    AEFX_INIT_PARAM_STRUCTURE(def, flags, ui_flags);
    PF_ADD_BUTTON(
        itemNames[0],
        ButtonTitle,
        ui_flags,
        flags,
        UnderlyingType(FilmSimulationCtrl::SETUP_BUTTON)
    );
    totalParams++;

    ///////////////////////////////////////////////////////////////////
    // GROUP START: FILM STOCK                                       //
    ///////////////////////////////////////////////////////////////////
    // Open group item
    AEFX_INIT_PARAM_STRUCTURE(def, flags, ui_flags);
    PF_ADD_TOPICX(
        itemNames[1],
        ui_disabled_flags,
        UnderlyingType(FilmSimulationCtrl::GROUP_START_FILM_PROPERTIES));
    totalParams++;

    // Films LixtBox
    AEFX_INIT_PARAM_STRUCTURE(def, flags, ui_flags);
    PF_ADD_POPUP(
        itemNames[2],
        UnderlyingType(film::eFILM_PROFILE::eTOTAL_FILMS_PROFILES),
        UnderlyingType(film::eFILM_PROFILE::eAGFA_APX_25),
        filmList,
        UnderlyingType(FilmSimulationCtrl::FILM_STOCK));
    totalParams++;

    // Film Format ListBox
    AEFX_INIT_PARAM_STRUCTURE(def, flags, ui_flags);
    PF_ADD_POPUP(
        itemNames[3],
        UnderlyingType(FilmFormatCtrl::eFILM_FORMAT_TOTAL_FORMATS),
        UnderlyingType(FilmFormatCtrl::eFILM_FORMAT_FF_35) + 1,
        FilmFormatCtrlStr,
        UnderlyingType(FilmSimulationCtrl::FILM_FORMAT));
    totalParams++;

//    AEFX_INIT_PARAM_STRUCTURE(def, flags, ui_flags);
//    PF_ADD_POPUP(
//        itemNames[4],
//        UnderlyingType(FilmFormatCtrl::eFILM_FORMAT_TOTAL_FORMATS),
//        UnderlyingType(FilmFormatCtrl::eFILM_FORMAT_FF_35),
//        FilmFormatCtrlStr,
//        UnderlyingType(FilmSimulationCtrl::PROCESS_VARIANT));
//    totalParams++;

    // Close group item
    AEFX_INIT_PARAM_STRUCTURE(def, flags, ui_flags);
    PF_END_TOPIC(UnderlyingType(FilmSimulationCtrl::GROUP_STOP_FILM_PROPERTIES));
    totalParams++;

    ///////////////////////////////////////////////////////////////////
    // GROUP START: EXPOSURE & TONE                                  //
    ///////////////////////////////////////////////////////////////////
    // Open group item
    AEFX_INIT_PARAM_STRUCTURE(def, flags, ui_flags);
    PF_ADD_TOPICX(
        itemNames[5],
        ui_disabled_flags,
        UnderlyingType(FilmSimulationCtrl::GROUP_START_EXPOSURE_AND_TONE));
    totalParams++;

    // Exposure float slider
    AEFX_INIT_PARAM_STRUCTURE(def, flags, ui_flags);
    PF_ADD_FLOAT_SLIDERX(
        itemNames[6],
        ExposureStopsMin,
        ExposureStopsMax,
        ExposureStopsMin,
        ExposureStopsMax,
        ExposureStopsDef,
        PF_Precision_TENTHS,
        0,
        0,
        UnderlyingType(FilmSimulationCtrl::EXPOSURE));
    totalParams++;

    // Exposure time exponential slider
    AEFX_INIT_PARAM_STRUCTURE(def, flags, ui_flags);
    PF_ADD_FLOAT_EXPONENTIAL_SLIDER(
        itemNames[7],
        ExposureTimeSOff,
        ExposureTimeSMax,
        ExposureTimeSOff,
        ExposureTimeSMax,
        AEFX_AUDIO_DEFAULT_CURVE_TOLERANCE,
        ExposureTimeSDef,
        PF_Precision_TENTHS,
        0,
        0,
        2.5,
        UnderlyingType(FilmSimulationCtrl::EXPOSURE_TIME));
    totalParams++;

    // Mid-Gray target float slider
    AEFX_INIT_PARAM_STRUCTURE(def, flags, ui_flags);
    PF_ADD_FLOAT_SLIDERX(
        itemNames[8],
        GreyTargetMin,
        GreyTargetMax,
        GreyTargetMin,
        GreyTargetMax,
        GreyTargetDef,
        PF_Precision_HUNDREDTHS,
        0,
        0,
        UnderlyingType(FilmSimulationCtrl::MID_GRAY_TARGET));
    totalParams++;

    // Black Point Stratch float slider
    AEFX_INIT_PARAM_STRUCTURE(def, flags, ui_flags);
    PF_ADD_FLOAT_SLIDERX(
        itemNames[9],
        BlackPointStretchMin,
        BlackPointStretchMax,
        BlackPointStretchMin,
        BlackPointStretchMax,
        BlackPointStretchDef,
        PF_Precision_HUNDREDTHS,
        0,
        0,
        UnderlyingType(FilmSimulationCtrl::BLACK_POINT_STRETCH));
    totalParams++;

    // Close group item
    AEFX_INIT_PARAM_STRUCTURE(def, flags, ui_flags);
    PF_END_TOPIC(UnderlyingType(FilmSimulationCtrl::GROUP_STOP_EXPOSURE_AND_TONE));
    totalParams++;

    ///////////////////////////////////////////////////////////////////
    // GROUP START: DEVELOPMENT                                      //
    ///////////////////////////////////////////////////////////////////
    // Open group item
    AEFX_INIT_PARAM_STRUCTURE(def, flags, ui_flags);
    PF_ADD_TOPICX(
        itemNames[10],
        ui_disabled_flags,
        UnderlyingType(FilmSimulationCtrl::GROUP_START_DEVELOPMENT));
    totalParams++;

    // Development enabled check-box
    AEFX_INIT_PARAM_STRUCTURE(def, flags, ui_flags);
    PF_ADD_CHECKBOXX(
        itemNames[11],
        FALSE, 
        0, 
        UnderlyingType(FilmSimulationCtrl::GROUP_START_DEVELOPMENT));
    totalParams++;

    // Development time slider
    AEFX_INIT_PARAM_STRUCTURE(def, flags, ui_disabled_flags);
    PF_ADD_FLOAT_SLIDERX(
        itemNames[12],
        DevelopmentMinutesMin,
        DevelopmentMinutesMax,
        DevelopmentMinutesMin,
        DevelopmentMinutesMax,
        DevelopmentMinutesMin,
        PF_Precision_TENTHS,
        0,
        0,
        UnderlyingType(FilmSimulationCtrl::DEVELOPMENT_TIME));
    totalParams++;

    // Development temperature slider
    AEFX_INIT_PARAM_STRUCTURE(def, flags, ui_disabled_flags);
    PF_ADD_FLOAT_SLIDERX(
        itemNames[13],
        DevelopmentCelsiusMin,
        DevelopmentCelsiusMax,
        DevelopmentCelsiusMin,
        DevelopmentCelsiusMax,
        DevelopmentCelsiusDef,
        PF_Precision_TENTHS,
        0,
        0,
        UnderlyingType(FilmSimulationCtrl::DEVELOPMENT_TEMPERATURE));
    totalParams++;

    // Close group item
    AEFX_INIT_PARAM_STRUCTURE(def, flags, ui_flags);
    PF_END_TOPIC(UnderlyingType(FilmSimulationCtrl::GROUP_STOP_DEVELOPMENT));
    totalParams++;

    out_data->num_params = totalParams;

    return PF_Err_NONE;
}