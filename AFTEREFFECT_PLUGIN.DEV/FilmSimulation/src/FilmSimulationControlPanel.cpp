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

    // Process Variant ListBox
    AEFX_INIT_PARAM_STRUCTURE(def, flags, ui_flags);
    PF_ADD_POPUP(
        itemNames[4],
        ProcessVariantCtrlCount,
        UnderlyingType(ProcessVariantCtrl::eAS_SHIPPED),
        ProcessVariantCtrlStr,
        UnderlyingType(FilmSimulationCtrl::PROCESS_VARIANT));
    totalParams++;

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
    // GROUP START: DEVELOPMENT & STORAGE                           //
    ///////////////////////////////////////////////////////////////////
    // Open group item
    AEFX_INIT_PARAM_STRUCTURE(def, flags, ui_disabled_flags);
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
        UnderlyingType(FilmSimulationCtrl::DEVELOPMENT_ENABLE));
    totalParams++;

    // Development time float slider
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

    // Development temperature float slider
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

    // Development & Storage slider
    PF_ADD_SLIDER(
        itemNames[14], 
        StorageYearsMin,
        StorageYearsMax,
        StorageYearsMin,
        StorageYearsMax,
        StorageYearsDef,
        UnderlyingType(FilmSimulationCtrl::YEARS_OF_DARK_STORAGE));

    // Close group item
    AEFX_INIT_PARAM_STRUCTURE(def, flags, ui_flags);
    PF_END_TOPIC(UnderlyingType(FilmSimulationCtrl::GROUP_STOP_DEVELOPMENT));
    totalParams++;

    ///////////////////////////////////////////////////////////////////
    // GROUP START: DEVELOPMENT & STORAGE                           //
    ///////////////////////////////////////////////////////////////////
    AEFX_INIT_PARAM_STRUCTURE(def, flags, ui_disabled_flags);
    PF_ADD_TOPICX(
        itemNames[15],
        ui_flags,
        UnderlyingType(FilmSimulationCtrl::GROUP_START_COLOR_WHITE_BALANCE));
    totalParams++;

    // Scene Colour Temperature slider
    AEFX_INIT_PARAM_STRUCTURE(def, flags, ui_flags);
    PF_ADD_SLIDER(
        itemNames[16],
        SceneKelvinMin,
        SceneKelvinMax,
        SceneKelvinMin,
        SceneKelvinMax,
        SceneKelvinDef,
        UnderlyingType(FilmSimulationCtrl::SCENE_COLOUR_TEMPERATURE));
    totalParams++;

    // White Balance Strength float slider
    PF_ADD_FLOAT_SLIDERX(
        itemNames[17],
        WbStrengthMin,
        WbStrengthMax,
        WbStrengthMin,
        WbStrengthMax,
        WbStrengthDef,
        PF_Precision_HUNDREDTHS,
        0,
        0,
        UnderlyingType(FilmSimulationCtrl::WHITE_BALANCE_STRENGTH));
    totalParams++;

    // Close group item
    AEFX_INIT_PARAM_STRUCTURE(def, flags, ui_flags);
    PF_END_TOPIC(UnderlyingType(FilmSimulationCtrl::GROUP_STOP_COLOR_WHITE_BALANCE));
    totalParams++;

    ///////////////////////////////////////////////////////////////////
    // GROUP START: PRINT & DUPLICATION                              //
    ///////////////////////////////////////////////////////////////////
    AEFX_INIT_PARAM_STRUCTURE(def, flags, ui_disabled_flags);
    PF_ADD_TOPICX(
        itemNames[18],
        ui_flags,
        UnderlyingType(FilmSimulationCtrl::GROUP_START_PRINT_AND_DUPLICATION));
    totalParams++;

    // Print Stock List Box
    PF_ADD_POPUP(
        itemNames[19],
        UnderlyingType(PrintStockCtrl::ePRINT_STOCK_TOTAL),
        UnderlyingType(PrintStockCtrl::eSTOCKS_OWN),
        PrintStockCtrlStr,
        UnderlyingType(FilmSimulationCtrl::PRINT_STOCK));
    totalParams++;

    // Duplication generation slider
    AEFX_INIT_PARAM_STRUCTURE(def, flags, ui_flags);
    PF_ADD_SLIDER(
        itemNames[20],
        GenerationsMin,
        GenerationsMax,
        GenerationsMin,
        GenerationsMax,
        GenerationsDef,
        UnderlyingType(FilmSimulationCtrl::DUPLICATION_GENERATION));
    totalParams++;

    // Intermediate stock List Box
    PF_ADD_POPUP(
        itemNames[21],
        UnderlyingType(DupeStockCtrl::ePRINT_STOCK_TOTAL),
        UnderlyingType(DupeStockCtrl::eDUPE_FINE_GRAIN) + 1,
        DupeStockCtrlStr,
        UnderlyingType(FilmSimulationCtrl::INTERMEDIATE_STOCK));
    totalParams++;

    // Print Grain check-box
    AEFX_INIT_PARAM_STRUCTURE(def, flags, ui_flags);
    PF_ADD_CHECKBOXX(
        itemNames[22],
        FALSE,
        0,
        UnderlyingType(FilmSimulationCtrl::PRINT_GRAIN));
    totalParams++;

    // Close group item
    AEFX_INIT_PARAM_STRUCTURE(def, flags, ui_flags);
    PF_END_TOPIC(UnderlyingType(FilmSimulationCtrl::GROUP_STOP_PRINT_AND_DUPLICATION));
    totalParams++;

    ///////////////////////////////////////////////////////////////////
    // GROUP START: EMULSION CHARACTER                               //
    ///////////////////////////////////////////////////////////////////
    AEFX_INIT_PARAM_STRUCTURE(def, flags, ui_disabled_flags);
    PF_ADD_TOPICX(
        itemNames[23],
        ui_flags,
        UnderlyingType(FilmSimulationCtrl::GROUP_START_EMULSION_CHARACTER));
    totalParams++;

    // Grain float slider
    AEFX_INIT_PARAM_STRUCTURE(def, flags, ui_disabled_flags);
    PF_ADD_FLOAT_SLIDERX(
        itemNames[24],
        GrainScaleMin,
        GrainScaleMax,
        GrainScaleMin,
        GrainScaleMax,
        GrainScaleDef,
        PF_Precision_HUNDREDTHS,
        0,
        0,
        UnderlyingType(FilmSimulationCtrl::GRAIN));
    totalParams++;

    // Halation float slider
    AEFX_INIT_PARAM_STRUCTURE(def, flags, ui_disabled_flags);
    PF_ADD_FLOAT_SLIDERX(
        itemNames[25],
        HalationScaleMin,
        HalationScaleMax,
        HalationScaleMin,
        HalationScaleMax,
        HalationScaleDef,
        PF_Precision_HUNDREDTHS,
        0,
        0,
        UnderlyingType(FilmSimulationCtrl::HALATION));
    totalParams++;

    // Halation float slider
    AEFX_INIT_PARAM_STRUCTURE(def, flags, ui_disabled_flags);
    PF_ADD_FLOAT_SLIDERX(
        itemNames[26],
        CouplerScaleMin,
        CouplerScaleMax,
        CouplerScaleMin,
        CouplerScaleMax,
        CouplerScaleDef,
        PF_Precision_HUNDREDTHS,
        0,
        0,
        UnderlyingType(FilmSimulationCtrl::DIR_COUPLERS));
    totalParams++;

    // Misregistration float slider
    AEFX_INIT_PARAM_STRUCTURE(def, flags, ui_disabled_flags);
    PF_ADD_FLOAT_SLIDERX(
        itemNames[27],
        MisregScaleMin,
        MisregScaleMax,
        MisregScaleMin,
        MisregScaleMax,
        MisregScaleDef,
        PF_Precision_HUNDREDTHS,
        0,
        0,
        UnderlyingType(FilmSimulationCtrl::MISREGISTRATION));
    totalParams++;

    // Coating Unevenness float slider
    AEFX_INIT_PARAM_STRUCTURE(def, flags, ui_disabled_flags);
    PF_ADD_FLOAT_SLIDERX(
        itemNames[28],
        CoatingScaleMin,
        CoatingScaleMax,
        CoatingScaleMin,
        CoatingScaleMax,
        CoatingScaleDef,
        PF_Precision_HUNDREDTHS,
        0,
        0,
        UnderlyingType(FilmSimulationCtrl::COATING_UNEVENNESS));
    totalParams++;

    // Reseau Reconstruction check-box
    AEFX_INIT_PARAM_STRUCTURE(def, flags, ui_disabled_flags);
    PF_ADD_CHECKBOXX(
        itemNames[29],
        TRUE,
        0,
        UnderlyingType(FilmSimulationCtrl::RESEAU_RECONSTRUCTION));
    totalParams++;

    // Close group item
    AEFX_INIT_PARAM_STRUCTURE(def, flags, ui_flags);
    PF_END_TOPIC(UnderlyingType(FilmSimulationCtrl::GROUP_STOP_EMULSION_CHARACTER));
    totalParams++;


    // Assign totalnumber of control items
    out_data->num_params = totalParams;

    return PF_Err_NONE;
}