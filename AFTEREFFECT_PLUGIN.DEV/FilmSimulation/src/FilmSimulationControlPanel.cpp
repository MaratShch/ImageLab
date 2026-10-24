#include "Common.hpp"
#include "AlgoControlEnums.hpp"
#include "AlgoAdobeControlEnums.hpp"
#include "FilmSimulation.hpp"
#include "film_enum.hpp"
#include "AlgoControlValues.hpp"

CACHE_ALIGN constexpr char filmList[] = {
    #include "film_names.txt"
};



PF_Err SetupControlElements (PF_InData* in_data, PF_OutData* out_data)
{
    CACHE_ALIGN PF_ParamDef	def;

    constexpr PF_ParamFlags   flags = PF_ParamFlag_SUPERVISE | PF_ParamFlag_CANNOT_TIME_VARY | PF_ParamFlag_CANNOT_INTERP;
    constexpr PF_ParamUIFlags ui_flags = PF_PUI_CONTROL;
    constexpr PF_ParamUIFlags ui_disabled_flags = ui_flags | PF_PUI_DISABLED;

    constexpr char Button1Title[] = "ImageLab2 Interface";
    constexpr char Button2Title[] = "Lut Generator";

    A_long totalParams = 0;

    // Setup Button
    AEFX_INIT_PARAM_STRUCTURE(def, flags, ui_flags);
    PF_ADD_BUTTON(
        itemNames[0],
        Button1Title,
        ui_flags,
        flags,
        UnderlyingType(FilmSimulationCtrl::SETUP_BUTTON)
    );
    totalParams++;

    // Setup Button
    AEFX_INIT_PARAM_STRUCTURE(def, flags, ui_flags);
    PF_ADD_BUTTON(
        itemNames[1],
        Button2Title,
        ui_flags,
        flags,
        UnderlyingType(FilmSimulationCtrl::LUT_GENERATE)
    );
    totalParams++;

    ///////////////////////////////////////////////////////////////////
    // GROUP START: FILM STOCK                                       //
    ///////////////////////////////////////////////////////////////////
    // Open group item
    AEFX_INIT_PARAM_STRUCTURE(def, flags, ui_flags);
    PF_ADD_TOPICX(
        itemNames[2],
        ui_disabled_flags,
        UnderlyingType(FilmSimulationCtrl::GROUP_START_FILM_PROPERTIES));
    totalParams++;

    // Films LixtBox
    AEFX_INIT_PARAM_STRUCTURE(def, flags, ui_flags);
    PF_ADD_POPUP(
        itemNames[3],
        UnderlyingType(film::eFILM_PROFILE::eTOTAL_FILMS_PROFILES),
        UnderlyingType(film::eFILM_PROFILE::eAGFA_APX_25) + 1,      // AE pop-ups are 1-based
        filmList,
        UnderlyingType(FilmSimulationCtrl::FILM_STOCK));
    totalParams++;

    // Film Format ListBox
    AEFX_INIT_PARAM_STRUCTURE(def, flags, ui_flags);
    PF_ADD_POPUP(
        itemNames[4],
        UnderlyingType(FilmFormatCtrl::eFILM_FORMAT_TOTAL_FORMATS),
        UnderlyingType(FilmFormatCtrl::eFILM_FORMAT_FF_35) + 1,
        FilmFormatCtrlStr,
        UnderlyingType(FilmSimulationCtrl::FILM_FORMAT));
    totalParams++;

    // Process Variant ListBox
    AEFX_INIT_PARAM_STRUCTURE(def, flags, ui_flags);
    PF_ADD_POPUP(
        itemNames[5],
        ProcessVariantCtrlCount,
        UnderlyingType(ProcessVariantCtrl::eAS_SHIPPED) + 1,        // AE pop-ups are 1-based
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
    AEFX_INIT_PARAM_STRUCTURE(def, flags, ui_disabled_flags);
    PF_ADD_TOPICX(
        itemNames[6],
        ui_disabled_flags,
        UnderlyingType(FilmSimulationCtrl::GROUP_START_EXPOSURE_AND_TONE));
    totalParams++;

    // Exposure float slider
    AEFX_INIT_PARAM_STRUCTURE(def, flags, ui_flags);
    PF_ADD_FLOAT_SLIDERX(
        itemNames[7],
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
        itemNames[8],
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
        itemNames[9],
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
        itemNames[10],
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
    AEFX_INIT_PARAM_STRUCTURE(def, flags, ui_disabled_flags);
    PF_END_TOPIC(UnderlyingType(FilmSimulationCtrl::GROUP_STOP_EXPOSURE_AND_TONE));
    totalParams++;

    ///////////////////////////////////////////////////////////////////
    // GROUP START: DEVELOPMENT & STORAGE                           //
    ///////////////////////////////////////////////////////////////////
    // Open group item
    AEFX_INIT_PARAM_STRUCTURE(def, flags, ui_disabled_flags);
    PF_ADD_TOPICX(
        itemNames[11],
        ui_disabled_flags,
        UnderlyingType(FilmSimulationCtrl::GROUP_START_DEVELOPMENT));
    totalParams++;

    // Development enabled check-box
    AEFX_INIT_PARAM_STRUCTURE(def, flags, ui_flags);
    PF_ADD_CHECKBOXX(
        itemNames[12],
        DevelopmentEnableUiDef ? TRUE : FALSE,
        0, 
        UnderlyingType(FilmSimulationCtrl::DEVELOPMENT_ENABLE));
    totalParams++;

    // Development time float slider
    AEFX_INIT_PARAM_STRUCTURE(def, flags, ui_disabled_flags);
    PF_ADD_FLOAT_SLIDERX(
        itemNames[13],
        DevelopmentMinutesMin,
        DevelopmentMinutesMax,
        DevelopmentMinutesMin,
        DevelopmentMinutesMax,
        DevelopmentMinutesUiDef,    // engine default is the -1 sentinel; Development Enable selects it
        PF_Precision_TENTHS,
        0,
        0,
        UnderlyingType(FilmSimulationCtrl::DEVELOPMENT_TIME));
    totalParams++;

    // Development temperature float slider
    AEFX_INIT_PARAM_STRUCTURE(def, flags, ui_disabled_flags);
    PF_ADD_FLOAT_SLIDERX(
        itemNames[14],
        DevelopmentCelsiusMin,
        DevelopmentCelsiusMax,
        DevelopmentCelsiusMin,
        DevelopmentCelsiusMax,
        DevelopmentCelsiusUiDef,    // engine default is the -1 sentinel; Development Enable selects it
        PF_Precision_TENTHS,
        0,
        0,
        UnderlyingType(FilmSimulationCtrl::DEVELOPMENT_TEMPERATURE));
    totalParams++;

    // Developer ListBox
    AEFX_INIT_PARAM_STRUCTURE(def, flags, ui_flags);
    PF_ADD_POPUP(
        itemNames[15],
        DeveloperCtrlCount,
        UnderlyingType(DeveloperCtrl::eAS_SHIPPED) + 1,             // AE pop-ups are 1-based
        DeveloperCtrlStr,
        UnderlyingType(FilmSimulationCtrl::DEVELOPER_INDEX));
    totalParams++;

    // Development & Storage slider
    PF_ADD_SLIDER(
        itemNames[16], 
        StorageYearsMin,
        StorageYearsMax,
        StorageYearsMin,
        StorageYearsMax,
        StorageYearsDef,
        UnderlyingType(FilmSimulationCtrl::YEARS_OF_DARK_STORAGE));
    totalParams++;

    // Storage Temperature
    PF_ADD_SLIDER(
        itemNames[17],
        StorageCelsiusMin,
        StorageCelsiusMax,
        StorageCelsiusMin,
        StorageCelsiusMax,
        StorageCelsiusDef,
        UnderlyingType(FilmSimulationCtrl::STORAGE_TEMPERATURE));
    totalParams++;

    // Batch Position float slider
    AEFX_INIT_PARAM_STRUCTURE(def, flags, ui_flags);
    PF_ADD_FLOAT_SLIDERX(
        itemNames[18],
        BatchPositionMin,
        BatchPositionMax,
        BatchPositionMin,
        BatchPositionMax,
        BatchPositionDef,
        PF_Precision_HUNDREDTHS,
        0,
        0,
        UnderlyingType(FilmSimulationCtrl::BATCH_POSITION));
    totalParams++;

    // Close group item
    AEFX_INIT_PARAM_STRUCTURE(def, flags, ui_disabled_flags);
    PF_END_TOPIC(UnderlyingType(FilmSimulationCtrl::GROUP_STOP_DEVELOPMENT));
    totalParams++;

    ///////////////////////////////////////////////////////////////////
    // GROUP START: DEVELOPMENT & STORAGE                           //
    ///////////////////////////////////////////////////////////////////
    AEFX_INIT_PARAM_STRUCTURE(def, flags, ui_disabled_flags);
    PF_ADD_TOPICX(
        itemNames[19],
        ui_flags,
        UnderlyingType(FilmSimulationCtrl::GROUP_START_COLOR_WHITE_BALANCE));
    totalParams++;

    // Scene Colour Temperature slider
    AEFX_INIT_PARAM_STRUCTURE(def, flags, ui_flags);
    PF_ADD_SLIDER(
        itemNames[20],
        SceneKelvinMin,
        SceneKelvinMax,
        SceneKelvinMin,
        SceneKelvinMax,
        SceneKelvinDef,
        UnderlyingType(FilmSimulationCtrl::SCENE_COLOUR_TEMPERATURE));
    totalParams++;

    // White Balance Strength float slider
    PF_ADD_FLOAT_SLIDERX(
        itemNames[21],
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
    AEFX_INIT_PARAM_STRUCTURE(def, flags, ui_disabled_flags);
    PF_END_TOPIC(UnderlyingType(FilmSimulationCtrl::GROUP_STOP_COLOR_WHITE_BALANCE));
    totalParams++;

    ///////////////////////////////////////////////////////////////////
    // GROUP START: PRINT & DUPLICATION                              //
    ///////////////////////////////////////////////////////////////////
    AEFX_INIT_PARAM_STRUCTURE(def, flags, ui_disabled_flags);
    PF_ADD_TOPICX(
        itemNames[22],
        ui_flags,
        UnderlyingType(FilmSimulationCtrl::GROUP_START_PRINT_AND_DUPLICATION));
    totalParams++;

    // Print Stock List Box
    PF_ADD_POPUP(
        itemNames[23],
        UnderlyingType(PrintStockCtrl::ePRINT_STOCK_TOTAL),
        UnderlyingType(PrintStockCtrl::eSTOCKS_OWN) + 1,            // AE pop-ups are 1-based
        PrintStockCtrlStr,
        UnderlyingType(FilmSimulationCtrl::PRINT_STOCK));
    totalParams++;

    // Duplication generation slider
    AEFX_INIT_PARAM_STRUCTURE(def, flags, ui_flags);
    PF_ADD_SLIDER(
        itemNames[24],
        GenerationsMin,
        GenerationsMax,
        GenerationsMin,
        GenerationsMax,
        GenerationsDef,
        UnderlyingType(FilmSimulationCtrl::DUPLICATION_GENERATION));
    totalParams++;

    // Intermediate stock List Box
    PF_ADD_POPUP(
        itemNames[25],
        UnderlyingType(DupeStockCtrl::ePRINT_STOCK_TOTAL),
        UnderlyingType(DupeStockCtrl::eDUPE_FINE_GRAIN) + 1,
        DupeStockCtrlStr,
        UnderlyingType(FilmSimulationCtrl::INTERMEDIATE_STOCK));
    totalParams++;

    // Print Grain check-box
    AEFX_INIT_PARAM_STRUCTURE(def, flags, ui_flags);
    PF_ADD_CHECKBOXX(
        itemNames[26],
        PrintGrainDef ? TRUE : FALSE,
        0,
        UnderlyingType(FilmSimulationCtrl::PRINT_GRAIN));
    totalParams++;

    // Close group item
    AEFX_INIT_PARAM_STRUCTURE(def, flags, ui_disabled_flags);
    PF_END_TOPIC(UnderlyingType(FilmSimulationCtrl::GROUP_STOP_PRINT_AND_DUPLICATION));
    totalParams++;

    ///////////////////////////////////////////////////////////////////
    // GROUP START: EMULSION CHARACTER                               //
    ///////////////////////////////////////////////////////////////////
    AEFX_INIT_PARAM_STRUCTURE(def, flags, ui_disabled_flags);
    PF_ADD_TOPICX(
        itemNames[27],
        ui_flags,
        UnderlyingType(FilmSimulationCtrl::GROUP_START_EMULSION_CHARACTER));
    totalParams++;

    // Grain float slider
    AEFX_INIT_PARAM_STRUCTURE(def, flags, ui_flags);
    PF_ADD_FLOAT_SLIDERX(
        itemNames[28],
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
    AEFX_INIT_PARAM_STRUCTURE(def, flags, ui_flags);
    PF_ADD_FLOAT_SLIDERX(
        itemNames[29],
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

    // DIR Couplers float slider
    AEFX_INIT_PARAM_STRUCTURE(def, flags, ui_flags);
    PF_ADD_FLOAT_SLIDERX(
        itemNames[30],
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
    AEFX_INIT_PARAM_STRUCTURE(def, flags, ui_flags);
    PF_ADD_FLOAT_SLIDERX(
        itemNames[31],
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
    AEFX_INIT_PARAM_STRUCTURE(def, flags, ui_flags);
    PF_ADD_FLOAT_SLIDERX(
        itemNames[32],
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
    AEFX_INIT_PARAM_STRUCTURE(def, flags, ui_flags);
    PF_ADD_CHECKBOXX(
        itemNames[33],
        TRUE,
        0,
        UnderlyingType(FilmSimulationCtrl::RESEAU_RECONSTRUCTION));
    totalParams++;

    // Close group item
    AEFX_INIT_PARAM_STRUCTURE(def, flags, ui_disabled_flags);
    PF_END_TOPIC(UnderlyingType(FilmSimulationCtrl::GROUP_STOP_EMULSION_CHARACTER));
    totalParams++;

    ///////////////////////////////////////////////////////////////////
    // GROUP START: LENS & READER                                    //
    ///////////////////////////////////////////////////////////////////
    AEFX_INIT_PARAM_STRUCTURE(def, flags, ui_disabled_flags);
    PF_ADD_TOPICX(
        itemNames[34],
        ui_flags,
        UnderlyingType(FilmSimulationCtrl::GROUP_START_LENS_AND_READER));
    totalParams++;

    // Veiling Flare: stock's own check-box (engine sentinel -1)
    AEFX_INIT_PARAM_STRUCTURE(def, flags, ui_flags);
    PF_ADD_CHECKBOXX(
        itemNames[63],
        FlareStockOwnUiDef ? TRUE : FALSE,
        0,
        UnderlyingType(FilmSimulationCtrl::VEILING_FLARE_AUTO));
    totalParams++;

    // Veiling Flare float slider
    AEFX_INIT_PARAM_STRUCTURE(def, flags, ui_disabled_flags);
    PF_ADD_FLOAT_SLIDERX(
        itemNames[35],
        FlareMin,
        FlareMax,
        FlareMin,
        FlareMax,
        FlareUiDef,                 // used when "Stock's Own" is unticked
        PF_Precision_THOUSANDTHS,
        0,
        0,
        UnderlyingType(FilmSimulationCtrl::VEILING_FLARE));
    totalParams++;

    // Corner Falloff: stock's own check-box (engine sentinel -1)
    AEFX_INIT_PARAM_STRUCTURE(def, flags, ui_flags);
    PF_ADD_CHECKBOXX(
        itemNames[36],
        VignetteStockOwnUiDef ? TRUE : FALSE,
        0,
        UnderlyingType(FilmSimulationCtrl::CORNER_FALLOFF_AUTO));
    totalParams++;

    // Corner Falloff float slider
    AEFX_INIT_PARAM_STRUCTURE(def, flags, ui_disabled_flags);
    PF_ADD_FLOAT_SLIDERX(
        itemNames[37],
        VignetteMin,
        VignetteMax,
        VignetteMin,
        VignetteMax,
        VignetteUiDef,              // used when "Stock's Own" is unticked
        PF_Precision_TENTHS,
        0,
        0,
        UnderlyingType(FilmSimulationCtrl::CORNER_FALLOFF));
    totalParams++;

    // Scanner Specularity float slider
    AEFX_INIT_PARAM_STRUCTURE(def, flags, ui_disabled_flags);
    PF_ADD_FLOAT_SLIDERX(
        itemNames[38],
        ScannerSpecularMin,
        ScannerSpecularMax,
        ScannerSpecularMin,
        ScannerSpecularMax,
        ScannerSpecularDef,
        PF_Precision_THOUSANDTHS,
        0,
        0,
        UnderlyingType(FilmSimulationCtrl::SCANNER_SPECULARITY));
    totalParams++;

    // Scanner Specularity float slider
    AEFX_INIT_PARAM_STRUCTURE(def, flags, ui_disabled_flags);
    PF_ADD_FLOAT_SLIDERX(
        itemNames[39],
        ScannerFixedPatternMin,
        ScannerFixedPatternMax,
        ScannerFixedPatternMin,
        ScannerFixedPatternMax,
        ScannerFixedPatternDef,
        PF_Precision_THOUSANDTHS,
        0,
        0,
        UnderlyingType(FilmSimulationCtrl::SCANNER_FIXED_PATTERN));
    totalParams++;

    // Close group item
    AEFX_INIT_PARAM_STRUCTURE(def, flags, ui_disabled_flags);
    PF_END_TOPIC(UnderlyingType(FilmSimulationCtrl::GROUP_STOP_LENS_AND_READER));
    totalParams++;

    ///////////////////////////////////////////////////////////////////
    // GROUP START: FILM DAMAGE AND AGE                              //
    ///////////////////////////////////////////////////////////////////
    AEFX_INIT_PARAM_STRUCTURE(def, flags, ui_disabled_flags);
    PF_ADD_TOPICX(
        itemNames[40],
        ui_flags,
        UnderlyingType(FilmSimulationCtrl::GROUP_START_FILM_DAMAGE_AND_AGE));
    totalParams++;

    // Enable Film Damage check-box
    AEFX_INIT_PARAM_STRUCTURE(def, flags, ui_flags);
    PF_ADD_CHECKBOXX(
        itemNames[41],
        FALSE,
        0,
        UnderlyingType(FilmSimulationCtrl::ENABLE_FILM_DAMAGE));
    totalParams++;

    // Overall Strength float slider
    AEFX_INIT_PARAM_STRUCTURE(def, flags, ui_disabled_flags);
    PF_ADD_FLOAT_SLIDERX(
        itemNames[42],
        DamageStrengthMin,
        DamageStrengthMax,
        DamageStrengthMin,
        DamageStrengthMax,
        DamageStrengthDef,
        PF_Precision_HUNDREDTHS,
        0,
        0,
        UnderlyingType(FilmSimulationCtrl::OVERALL_STRENGHT));
    totalParams++;

    // Damage Seed button
    AEFX_INIT_PARAM_STRUCTURE(def, flags, ui_disabled_flags);
    //         itemNames[40],

    // Dust float slider
    AEFX_INIT_PARAM_STRUCTURE(def, flags, ui_disabled_flags);
    PF_ADD_FLOAT_SLIDERX(
        itemNames[43],
        DustLevelMin,
        DustLevelMax,
        DustLevelMin,
        DustLevelMax,
        DustLevelDef,
        PF_Precision_HUNDREDTHS,
        0,
        0,
        UnderlyingType(FilmSimulationCtrl::DUST));
    totalParams++;

    // Debris float slider
    AEFX_INIT_PARAM_STRUCTURE(def, flags, ui_disabled_flags);
    PF_ADD_FLOAT_SLIDERX(
        itemNames[44],
        DebrisLevelMin,
        DebrisLevelMax,
        DebrisLevelMin,
        DebrisLevelMax,
        DebrisLevelDef,
        PF_Precision_HUNDREDTHS,
        0,
        0,
        UnderlyingType(FilmSimulationCtrl::DEBRIS));
    totalParams++;

    // Fibres float slider
    AEFX_INIT_PARAM_STRUCTURE(def, flags, ui_disabled_flags);
    PF_ADD_FLOAT_SLIDERX(
        itemNames[45],
        FibreLevelMin,
        FibreLevelMax,
        FibreLevelMin,
        FibreLevelMax,
        FibreLevelDef,
        PF_Precision_HUNDREDTHS,
        0,
        0,
        UnderlyingType(FilmSimulationCtrl::FIBRES));
    totalParams++;

    // Clumping float slider
    AEFX_INIT_PARAM_STRUCTURE(def, flags, ui_disabled_flags);
    PF_ADD_FLOAT_SLIDERX(
        itemNames[46],
        DirtClumpingMin,
        DirtClumpingMax,
        DirtClumpingMin,
        DirtClumpingMax,
        DirtClumpingDef,
        PF_Precision_HUNDREDTHS,
        0,
        0,
        UnderlyingType(FilmSimulationCtrl::CLUMPING));
    totalParams++;

    // Gate Dirt float slider
    AEFX_INIT_PARAM_STRUCTURE(def, flags, ui_disabled_flags);
    PF_ADD_FLOAT_SLIDERX(
        itemNames[47],
        GateDirtMin,
        GateDirtMax,
        GateDirtMin,
        GateDirtMax,
        GateDirtDef,
        PF_Precision_HUNDREDTHS,
        0,
        0,
        UnderlyingType(FilmSimulationCtrl::GATE_DIRT));
    totalParams++;

    // Gate Weave float slider
    AEFX_INIT_PARAM_STRUCTURE(def, flags, ui_disabled_flags);
    PF_ADD_FLOAT_SLIDERX(
        itemNames[48],
        WeaveAmountMin,
        WeaveAmountMax,
        WeaveAmountMin,
        WeaveAmountMax,
        WeaveAmountDef,
        PF_Precision_HUNDREDTHS,
        0,
        0,
        UnderlyingType(FilmSimulationCtrl::GATE_WAVE));
    totalParams++;

    // Splice & Tear Events float slider
    AEFX_INIT_PARAM_STRUCTURE(def, flags, ui_disabled_flags);
    PF_ADD_FLOAT_SLIDERX(
        itemNames[49],
        SpliceAndTearsEventstMin,
        SpliceAndTearsEventstMax,
        SpliceAndTearsEventstMin,
        SpliceAndTearsEventstMax,
        SpliceAndTearsEventstDef,
        PF_Precision_HUNDREDTHS,
        0,
        0,
        UnderlyingType(FilmSimulationCtrl::SPLICE_AND_TEAR_EVENTS));
    totalParams++;

    // Transport Scratches float slider
    AEFX_INIT_PARAM_STRUCTURE(def, flags, ui_disabled_flags);
    PF_ADD_FLOAT_SLIDERX(
        itemNames[50],
        ScratchTransportMin,
        ScratchTransportMax,
        ScratchTransportMin,
        ScratchTransportMax,
        ScratchTransportDef,
        PF_Precision_HUNDREDTHS,
        0,
        0,
        UnderlyingType(FilmSimulationCtrl::TRANSPORT_SCRATCHES));
    totalParams++;

    // Handling Scratches float slider
    AEFX_INIT_PARAM_STRUCTURE(def, flags, ui_disabled_flags);
    PF_ADD_FLOAT_SLIDERX(
        itemNames[51],
        ScratchHandlingMin,
        ScratchHandlingMax,
        ScratchHandlingMin,
        ScratchHandlingMax,
        ScratchHandlingDef,
        PF_Precision_HUNDREDTHS,
        0,
        0,
        UnderlyingType(FilmSimulationCtrl::HANDLING_SCRATCHES));
    totalParams++;

    // Processing Quality float slider
    AEFX_INIT_PARAM_STRUCTURE(def, flags, ui_disabled_flags);
    PF_ADD_FLOAT_SLIDERX(
        itemNames[52],
        ProcessingQualityMin,
        ProcessingQualityMax,
        ProcessingQualityMin,
        ProcessingQualityMax,
        ProcessingQualityDef,
        PF_Precision_HUNDREDTHS,
        0,
        0,
        UnderlyingType(FilmSimulationCtrl::PROCESSING_QUALITY));
    totalParams++;

    // Drying Marks float slider
    AEFX_INIT_PARAM_STRUCTURE(def, flags, ui_disabled_flags);
    PF_ADD_FLOAT_SLIDERX(
        itemNames[53],
        DryingMarksMin,
        DryingMarksMax,
        DryingMarksMin,
        DryingMarksMax,
        DryingMarksDef,
        PF_Precision_HUNDREDTHS,
        0,
        0,
        UnderlyingType(FilmSimulationCtrl::DRYING_MARKS));
    totalParams++;

    // Storage Severity float slider
    AEFX_INIT_PARAM_STRUCTURE(def, flags, ui_disabled_flags);
    PF_ADD_FLOAT_SLIDERX(
        itemNames[54],
        StorageSeverityMin,
        StorageSeverityMax,
        StorageSeverityMin,
        StorageSeverityMax,
        StorageSeverityDef,
        PF_Precision_HUNDREDTHS,
        0,
        0,
        UnderlyingType(FilmSimulationCtrl::STORAGE_SEVERITY));
    totalParams++;

    // Colour Veil float slider
    AEFX_INIT_PARAM_STRUCTURE(def, flags, ui_disabled_flags);
    PF_ADD_FLOAT_SLIDERX(
        itemNames[55],
        ColourVeilMin,
        ColourVeilMax,
        ColourVeilMin,
        ColourVeilMax,
        ColourVeilDef,
        PF_Precision_HUNDREDTHS,
        0,
        0,
        UnderlyingType(FilmSimulationCtrl::COLOUR_VEIL));
    totalParams++;

    // Printer Flicker float slider
    AEFX_INIT_PARAM_STRUCTURE(def, flags, ui_disabled_flags);
    PF_ADD_FLOAT_SLIDERX(
        itemNames[56],
        FlickerStopsMin,
        FlickerStopsMax,
        FlickerStopsMin,
        FlickerStopsMax,
        FlickerStopsDef,
        PF_Precision_HUNDREDTHS,
        0,
        0,
        UnderlyingType(FilmSimulationCtrl::PRINTER_FLICKER));
    totalParams++;

    // Scanner Artifacts float slider
    AEFX_INIT_PARAM_STRUCTURE(def, flags, ui_disabled_flags);
    PF_ADD_FLOAT_SLIDERX(
        itemNames[57],
        ScannerArtifactsMin,
        ScannerArtifactsMax,
        ScannerArtifactsMin,
        ScannerArtifactsMax,
        ScannerArtifactsDef,
        PF_Precision_HUNDREDTHS,
        0,
        0,
        UnderlyingType(FilmSimulationCtrl::SCANNER_ARTIFACTS));
    totalParams++;

    // Close group item
    AEFX_INIT_PARAM_STRUCTURE(def, flags, ui_disabled_flags);
    PF_END_TOPIC(UnderlyingType(FilmSimulationCtrl::GROUP_STOP_FILM_DAMAGE_AND_AGE));
    totalParams++;

    ///////////////////////////////////////////////////////////////////
    // GROUP START: SPECTRAL AND TEMPORAL                            //
    ///////////////////////////////////////////////////////////////////
    AEFX_INIT_PARAM_STRUCTURE(def, flags, ui_disabled_flags);
    PF_ADD_TOPICX(
        itemNames[58],
        ui_flags,
        UnderlyingType(FilmSimulationCtrl::GROUP_START_SPECTRAL_AND_TEMPORAL));
    totalParams++;

    // Spectral taking Matrix
    AEFX_INIT_PARAM_STRUCTURE(def, flags, ui_flags);
    PF_ADD_CHECKBOXX(
        itemNames[59],
        FALSE,
        0,
        UnderlyingType(FilmSimulationCtrl::SPECTRAL_TAKING_MATRIX));
    totalParams++;

    // Grain Temporal Mode
    PF_ADD_POPUP(
        itemNames[60],
        UnderlyingType(GrainTemporalModeCtrl::eGRAIN_TEMPORAL_TOTAL),
        UnderlyingType(GrainTemporalModeCtrl::eGRAIN_TEMPORAL_STILL) + 1,   // AE pop-ups are 1-based
        GrainTemporalModeCtrlStr,
        UnderlyingType(FilmSimulationCtrl::GRAIN_TEMPORAL_MODE));
    totalParams++;

    // Spectral Upsampling
    AEFX_INIT_PARAM_STRUCTURE(def, flags, ui_flags);
    PF_ADD_CHECKBOXX(
        itemNames[61],
        FALSE,
        0,
        UnderlyingType(FilmSimulationCtrl::SPECTRAL_UPSAMPLING));
    totalParams++;

    // Close group item
    AEFX_INIT_PARAM_STRUCTURE(def, flags, ui_disabled_flags);
    PF_END_TOPIC(UnderlyingType(FilmSimulationCtrl::GROUP_STOP_SPECTRAL_AND_TEMPORAL));
    totalParams++;

    ///////////////////////////////////////////////////////////////////
    // GROUP START: STOP RAW STOCK                                   //
    ///////////////////////////////////////////////////////////////////
    AEFX_INIT_PARAM_STRUCTURE(def, flags, ui_disabled_flags);
    PF_ADD_TOPICX(
        itemNames[62],
        ui_flags,
        UnderlyingType(FilmSimulationCtrl::GROUP_START_RAW_STOCK));
    totalParams++;

    PF_ADD_FLOAT_SLIDERX(
        itemNames[63],
        FilmAgeMin,
        FilmAgeMax,
        FilmAgeMin,
        FilmAgeMax,
        FilmAgeDef,
        PF_Precision_INTEGER,
        0,
        0,
        UnderlyingType(FilmSimulationCtrl::FILM_AGE_EXPIRED_STOCK));
    totalParams++;

    AEFX_INIT_PARAM_STRUCTURE(def, flags, ui_disabled_flags);
    PF_END_TOPIC(UnderlyingType(FilmSimulationCtrl::GROUP_STOP_RAW_STOCK));
    totalParams++;

    // AE counts the input layer (param 0) in num_params; totalParams does not.
    totalParams++;

    // params[] is indexed by FilmSimulationCtrl, so the registration order above
    // must match the enumeration one for one. A control added to the enum and not
    // registered (as Batch Position was) shifts every later control by one.
    if (totalParams != static_cast<A_long>(UnderlyingType(FilmSimulationCtrl::TOTAL_PARAMS)))
        return PF_Err_INTERNAL_STRUCT_DAMAGED;

    // Set total number of control items
    out_data->num_params = totalParams;

    return PF_Err_NONE;
}