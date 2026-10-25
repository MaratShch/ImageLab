#ifndef __IMAGE_LAB2_ALGO_ADOBE_CONTROL_ENUMERATORS__
#define __IMAGE_LAB2_ALGO_ADOBE_CONTROL_ENUMERATORS__

#include "CompileTimeUtils.hpp"

enum class FilmSimulationCtrl : int32_t
{
    VIDEO_INPUT,
    SETUP_BUTTON, // button for run dialog control
    LUT_GENERATE, // button for generate LUT
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
    AIM_EXPOSURE,           // check-box: expose to the data sheet's grey-card aim (bit 57)
    GROUP_STOP_EXPOSURE_AND_TONE,
    GROUP_START_DEVELOPMENT,
    DEVELOPMENT_ENABLE,
    DEVELOPMENT_TIME,
    DEVELOPMENT_TEMPERATURE,
    DEVELOPER_INDEX,
    YEARS_OF_DARK_STORAGE,
    STORAGE_TEMPERATURE,
    BATCH_POSITION,
    GROUP_STOP_DEVELOPMENT,
    GROUP_START_COLOR_WHITE_BALANCE,
    SCENE_COLOUR_TEMPERATURE,
    WHITE_BALANCE_STRENGTH,
    GROUP_STOP_COLOR_WHITE_BALANCE,
    GROUP_START_PRINT_AND_DUPLICATION,
    PRINT_STOCK,
    DUPLICATION_GENERATION,
    INTERMEDIATE_STOCK,
    PRINT_GRAIN,
    GROUP_STOP_PRINT_AND_DUPLICATION,
    GROUP_START_EMULSION_CHARACTER,
    GRAIN,
    HALATION,
    DIR_COUPLERS,
    MISREGISTRATION,
    COATING_UNEVENNESS,
    RESEAU_RECONSTRUCTION,
    GROUP_STOP_EMULSION_CHARACTER,
    GROUP_START_LENS_AND_READER,
    VEILING_FLARE_AUTO,     // check-box: use the stock's own flare (engine sentinel -1)
    VEILING_FLARE,
    CORNER_FALLOFF_AUTO,    // check-box: use the stock's own falloff (engine sentinel -1)
    CORNER_FALLOFF,
    SCANNER_SPECULARITY,
    SCANNER_FIXED_PATTERN,
    GROUP_STOP_LENS_AND_READER,
    GROUP_START_FILM_DAMAGE_AND_AGE,
    ENABLE_FILM_DAMAGE,
    OVERALL_STRENGHT,
    DAMAGE_SEED,            // int slider: offset added to DamageSeedDef (bit 30)
    DUST,
    DEBRIS,
    FIBRES,
    CLUMPING,
    GATE_DIRT,
    GATE_WAVE,
    SPLICE_AND_TEAR_EVENTS,
    TRANSPORT_SCRATCHES,
    HANDLING_SCRATCHES,
    PROCESSING_QUALITY,
    DRYING_MARKS,
    STORAGE_SEVERITY,
    COLOUR_VEIL,
    PRINTER_FLICKER,
    SCANNER_ARTIFACTS,
    DEFECT_READER_OPTICS,   // check-box (bit 51)
    TRANSPORT_SCRATCH_LIFETIME, // seconds, 0 = stock's own (bit 52)
    PERSISTENT_DUST_LIFETIME,   // seconds (bit 53)
    DEFECT_VARIABILITY,     // 0..1 (bit 54)
    DEFECT_VISIBILITY,      // 0..3 (bit 55)
    GROUP_STOP_FILM_DAMAGE_AND_AGE,
    GROUP_START_SPECTRAL_AND_TEMPORAL,
    SPECTRAL_TAKING_MATRIX,
    GRAIN_TEMPORAL_MODE,
    GRAIN_SEED,             // int slider: offset added to SeedDef (bit 46)
    SPECTRAL_UPSAMPLING,
    GROUP_STOP_SPECTRAL_AND_TEMPORAL,
    GROUP_START_RAW_STOCK,
    FILM_AGE_EXPIRED_STOCK,
    GROUP_STOP_RAW_STOCK,
    TOTAL_PARAMS
};

// Control names, in the ORDER THE CONTROLS ARE ADDED in SetupControlElements():
// the N-th named control uses itemNames[N] (the comment is N). Group ends have
// no name and take no entry. Inserting a control = inserting its name at the
// same position here and renumbering the itemNames[] uses after it.
constexpr char itemNames[][32] = {
    " ",                            // 0
    " ",                            // 1
    "Film Stock",                   // 2
    "Film Stock",                   // 3
    "Film Format",                  // 4
    "Process Variant",              // 5
    "Exposure & Tone",              // 6
    "Exposure",                     // 7
    "Exposure Time",                // 8
    "Mid-Gray Target",              // 9
    "Black Point Stretch",          // 10
    "Aim Exposure",                 // 11
    "Development And Storage",      // 12
    "Development Enable",           // 13
    "Development Time",             // 14
    "Development Temperature",      // 15
    "Developer",                    // 16
    "Years of Dark Storage",        // 17
    "Storage Temperature",          // 18
    "Batch Position",               // 19
    "Colour & White Balance",       // 20
    "Scene Colour Temperature",     // 21
    "White Balance Strength",       // 22
    "Print & Duplication",          // 23
    "Print Stock",                  // 24
    "Duplication Generation",       // 25
    "Intermediate Stock",           // 26
    "Print Grain",                  // 27
    "Emulsion Character",           // 28
    "Grain",                        // 29
    "Halation",                     // 30
    "DIR Couplers",                 // 31
    "Misregistration",              // 32
    "Coating Unevenness",           // 33
    "Reseau Reconstruction",        // 34
    "Lens and Reader",              // 35
    "Veiling Flare: Stock's Own",   // 36
    "Veiling Flare",                // 37
    "Corner Falloff: Stock's Own",  // 38
    "Corner Falloff",               // 39
    "Scanner Specularity",          // 40
    "Scanner Fixed Pattern",        // 41
    "Film Damage and Age",          // 42
    "Enable Film Damage",           // 43
    "Overall Strength",             // 44
    "Damage Seed",                  // 45
    "Dust",                         // 46
    "Debris",                       // 47
    "Fibres",                       // 48
    "Clumping",                     // 49
    "Gate Dirt",                    // 50
    "Gate Wave",                    // 51
    "Splice and Tear Events",       // 52
    "Transport Scratches",          // 53
    "Handling Scratches",           // 54
    "Processing Quality",           // 55
    "Drying Marks",                 // 56
    "Storage Severity",             // 57
    "Colour Veil",                  // 58
    "Printer Flicker",              // 59
    "Scanner Artifacts",            // 60
    "Defect Reader Optics",         // 61
    "Transport Scratch Lifetime",   // 62
    "Persistent Dust Lifetime",     // 63
    "Defect Temporal Variability",  // 64
    "Defect Visibility",            // 65
    "Spectral and Temporal",        // 66
    "Spectral Taking Matrix",       // 67
    "Grain Temporal Mode",          // 68
    "Grain Seed",                   // 69
    "Spectral Upsampling",          // 70
    "Raw Stock",                    // 71
    "Film Age (expired stock)"      // 72
};

static_assert(sizeof(itemNames) / sizeof(itemNames[0]) == 73, "itemNames: one name per named control - update the count when a control is added");


#endif // __IMAGE_LAB2_ALGO_ADOBE_CONTROL_ENUMERATORS__