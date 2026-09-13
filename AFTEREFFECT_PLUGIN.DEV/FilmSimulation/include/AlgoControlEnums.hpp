#ifndef __IMAGE_LAB2_ALGO_CONTROL_ENUMERATORS__
#define __IMAGE_LAB2_ALGO_CONTROL_ENUMERATORS__

#include <cstdint>

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
// string representation for filmFormat in AlgoControl structure  [used for fill ListBox]
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

// printStock in AlgoControl structure
enum class PrintStockCtrl : int32_t
{
    STOCKS_OWN,
    SCAN_DI,
    KODAK_2383_RELEASE,
    KODAK_VISION3_DI_2254,
    EASTMANCOLOR_5382_1953,
    TECHNICOLOR_IB,
    DUPE_FINE_GRAIN,
    CP_1_POSITIVE,
    CP_3_POSITIVE,
    CP_6_POSITIVE,
    TASMA_POSITIVE_28
};
// string representation for printStock in AlgoControl structure  [used for fill ListBox]
constexpr char PrintStockCtrlStr[]
{
    "Stock own|"
    "Scan DI|"
    "Kodak 2383 Release|"
    "Kodak Vision3 DI 2254|"
    "Eastman Color 5382 (1953y)|"
    "Technicolor IB|"
    "Dupe Fine Grain|"
    "CP-1 Positive|"
    "CP-3 Positive|"
    "CP-6 Positive|"
    "Tasma Positive 2.8"
};

using DupeStockCtrl = PrintStockCtrl;
constexpr char DupeStockCtrlStr[]
{
    "Stock own|"
    "Scan DI|"
    "Kodak 2383 Release|"
    "Kodak Vision3 DI 2254|"
    "Eastman Color 5382 (1953y)|"
    "Technicolor IB|"
    "Dupe Fine Grain|"
    "CP-1 Positive|"
    "CP-3 Positive|"
    "CP-6 Positive|"
    "Tasma Positive 2.8" 
};

// processVariant in AlgoControl structure
enum class FilmProcessVariant : int32_t
{
    eFILM_PROCESS_EI_1600 = 0,
    eFILM_PROCESS_CS2_TWO_BATH_KIT,
    eFILM_KODAK_H24,
    eFILM_PROCESS_TOTAL
};
// string representation for processVariant in AlgoControl structure [used for fill ListBox]
constexpr char FilmProcessVariant[] =
{
    "EI 1600|"
    "CS2 TWO BATHS KIT|" 
    "KODAK H24"
};

// exposureStops in AlgoControl structures
constexpr double ExposureStopsMin = -4.0;
constexpr double ExposureStopsMax = 4.0;
constexpr double ExposureStopsDef = 0.0;
constexpr double ExposureStopsStep = 0.1;


#endif // __IMAGE_LAB2_ALGO_CONTROL_ENUMERATORS__