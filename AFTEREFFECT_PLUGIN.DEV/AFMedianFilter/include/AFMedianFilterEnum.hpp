#ifndef __IMAGE_LAB_ADAPTIVE_FREQUENCY_MEDIAN_FILTER_ENUMS__
#define __IMAGE_LAB_ADAPTIVE_FREQUENCY_MEDIAN_FILTER_ENUMS__

#include <cstdint>
#include "CompileTimeUtils.hpp"


enum class AFMF : int32_t
{
    eIMAGE_AFMEDIAN_INPUT,
    eIMAGE_AFMEDIAN_PARAM_RADIUS,
    eIMAGE_AFMEDIAN_PARAM_TOLERANCE,
    eIMAGE_AFMEDIAN_PARAM_ITERATIONS,
    eIMAGE_AFMEDIAN_TOTAL_CONTROLS
};

constexpr char AFMFControlsStr[][24] =
{
    "Window Size",
    "Filter Tolerance",
    "Iterations Number"
};

constexpr int32_t kerenlRadiusMin = 1;
constexpr int32_t kernelRadiusMax = 8;
constexpr int32_t kernelRadiusDef = kerenlRadiusMin;

enum class AFMF_RadiusSize : int32_t
{
    eIMAGE_AFMEDIAN_WINDOW_DISABLED = 0,
    eIMAGE_AFMEDIAN_WINDOW_3x3 = 1,
    eIMAGE_AFMEDIAN_WINDOW_5x5,
    eIMAGE_AFMEDIAN_WINDOW_7x7,
    eIMAGE_AFMEDIAN_WINDOW_9x9,
    eIMAGE_AFMEDIAN_WINDOW_11x11,
    eIMAGE_AFMEDIAN_WINDOW_13x13,
    eIMAGE_AFMEDIAN_WINDOW_15x15,
    eIMAGE_AFMEDIAN_WINDOW_17x17,
    eIMAGE_AFMEDIAN_WINDOW_TOTAL_VARIANTS
};

constexpr char windowSizeStr[] = // computed as: FilterRadiua * 2 + 1
{
    "DISABLE|"
    "   3x3|"
    "   5x5|"
    "   7x7|"
    "   9x9|"
    " 11x11|"
    " 13x13|"
    " 15x15|"
    " 17x17"
};

constexpr float noiseToleranceMin = 0.f;
constexpr float noiseToleranceMax = 15.f;
constexpr float noiseToleranceDef = noiseToleranceMin;

constexpr char noiseToleranceStr[] = "Noise Tolerance";

constexpr int32_t iterPassMin = 1;
constexpr int32_t iterPassMax = 4;
constexpr int32_t iterPassDef = iterPassMin;

enum class AFMF_Iterations : int32_t
{
    eIMAGE_AFMEDIAN_ITER_1 = 0,
    eIMAGE_AFMEDIAN_ITER_2,
    eIMAGE_AFMEDIAN_ITER_3,
    eIMAGE_AFMEDIAN_ITER_4,
    eIMAGE_AFMEDIAN_ITER_TOTAL_VARIANTS
};
constexpr char iterPassStr[] = 
{
    "x 1|"
    "x 2|"
    "x 3|"
    "x 4"
};


#endif // __IMAGE_LAB_ADAPTIVE_FREQUENCY_MEDIAN_FILTER_ENUMS__