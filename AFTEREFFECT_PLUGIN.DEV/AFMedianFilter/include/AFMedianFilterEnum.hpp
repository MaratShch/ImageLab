#ifndef __IMAGE_LAB_ADAPTIVE_FREQUENCY_MEDIAN_FILTER_ENUMS__
#define __IMAGE_LAB_ADAPTIVE_FREQUENCY_MEDIAN_FILTER_ENUMS__

#include <cstdint>
#include "CompileTimeUtils.hpp"


enum class AFMF : int32_t
{
    eIMAGE_AFMEDIAN_INPUT,
    eIMAGE_AFMEDIAN_INPUT_TYPE,         // control - Luminance or RGB
    eIMAGE_AFMEDIAN_OUTPUT_TYPE,        // control - Denoised iage or Noise map
    eIMAGE_AFMEDIAN_PARAM_RADIUS,       // control - AFMF radius in pixels
    eIMAGE_AFMEDIAN_PARAM_TOLERANCE,    // control - AFMF tolerance
    eIMAGE_AFMEDIAN_PARAM_ITERATIONS,   // control - AAFMF number of iterations
    eIMAGE_AFMEDIAN_TOTAL_CONTROLS      
};

constexpr char AFMFControlsStr[][24] =
{
    "Input Type",
    "Output Type",
    "Window Size",
    "Filter Tolerance",
    "Iterations Number"
};

enum class AFMF_Input : int32_t
{
    AFMF_INPUT_LUMINANCE = 0,
    AFMF_INPUT_ALL_RGB,
    AFMF_INPUT_TOTALS
};

constexpr char afmfInputStr[] =
{
    "Luminance only (faster)|"
    "All channels (RGB)"
};

enum class AFMF_Output : int32_t
{
    AFMF_OUTPUT_IMAGE = 0,
    AFMF_OUTPUT_NOISE_MAP,
    AFMF_OUTPUT_TOTALS
};

constexpr char afmfOutputStr[] =
{
    "Denoised Image|"
    "Noise Map      "
};

constexpr int32_t kernelRadiusMin = 1;
constexpr int32_t kernelRadiusMax = 8;
constexpr int32_t kernelRadiusDef = kernelRadiusMin;

enum class AFMF_RadiusSize : int32_t
{
    eIMAGE_AFMEDIAN_WINDOW_3x3 = 0,
    eIMAGE_AFMEDIAN_WINDOW_5x5,
    eIMAGE_AFMEDIAN_WINDOW_7x7,
    eIMAGE_AFMEDIAN_WINDOW_9x9,
    eIMAGE_AFMEDIAN_WINDOW_11x11,
    eIMAGE_AFMEDIAN_WINDOW_13x13,
    eIMAGE_AFMEDIAN_WINDOW_15x15,
    eIMAGE_AFMEDIAN_WINDOW_17x17,
    eIMAGE_AFMEDIAN_WINDOW_TOTAL_VARIANTS
};

constexpr char windowSizeStr[] = // computed as: FilterRadius * 2 + 1
{
    "   3x3|"
    "   5x5|"
    "   7x7|"
    "   9x9|"
    " 11x11|"
    " 13x13|"
    " 15x15|"
    " 17x17"
};

constexpr float noiseToleranceMin = 0.0f; 
constexpr float noiseToleranceMax = 10.f;
constexpr float noiseToleranceDef = 1.0f;

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