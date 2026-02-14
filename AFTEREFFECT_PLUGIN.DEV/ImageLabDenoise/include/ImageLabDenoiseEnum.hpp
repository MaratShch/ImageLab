#ifndef __IMAGE_LAB_DENOISE_FILTER_ENUMERATORS__
#define __IMAGE_LAB_DENOISE_FILTER_ENUMERATORS__

#include <cstdint>
#include "AE_Effect.h"
#include "CompileTimeUtils.hpp"

enum class DenoiseControl : int32_t
{
    eIMAGE_LAB_DENOISE_INPUT,
    eIMAGE_LAB_DENOISE_AMOUNT,
    eIMAGE_LAB_DENOISE_LUMA_STRENCGTH,
    eIMAGE_LAB_DENOISE_CHROMA_STRENCGTH,
    eIMAGE_LAB_DEENOISE_DETAILS_PRESERVATION,
    eIMAGE_LAB_DENOSIE_MATCH_SENSITIVITY,
    eIMAGE_LAB_DENOISE_SEARCH_RADIUS,
    eIMAGE_LAB_DENOISE_STRIDE,
    eIMAGE_LAB_DENOISE_LOW_FREQUENCY_MULT,
    eIMAGE_LAB_DENOISE_HIGH_FREQUENCY_MULT,
    eIMAGE_LAB_DENOISE_CONTROLS
};

constexpr inline float Slider2Value (const int32_t sliderVal) noexcept
{
    return (static_cast<float>(sliderVal) / 100.f);
}

constexpr inline float FreqSlider2Value (const int32_t sliderVal) noexcept
{
    return (static_cast<float>(sliderVal + 100) / 100.f);
}

enum class ProcAccuracy : int32_t
{
    AccDraft = 0,	// Draft (Stride 5) - Fastest for scrubbing the timeline.
    AccStandard,	// Standard (Stride 3) - Good balance.
    AccHigh,		// High (Stride 2) - Standard high-quality.
    AccMaster	 	// Master (Stride 1) - Slowest, best for final render.
};

constexpr int32_t eDenoiseAmountMin = 0;
constexpr int32_t eDenoiseAmountMax = 200;
constexpr int32_t eDenoiseAmountDef = 100;

constexpr int32_t eDetailPreservMin = 0;
constexpr int32_t eDetailPreservMax = 50;
constexpr int32_t eDetailPreservDef = 25;

constexpr float eMatchSensitivityMin = 0.5f;
constexpr float eMatchSensitivityMax = 5.0f;
constexpr float eMatchSensitivityDef = 2.5f;

constexpr int32_t eSearchRadiusMin = 4;
constexpr int32_t eSearchRadiusMax = 32;
constexpr int32_t eSearchRadiusDef = 10;

constexpr int32_t eLowFreqMultMin = -100;
constexpr int32_t eLowFreqMultMax = 0;
constexpr int32_t eLowFreqMultDef = 100;

constexpr int32_t eHighFreqMultMin = -100;
constexpr int32_t eHighFreqMultMax = 0;
constexpr int32_t eHighFreqMultDef = 100;



#endif // __IMAGE_LAB_DENOISE_FILTER_ENUMERATORS__