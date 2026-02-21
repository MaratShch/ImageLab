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



#endif // __IMAGE_LAB_DENOISE_FILTER_ENUMERATORS__