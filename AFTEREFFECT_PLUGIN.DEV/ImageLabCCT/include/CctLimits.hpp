#ifndef __IMAGE_LAB_IMAGE_COLOR_TEMPERATURE_VALUES_LIMITS__
#define __IMAGE_LAB_IMAGE_COLOR_TEMPERATURE_VALUES_LIMITS__

namespace CCT_Limits
{
    constexpr float waveLenMin = 380.f; // nm
    constexpr float waveLenMax = 740.f; // nm
    constexpr float waveLenStep = 0.5f;  // nm

    constexpr float cctMin = 1000.f;    // K  
    constexpr float cctMax = 16000.f;   // K
    constexpr float cctStep = 10.f;     // K 
} // namespace CCT_Limits

#endif // __IMAGE_LAB_IMAGE_COLOR_TEMPERATURE_VALUES_LIMITS__
