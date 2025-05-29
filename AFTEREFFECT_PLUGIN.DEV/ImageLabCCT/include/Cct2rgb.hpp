#ifndef __IMAGE_LAB_IMAGE_COLOR_TEMPERATURE_RGB_MATCH__
#define __IMAGE_LAB_IMAGE_COLOR_TEMPERATURE_RGB_MATCH__

#include <array>

constexpr size_t CctLimitMin  = 1000u;
constexpr size_t CctLimitMax  = 25000u;
constexpr size_t CctLimitStep = 100u;
constexpr size_t CctBarSize = static_cast<size_t>(1) + (CctLimitMax - CctLimitMin) / CctLimitStep;

using ColorTriplet = std::array<unsigned char, 3>;

const std::array<ColorTriplet, CctBarSize>& get_cct_map_for_gui (void) noexcept;


#endif // __IMAGE_LAB_IMAGE_COLOR_TEMPERATURE_RGB_MATCH__