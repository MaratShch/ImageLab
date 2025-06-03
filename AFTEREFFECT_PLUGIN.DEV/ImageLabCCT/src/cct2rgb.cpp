#include "Cct2rgb.hpp"

constexpr std::array<ColorTriplet, CctBarSize> CCT2RGB = {
    #include "CCT_RGB_1931_1000K_25000K_step100K.txt"
};


const std::array<ColorTriplet, CctBarSize>& get_cct_map_for_gui (void) noexcept
{
    return CCT2RGB;
}

const std::array<ColorTriplet, CctBarSize>& get_cct_map_for_gui (size_t& size) noexcept
{
    size = CctBarSize;
    return CCT2RGB;
}

const ColorTriplet get_rgb_by_cct (const uint32_t cct) noexcept
{
    constexpr ColorTriplet Bad { 0u, 0u, 0u };
    return (cct >= 1000u & cct <= 25000u ? CCT2RGB[(cct - 1000u)/100u] : Bad);
}