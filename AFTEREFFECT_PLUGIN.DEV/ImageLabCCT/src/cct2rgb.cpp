#include "Cct2rgb.hpp"

constexpr std::array<ColorTriplet, CctBarSize> CCT2RGB = {
    #include "CCT_RGB_1931_1000K_25000K_step100K.txt"
};

const std::array<ColorTriplet, CctBarSize>& get_cct_map_for_gui (void) noexcept
{
    return CCT2RGB;
}