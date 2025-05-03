#ifndef __IMAGE_LAB_IMAGE_COLOR_TEMPERATURE_CHROMATICITY_VALUES__
#define __IMAGE_LAB_IMAGE_COLOR_TEMPERATURE_CHROMATICITY_VALUES__

#include <vector>
#include <utility>
#include <tuple>
#include "ColorCurves.hpp"

template<typename T, typename std::enable_if<std::is_floating_point<T>::value>::type* = nullptr>
inline std::vector<std::pair<T, T>> compute_chromaticity_values (const std::vector<std::vector<T>>& observer)
{
    std::vector<std::pair<T, T>> chromaticity_vector{};
    for (const auto& XYZ : observer)
    {
        // x = X / (X + Y + Z)
        const T x = XYZ[0] / (XYZ[0] + XYZ[1] + XYZ[2]);
        // y = Y / (X + Y + Z)
        const T y = XYZ[1] / (XYZ[0] + XYZ[1] + XYZ[2]);
        chromaticity_vector.push_back({std::make_pair(x, y)});
    }

    return chromaticity_vector;
}




#endif // __IMAGE_LAB_IMAGE_COLOR_TEMPERATURE_CHROMATICITY_VALUES__
