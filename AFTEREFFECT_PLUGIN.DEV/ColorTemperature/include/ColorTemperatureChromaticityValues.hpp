#ifndef __IMAGE_LAB_IMAGE_COLOR_TEMPERATURE_CHROMATICITY_VALUES__
#define __IMAGE_LAB_IMAGE_COLOR_TEMPERATURE_CHROMATICITY_VALUES__

#include <utility>
#include <tuple>
#include "ColorCurves.hpp"

template<typename T, typename std::enable_if<std::is_floating_point<T>::value>::type* = nullptr>
inline std::vector<const std::pair<T, T>> compute_chromaticity_values (const std::vector<std::vector<T>>& observer) noexcept
{
    std::vector<const std::pair<T, T>> chromaticity_vector{};
    for (const auto& XYZ : observer)
    {
        // x = X / (X + Y + Z)
        const T x = XYZ[0] / (XYZ[0] + XYZ[1] + XYZ[2]);
        // y = Y / (X + Y + Z)
        const T y = XYZ[1] / (XYZ[0] + XYZ[1] + XYZ[2]);
        chromaticity_vector.push_back({ x, y });
    }

    return chromaticity_vector;
}


template<typename T, typename std::enable_if<std::is_floating_point<T>::value>::type* = nullptr>
inline std::tuple<const T, const T, const T> compute_XYZ(const std::vector<std::vector<T>>& observer, const std::vector<T>& illuminant, const T& waveStep) noexcept
{
    T scalarX = static_cast<T>(0);
    T scalarY = static_cast<T>(0);
    T scalarZ = static_cast<T>(0);

    if (observer.size() == illuminant.size())
    {
        const auto size = illuminant.size();
        for (size_t i = 0; i < size; i++)
        {
            scalarX += (observer[i][0] * illuminant[i] * waveStep);
            scalarY += (observer[i][1] * illuminant[i] * waveStep);
            scalarZ += (observer[i][2] * illuminant[i] * waveStep);
        }
    }

    return std::make_tuple(scalarX, scalarY, scalarZ);
}


#endif // __IMAGE_LAB_IMAGE_COLOR_TEMPERATURE_CHROMATICITY_VALUES__