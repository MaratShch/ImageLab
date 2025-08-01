#ifndef __IMAGE_LAB_IMAGE_COLOR_TEMPERATURES_ALGO_LUT__
#define __IMAGE_LAB_IMAGE_COLOR_TEMPERATURES_ALGO_LUT__


#include <limits>
#include <type_traits>
#include "ColorCurves.hpp"
#include "ColorIlluminant.hpp"
#include "ColorTemperatureChromaticityValues.hpp"
#include "ColorTransformMatrix.hpp"

template<typename T, typename std::enable_if<std::is_floating_point<T>::value>::type* = nullptr>
struct CCT_LUT_Entry
{
   T cct;
   T u;
   T v;
   T Duv;
};


template<typename T, typename std::enable_if<std::is_floating_point<T>::value>::type* = nullptr>
inline void xy_to_uv (const T& x, const T& y, T& u_prime, T& v_prime) noexcept
{
    const T denom = static_cast<T>(-2) * x + static_cast<T>(12) * y + static_cast<T>(3);
    u_prime = (static_cast<T>(4) * x) / denom;
    v_prime = (static_cast<T>(6) * y) / denom;
    return;
}

template<typename T, typename std::enable_if<std::is_floating_point<T>::value>::type* = nullptr>
inline void XYZ_to_uv_1960 (const T& X, const T& Y, const T& Z, T& u_prime, T& v_prime) noexcept
{
    // u = (4*X) / (X + 15*Y + 3*Z)
    // v = (6*Y) / (X + 15*Y + 3*Z)
    const T denom = X + static_cast<T>(15) * Y + static_cast<T>(3) * Z;
    u_prime = static_cast<T>(0) != denom ? (static_cast<T>(4) * X) / denom : static_cast<T>(0);
    v_prime = static_cast<T>(0) != denom ? (static_cast<T>(6) * Y) / denom : static_cast<T>(0);
    return;
}

template<typename T, typename std::enable_if<std::is_floating_point<T>::value>::type* = nullptr>
inline void XYZ_to_uv_1976 (const T& X, const T& Y, const T& Z, T& u_prime, T& v_prime) noexcept
{
    // u = (4*X) / (X + 15*Y + 3*Z)
    // v = (9*Y) / (X + 15*Y + 3*Z)
    const T denom = X + static_cast<T>(15) * Y + static_cast<T>(3) * Z;
    u_prime = static_cast<T>(0) != denom ? (static_cast<T>(4) * X) / denom : static_cast<T>(0);
    v_prime = static_cast<T>(0) != denom ? (static_cast<T>(9) * Y) / denom : static_cast<T>(0);
    return;
}

template<typename T, typename std::enable_if<std::is_floating_point<T>::value>::type* = nullptr>
inline const T polinomial_compute_duv (const T& u_prime, const T& v_prime) noexcept
{
    const T u_diff = u_prime - static_cast<T>(0.292);
    const T v_diff = v_prime - static_cast<T>(0.240);
    const T Lfp = std::sqrt(u_diff * u_diff + v_diff * v_diff);
    const T a = std::acos(u_diff / Lfp);

    constexpr T k6{ static_cast<T>(-0.00616793)};
    constexpr T k5{ static_cast<T>( 0.0893944) };
    constexpr T k4{ static_cast<T>(-0.5179722) };
    constexpr T k3{ static_cast<T>( 1.5317403) };
    constexpr T k2{ static_cast<T>(-2.4243787) };
    constexpr T k1{ static_cast<T>( 1.925865)  };
    constexpr T k0{ static_cast<T>(-0.471106)  };
 
    const T a2 = a  * a;
    const T a3 = a2 * a;
    const T a4 = a2 * a2;
    const T a5 = a2 * a3;
    const T a6 = a3 * a3;

    const T Lbb = k6 * a6 + k5 * a5 + k4 * a4 + k3 * a3 + k2 * a2 + k1 * a + k0;

    return (Lfp - Lbb);
}



template<typename T, typename std::enable_if<std::is_floating_point<T>::value>::type* = nullptr>
std::vector<CCT_LUT_Entry<T>> initLut (T waveMin, T waveMax, T waveStep, T cctMin, T cctMax, T cctStep, const eCOLOR_OBSERVER observer) noexcept
{
    const std::size_t lutSize = static_cast<std::size_t>((cctMax - cctMin) / cctStep) + 1ull;
    std::vector<CCT_LUT_Entry<T>> cctLut(lutSize);

    // initialize Color Observer
    auto const& colorObserver = (observer_CIE_1931 == observer ?
        generate_color_curves_1931_observer(waveMin, waveMax, waveStep) : generate_color_curves_1964_observer(waveMin, waveMax, waveStep));

    int32_t idx = 0;

    for (T cctVal = cctMin; cctVal <= cctMax; cctVal += cctStep)
    {
        auto const& colorIllumnant = init_illuminant(waveMin, waveMax, waveStep, cctVal);
        std::tuple<const T, const T, const T> scalar_XYZ = compute_scalar_XYZ(colorObserver, colorIllumnant);

        const T X = std::get<0>(scalar_XYZ);
        const T Y = std::get<1>(scalar_XYZ);
        const T Z = std::get<2>(scalar_XYZ);

        T u_prime, v_prime;
        XYZ_to_uv_1960 (X, Y, Z, u_prime, v_prime);

        const T Duv = polinomial_compute_duv(u_prime, v_prime);

        const CCT_LUT_Entry<T> lutElement
        {
            cctVal,
            u_prime,
            v_prime,
            Duv
        };

        cctLut[idx] = lutElement;
        idx++;
    } // for (T cctVal = cctMin; cctVal <= cctMax; cctVal += cctStep)

    return cctLut;
}


// Function to calculate CCT (now templated)
template<typename T, typename std::enable_if<std::is_floating_point<T>::value>::type* = nullptr>
std::pair<T, CCT_LUT_Entry<T>> calculateCCT_NearestNeighbor
(
    const T& u_input, 
    const T& v_input,
    const std::vector<CCT_LUT_Entry<T>>& lut
) noexcept
{
    T min_distance{ (std::numeric_limits<T>::max)() };
    size_t closest_index = 0;
   
    // Iterate through the LUT to find the nearest neighbor
    for (size_t i = 0; i < lut.size(); ++i)
    {
        const T diff_u = u_input - lut[i].u;
        const T diff_v = v_input - lut[i].v;
        const T distance = std::sqrt((diff_u * diff_u) + (diff_v * diff_v));

        if (distance < min_distance)
        {
            min_distance = distance;
            closest_index = i;
        }
     }

    // Return the CCT and the closest LUT entry
    return {lut[closest_index].cct, lut[closest_index]};
}


#endif // __IMAGE_LAB_IMAGE_COLOR_TEMPERATURES_ALGO_LUT__
