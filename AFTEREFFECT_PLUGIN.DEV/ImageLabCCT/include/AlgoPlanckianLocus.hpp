#ifndef __IMAGE_LAB_PLANCKIAN_LOCUS_ALGORITHM__
#define __IMAGE_LAB_PLANCKIAN_LOCUS_ALGORITHM__

#include <cmath>
#include <vector>
#include <algorithm>

namespace PlanckianLocus
{
	template<typename T, typename std::enable_if<std::is_floating_point<T>::value>::type* = nullptr>
	inline T Planck (const T& wave_length, const T& white_point) noexcept
	{
        constexpr T c{ static_cast<T>(299792458.0) };       /* Speed of light in m / s              */
        constexpr T h{ static_cast<T>(6.62607015e-34) };    /* Planck's constant in m^2 kg/s        */
		constexpr T k{ static_cast<T>(1.380649e-23) };      /* Boltzmann's constant in m^2 kg/s^2 K */
	    constexpr T pi{ static_cast<T>(3.14159265358979323846) };
        constexpr T div{ static_cast<T>(1e9) };

        constexpr T first_radiaton_constant{ static_cast<T>(2.0) * pi * h * c * c }; /* 2 * pi * h * c ^ 2 */
        constexpr T second_radiation_constant{ (h * c) / k };

		const T lambda = wave_length / div;	/* Convert wave length from nano-meters to meters */
		return (first_radiaton_constant / (std::pow(lambda, static_cast<T>(5.0)) * (std::exp(second_radiation_constant / (lambda * white_point)) - static_cast<T>(1.0))));
	}


	template<typename T, typename std::enable_if<std::is_floating_point<T>::value>::type* = nullptr>
	inline void SpectralRadianceNormalize (const T& sr_max_value, std::vector<T>& SpectralRadianceVector) noexcept
	{
		const T sr_max_reciproc = static_cast<T>(1.0) / sr_max_value;
		const size_t vecSize = SpectralRadianceVector.size();
		for (size_t i = 0; i < vecSize; i++)
			SpectralRadianceVector[i] *= sr_max_reciproc;
		return;
	}


    // Function to calculate Euclidean distance
    template<typename T, typename std::enable_if<std::is_floating_point<T>::value>::type* = nullptr>
    inline constexpr T squared_distance(const T& x1, const T& y1, const T& x2, const T& y2) noexcept
    {
        return std::sqrt((x2 - x1) * (x2 - x1) + (y2 - y1) * (y2 - y1));
    }


}

#endif /* __IMAGE_LAB_PLANCKIAN_LOCUS_ALGORITHM__ */
