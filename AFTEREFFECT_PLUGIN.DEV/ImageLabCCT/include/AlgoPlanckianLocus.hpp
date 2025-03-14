#ifndef __IMAGE_LAB_PLANCKIAN_LOCUS_ALGORITHM__
#define __IMAGE_LAB_PLANCKIAN_LOCUS_ALGORITHM__

#include <cmath>
#include <vector>
#include <algorithm>

namespace PlanckianLocus
{
    template<typename T, typename std::enable_if<std::is_floating_point<T>::value>::type* = nullptr>
    using CCT_LUT_Entry = std::tuple<T /* CCT */, T /* Duv */, T /* x */, T /* y */>;

	template<typename T, typename std::enable_if<std::is_floating_point<T>::value>::type* = nullptr>
	inline T Planck (const T& wave_length, const T& white_point) noexcept
	{
		constexpr T c{ static_cast<T>(299792458.0) };		/* Speed of light in m / s				*/
		constexpr T	h{ static_cast<T>(6.62607015e-34) };	/* Planck's constant in m^2 kg/s		*/
		constexpr T	k{ static_cast<T>(1.380649e-23) };		/* Boltzmann's constant in m^2 kg/s^2 K	*/
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
    inline constexpr T distance(const T& x1, const T& y1, const T& x2, const T& y2) noexcept
    {
        return std::sqrt((x2 - x1) * (x2 - x1) + (y2 - y1) * (y2 - y1));
    }

    // Function to calculate Duv
    template<typename T, typename std::enable_if<std::is_floating_point<T>::value>::type* = nullptr>
    inline constexpr T calculateDuv(const T& x_input, const T& y_input, const T& x_locus, const T& y_locus) noexcept
    {
        // Simplified Duv calculation.  Assumes a relatively small distance.
        // For higher accuracy, you'd ideally use a more complex formula
        // that accounts for the curvature of the chromaticity diagram.
        return (y_input - y_locus) - (static_cast<T>(0.3320) * (x_input - x_locus));
    }


    template<typename T, typename std::enable_if<std::is_floating_point<T>::value>::type* = nullptr>
    void projectToPlanckianLocus(T input_x, T input_y, T& output_x, T& output_y, T& output_duv, const std::vector<CCT_LUT_Entry<T>>& planckianLocus) noexcept
    {
        if (planckianLocus.empty())
        {
            output_x = static_cast<T>(0.0);
            output_y = static_cast<T>(0.0);
            output_duv = static_cast<T>(0.0);
            return;
        }

        // Find the closest point on the Planckian locus
        //   auto closest_it = std::min_element(
        //       planckianLocus.begin(), planckianLocus.end(),
        //       [input_x, input_y](const CCT_LUT_Entry<T>& p1, const CCT_LUT_Entry<T>& p2) {
        //       return distance(input_x, input_y, std::get<2>(p1), std::get<3>(p1)) < distance(input_x, input_y, std:get<2>(p2), std::get<3>(p2));
        //   });

        // Get the x, y, and CCT of the closest point (dereference the iterator)
        //   output_x = closest_it->x;
        //   output_y = closest_it->y;
        // output_cct = closest_it->cct;  // Not needed here

        // Calculate Duv (simplified)
        output_duv = calculateDuv(input_x, input_y, output_x, output_y);
    }

} // namespace PlanckianLocus

#endif /* __IMAGE_LAB_PLANCKIAN_LOCUS_ALGORITHM__ */