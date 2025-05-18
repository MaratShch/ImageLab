#ifndef __IMAGE_LAB_PLANCKIAN_LOCUS_ALGORITHM__
#define __IMAGE_LAB_PLANCKIAN_LOCUS_ALGORITHM__

#include <type_traits>
#include <limits>
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
    inline T Planck_Accurated (const T& wave_length_nm, const T& temperature_K) noexcept
    {
        // Ensure inputs are physically reasonable if necessary
        if (wave_length_nm <= static_cast<T>(0.0) || temperature_K <= static_cast<T>(0.0)) {
            return static_cast<T>(0.0); // Or handle error appropriately
        }

        // Use full precision literals for constants, ensure T conversion
        constexpr T h = static_cast<T>(6.62607015e-34);    // Planck's constant J·s
        constexpr T c = static_cast<T>(299792458.0);       // Speed of light m/s
        constexpr T k = static_cast<T>(1.380649e-23);      // Boltzmann constant J/K
        constexpr T pi = static_cast<T>(3.1415926535897932384626433832795); // More digits for pi

       // Precompute constants with care
       // C1 = 2 * pi * h * c^2  (Units: J·s * m^2/s^2 = J·m^2/s)
        constexpr T c_sq = c * c;
        constexpr T C1 = static_cast<T>(2.0) * pi * h * c_sq;

        // C2 = h * c / k         (Units: J·s * m/s / (J/K) = m·K)
        constexpr T hc = h * c;
        constexpr T C2 = hc / k;

        // Convert wavelength from nm to m
        constexpr T nm_to_m_factor = static_cast<T>(1e-9);
        const T lambda_m = wave_length_nm * nm_to_m_factor;

        // Avoid division by zero if lambda_m is zero (though checked above)
        if (lambda_m == static_cast<T>(0.0)) {
            return static_cast<T>(0.0);
        }

        const T lambda_m_pow5 = std::pow(lambda_m, static_cast<T>(5.0));
        // Check for underflow/overflow in pow, though less likely for typical lambda_m
        if (lambda_m_pow5 == static_cast<T>(0.0) && lambda_m != static_cast<T>(0.0)) {
            // Underflow in pow, result is effectively zero
            return static_cast<T>(0.0);
        }

        const T exp_denominator_term = lambda_m * temperature_K;
        if (exp_denominator_term == static_cast<T>(0.0)) {
            // Leads to division by zero in exp_arg or C2 is zero.
            // If C2 is non-zero, implies infinite exp_arg.
            // If C2 is zero, exp_arg is NaN or Inf.
            // For physical Planck, if T=0 or lambda=0, intensity is 0.
            // If T is very large, or lambda is very small, exp_arg can be large.
            // This case needs careful consideration of physical limits.
            // If exp_arg would be infinite, exp(inf)-1 is inf, result is 0.
            return static_cast<T>(0.0); // Simplified handling
        }

        const T exp_arg = C2 / exp_denominator_term;

        // Check for potential overflow for exp_arg
        // Max exp argument for double before overflow is ~709.78
        // Max exp argument for float before overflow is ~88.72
        if (exp_arg > std::numeric_limits<T>::max_exponent10 * std::log(static_cast<T>(10.0)) * static_cast<T>(0.9))
        { // Heuristic
          // A more direct check:
          // T max_exp_val = std::log(std::numeric_limits<T>::max()); // Max value exp can take before its result overflows
          // if (exp_arg > max_exp_val) {
          // Denominator becomes extremely large, so Planck radiance is effectively zero.
            return static_cast<T>(0.0);
        }

        // Use std::expm1 for accuracy when exp_arg is small
        T exp_val_minus_1;
#if __cplusplus >= 201103L || (defined(_MSC_VER) && _MSC_VER >= 1800)
        exp_val_minus_1 = std::expm1(exp_arg);
#else
        // Fallback if std::expm1 is not available
        // This path is less accurate for small exp_arg
        if (std::abs(exp_arg) < static_cast<T>(1e-5)) { // For small x, exp(x)-1 approx x
                                                        // Taylor series: x + x^2/2! + x^3/3! ...
                                                        // Using a few terms can be better than std::exp(exp_arg) - 1.0
                                                        // T x = exp_arg;
                                                        // exp_val_minus_1 = x * (static_cast<T>(1.0) + x * (static_cast<T>(0.5) + x * static_cast<T>(1.0/6.0)));
                                                        // However, sticking to original formulation if no expm1
            exp_val_minus_1 = std::exp(exp_arg) - static_cast<T>(1.0);
        }
        else {
            exp_val_minus_1 = std::exp(exp_arg) - static_cast<T>(1.0);
        }
#endif

        if (exp_val_minus_1 == static_cast<T>(0.0)) {
            // This happens if exp_arg was exactly 0 (or expm1(0)=0).
            // Denominator is zero, implies infinite radiance, which is unphysical.
            // Or it could be due to underflow if exp_arg was very negative, making expm1(exp_arg) = -1.0.
            // If exp_arg is very negative, expm1(exp_arg) -> -1.
            // lambda_pow5 * (-1) is negative. C1 is positive. Result should be negative (unphysical).
            // This usually means we are far in the tail where intensity is negligible.
            if (exp_arg < static_cast<T>(0.0)) return static_cast<T>(0.0); // If exp_arg is negative, intensity is tiny.
                                                                           // If exp_arg is 0, it suggests an issue with inputs or limits.
                                                                           // For now, treating as zero intensity.
            return std::numeric_limits<T>::infinity(); // Or some other indicator of singularity
                                                       // Or more safely, return 0 if it's a limiting case
        }

        // Final computation
        // Denominator = lambda_m_pow5 * exp_val_minus_1
        // Avoid multiplying two potentially very small or very large numbers if possible,
        // but here it seems direct.
        T denominator = lambda_m_pow5 * exp_val_minus_1;
        if (denominator == static_cast<T>(0.0) && C1 != static_cast<T>(0.0)) {
            return (C1 > static_cast<T>(0.0) ? std::numeric_limits<T>::infinity() : -std::numeric_limits<T>::infinity());
            // Or handle as 0 if it's a physical limit
        }
        if (denominator == static_cast<T>(0.0) && C1 == static_cast<T>(0.0)) {
            return static_cast<T>(0.0); // 0/0 case, treat as 0.
        }

        return C1 / denominator;
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
