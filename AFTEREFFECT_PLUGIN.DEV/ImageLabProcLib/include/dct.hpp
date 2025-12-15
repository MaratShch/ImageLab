#include <cstdint>
#include <cmath>
#include "Common.hpp"

namespace FourierTransform
{

    template <typename T>
    inline void dct_1D
    (
        const T* RESTRICT in,  // Input: Array of N Real numbers (Pixels)
        T* RESTRICT out,       // Output: Array of N Real numbers (Coefficients)
        int32_t N
    ) noexcept
    {
        constexpr T PI = static_cast<T>(3.14159265358979323846);
        
        // Normalization factors for Orthonormality (Energy Preservation)
        // If you don't do this, the output values will be huge.
        const T alpha0 = static_cast<T>(std::sqrt(static_cast<T>(1.0) / N));
        const T alphaK = static_cast<T>(std::sqrt(static_cast<T>(2.0) / N));

        for (int32_t k = 0; k < N; k++)
        {
            T sum{0};

            // Determine the scaling factor for this frequency k
            const T s = (k == 0) ? alpha0 : alphaK;

            for (int32_t n = 0; n < N; n++)
            {
                // 1. Read Real Input only
                const T xr = in[n];

                // 2. The DCT-II Angle Formula:
                // (n + 0.5) shifts the sample to the "center" of the pixel
                // This is what prevents the "Edge Jump" artifact.
                const T angle = (PI / static_cast<T>(N)) * (static_cast<T>(n) + static_cast<T>(0.5)) * static_cast<T>(k);

                // 3. Compute Cosine
                // Optimization: In production, these Cosines should be a Lookup Table!
                sum += xr * std::cos(angle); 
            }

            // 4. Write Real Output only
            out[k] = s * sum;
        }
    }

}