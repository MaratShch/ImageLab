#ifndef __IMAGE_LAB_ALGORITHM_WINOGRAD_FFT__
#define __IMAGE_LAB_ALGORITHM_WINOGRAD_FFT__

#include <cmath>

void winograd_1d_fft_avx2(const float*  __restrict input, size_t input_size, float*  __restrict output_real, float*  __restrict output_imag, size_t output_size) noexcept;
void winograd_1d_fft_avx2(const double* __restrict input, size_t input_size, double* __restrict output_real, double* __restrict output_imag, size_t output_size) noexcept;


template<typename T, typename std::enable_if<std::is_floating_point<T>::value>::type* = nullptr>
inline void winograd_1d_fft(const T* __restrict input, size_t elem_numbers, T* __restrict output_real, T* __restrict output_imag) noexcept
{
    constexpr T PIx2{ static_cast<const T>(2.0 * 3.14159265358979323846) };
    const T PIx2DivSize = PIx2 / static_cast<T>(elem_numbers);

    for (size_t k = 0; k < elem_numbers; k++)
    {
        T real = static_cast<T>(0);
        T imag = static_cast<T>(0);
        for (size_t n = 0; n < elem_numbers; n++)
        {
            const T fVal = PIx2DivSize * static_cast<T>(n * k);
            const T cos_term = std::cos(fVal);
            const T sin_term = -std::sin(fVal);

            /* Accumulate the real and imaginary parts */
            real += (input[n] * cos_term);
            imag += (input[n] * sin_term);
        }
        output_real[k] = real;
        output_imag[k] = imag;
    }

    return;
}

#endif // __IMAGE_LAB_ALGORITHM_WINOGRAD_FFT__