#include "algo_fft.hpp"

void winograd_1d_fft_avx2 (const float* __restrict input, size_t input_size, float* __restrict output_real, float* __restrict output_imag, size_t output_size) noexcept
{
	winograd_1d_fft (input, input_size, output_real, output_imag);
	return;
}

void winograd_1d_fft_avx2 (const double* __restrict input, size_t input_size, double* __restrict output_real, double* __restrict output_imag, size_t output_size) noexcept
{
	winograd_1d_fft (input, input_size, output_real, output_imag);
	return;
}

/* __m256 _mm256_sincos_ps (__m256 * mem_addr, __m256 a) */