#pragma once

#include "fft_prime.hpp"
#include "fft_mixed_radix.hpp"
#include "fft_split_radix.hpp"
#include "fft_cooley_tukey.hpp"
#include "fft_czt.hpp"

namespace FourierTransform
{
void mixed_radix_fft_1D (const float* __restrict in, float* __restrict out, int32_t size) noexcept;
void mixed_radix_fft_1D(const double* __restrict in, double* __restrict out, int32_t size) noexcept;

void mixed_radix_ifft_1D (const float* __restrict in, float* __restrict out, int32_t size) noexcept;
void mixed_radix_ifft_1D (const double* __restrict in, double* __restrict out, int32_t size) noexcept;

}