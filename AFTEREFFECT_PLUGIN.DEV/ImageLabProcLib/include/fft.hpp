#pragma once

#include <cstdint>
#include "fft_prime.hpp"
#include "fft_mixed_radix.hpp"
#include "fft_split_radix.hpp"
#include "fft_cooley_tukey.hpp"
#include "fft_czt.hpp"

namespace FourierTransform
{

void mixed_radix_fft_1D (const float*  __restrict in, float*  __restrict out, int32_t size) noexcept;
void mixed_radix_fft_1D(const double*  __restrict in, double* __restrict out, int32_t size) noexcept;
void mixed_radix_fft_2D (const float*  __restrict in, float*  __restrict scratch, float*  __restrict out, int32_t width, int32_t height) noexcept;
void mixed_radix_fft_2D (const double* __restrict in, double* __restrict scratch, double* __restrict out, int32_t width, int32_t height) noexcept;

void mixed_radix_ifft_1D (const float*  __restrict in, float*  __restrict out, int32_t size) noexcept;
void mixed_radix_ifft_1D (const double* __restrict in, double* __restrict out, int32_t size) noexcept;
void mixed_radix_ifft_2D (const float*  __restrict in, float*  __restrict scratch, float*  __restrict out, int32_t width, int32_t height) noexcept;
void mixed_radix_ifft_2D (const double* __restrict in, double* __restrict scratch, double* __restrict out, int32_t width, int32_t height) noexcept;

}