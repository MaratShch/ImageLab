#pragma once

#include <cstdint>
#include "fft_prime.hpp"
#include "fft_mixed_radix.hpp"
#include "fft_split_radix.hpp"
#include "fft_cooley_tukey.hpp"
#include "fft_czt.hpp"
#include "Common.hpp"

namespace FourierTransform
{

void mixed_radix_fft_1D (const float*  in, float*  out, int32_t size) noexcept;
void mixed_radix_fft_1D (const double* in, double* out, int32_t size) noexcept;
void mixed_radix_fft_2D (const float*  RESTRICT in, float*  RESTRICT scratch, float*  RESTRICT out, int32_t width, int32_t height) noexcept;
void mixed_radix_fft_2D (const double* RESTRICT in, double* RESTRICT scratch, double* RESTRICT out, int32_t width, int32_t height) noexcept;

void mixed_radix_ifft_1D (const float*  RESTRICT in, float*  RESTRICT out, int32_t size) noexcept;
void mixed_radix_ifft_1D (const double* RESTRICT in, double* RESTRICT out, int32_t size) noexcept;
void mixed_radix_ifft_2D (const float*  RESTRICT in, float*  RESTRICT scratch, float*  RESTRICT out, int32_t width, int32_t height) noexcept;
void mixed_radix_ifft_2D (const double* RESTRICT in, double* RESTRICT scratch, double* RESTRICT out, int32_t width, int32_t height) noexcept;

void mixed_radix_fft_1D_ex  (const float*  in, float*  out, const float*  RESTRICT twiddleT, int32_t size) noexcept;
void mixed_radix_fft_1D_ex  (const double* in, double* out, const double* RESTRICT twiddleT, int32_t size) noexcept;

void mixed_radix_fft_2D_ex (const float*  RESTRICT in, float*  RESTRICT scratch, float*  RESTRICT out, const float*  RESTRICT twiddleX, const float*  RESTRICT twiddleY, int32_t width, int32_t height) noexcept;
void mixed_radix_fft_2D_ex (const double* RESTRICT in, double* RESTRICT scratch, double* RESTRICT out, const double* RESTRICT twiddleX, const double* RESTRICT twiddleY, int32_t width, int32_t height) noexcept;

}