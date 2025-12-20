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

void mixed_radix_fft_1D (const float*  in, float*  out, ptrdiff_t size) noexcept;
void mixed_radix_fft_1D (const double* in, double* out, ptrdiff_t size) noexcept;
void mixed_radix_fft_2D (const float*  RESTRICT in, float*  RESTRICT scratch, float*  RESTRICT out, ptrdiff_t width, ptrdiff_t height) noexcept;
void mixed_radix_fft_2D (const double* RESTRICT in, double* RESTRICT scratch, double* RESTRICT out, ptrdiff_t width, ptrdiff_t height) noexcept;

void mixed_radix_ifft_1D (const float*  RESTRICT in, float*  RESTRICT out, ptrdiff_t size) noexcept;
void mixed_radix_ifft_1D (const double* RESTRICT in, double* RESTRICT out, ptrdiff_t size) noexcept;
void mixed_radix_ifft_2D (const float*  RESTRICT in, float*  RESTRICT scratch, float*  RESTRICT out, ptrdiff_t width, ptrdiff_t height) noexcept;
void mixed_radix_ifft_2D (const double* RESTRICT in, double* RESTRICT scratch, double* RESTRICT out, ptrdiff_t width, ptrdiff_t height) noexcept;

void mixed_radix_fft_1D_ex  (const float*  in, float*  out, const float*  RESTRICT twiddleT, ptrdiff_t size) noexcept;
void mixed_radix_fft_1D_ex  (const double* in, double* out, const double* RESTRICT twiddleT, ptrdiff_t size) noexcept;

void mixed_radix_fft_2D_ex (const float*  RESTRICT in, float*  RESTRICT scratch, float*  RESTRICT out, const float*  RESTRICT twiddleX, const float*  RESTRICT twiddleY, ptrdiff_t width, ptrdiff_t height) noexcept;
void mixed_radix_fft_2D_ex (const double* RESTRICT in, double* RESTRICT scratch, double* RESTRICT out, const double* RESTRICT twiddleX, const double* RESTRICT twiddleY, ptrdiff_t width, ptrdiff_t height) noexcept;

}