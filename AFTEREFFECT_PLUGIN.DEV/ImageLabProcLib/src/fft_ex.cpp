#include <vector>
#include "fft.hpp"
#include "dft.hpp"
#include "utils.hpp"

void FourierTransform::mixed_radix_fft_1D_ex (const float* in, float* out, const float* RESTRICT twiddle_table, ptrdiff_t size) noexcept
{
    return;
}

void FourierTransform::mixed_radix_fft_1D_ex (const double* in, double* out, const double* RESTRICT twiddle_table, ptrdiff_t size) noexcept
{
    return;
}


void FourierTransform::mixed_radix_fft_2D_ex
(
    const float*  RESTRICT in, 
    float*  RESTRICT scratch, 
    float*  RESTRICT out, 
    const float*  RESTRICT twiddleX, 
    const float*  RESTRICT twiddleY, 
    ptrdiff_t width,
    ptrdiff_t height
) noexcept
{
    // TODO
    return;
}

void FourierTransform::mixed_radix_fft_2D_ex
(
    const double* RESTRICT in, 
    double* RESTRICT scratch, 
    double* RESTRICT out, 
    const double* RESTRICT twiddleX, 
    const double* RESTRICT twiddleY, 
    ptrdiff_t width,
    ptrdiff_t height
) noexcept
{
    // TODO
    return;
}
