#include <vector>
#include "fft.hpp"
#include "dft.hpp"
#include "utils.hpp"

void FourierTransform::mixed_radix_fft_1D_ex (const float* in, float* out, const float* RESTRICT twiddle_table, int32_t size) noexcept
{
    return;
}

void FourierTransform::mixed_radix_fft_1D_ex (const double* in, double* out, const double* RESTRICT twiddle_table, int32_t size) noexcept
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
    int32_t width, 
    int32_t height
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
    int32_t width, 
    int32_t height
) noexcept
{
    // TODO
    return;
}
