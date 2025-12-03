#include "ProcLibExport.hpp"
#include "fft.hpp"

void fft_f32 (const float* in, float* out, int size)
{
    if (nullptr != in && nullptr != out && size >= 2 && in != out)
        FourierTransform::mixed_radix_fft_1D (in, out, size);

    return;
}

void fft_f64 (const double* in, double* out, int size)
{
    if (nullptr != in && nullptr != out && size >= 2 && in != out)
        FourierTransform::mixed_radix_fft_1D(in, out, size);

    return;
}

void fft2d_f32 (const float* in, float* scratch, float* out, int sizeX, int sizeY)
{
    if (nullptr != in && nullptr != out && nullptr != scratch)
        FourierTransform::mixed_radix_fft_2D(in, scratch, out, sizeX, sizeY);

    return;
}

void fft2d_f64 (const double* in, double* scratch, double* out, int sizeX, int sizeY)
{
    if (nullptr != in && nullptr != out && nullptr != scratch)
        FourierTransform::mixed_radix_fft_2D(in, scratch, out, sizeX, sizeY);

    return;
}


void ifft_f32 (const float* in, float* out, int size)
{
    if (nullptr != in && nullptr != out && size >= 2 && in != out)
        FourierTransform::mixed_radix_ifft_1D(in, out, size);

    return;
}

void ifft_f64 (const double* in, double* out, int size)
{
    if (nullptr != in && nullptr != out && size >= 2 && in != out)
        FourierTransform::mixed_radix_ifft_1D(in, out, size);

    return;
}


void ifft2d_f32 (const float* in, float* scratch, float* out, int sizeX, int sizeY)
{
    if (nullptr != in && nullptr != out && nullptr != scratch)
        FourierTransform::mixed_radix_ifft_2D(in, scratch, out, sizeX, sizeY);

    return;
}

void ifft2d_f64 (const double* in, double* scratch, double* out, int sizeX, int sizeY)
{
    if (nullptr != in && nullptr != out && nullptr != scratch)
        FourierTransform::mixed_radix_ifft_2D(in, scratch, out, sizeX, sizeY);

    return;
}
