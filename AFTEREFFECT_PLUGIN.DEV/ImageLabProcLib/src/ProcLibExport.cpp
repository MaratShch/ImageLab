#include "ProcLibExport.hpp"
#include "fft.hpp"
#include <vector>

int compute_prime (int imgSize, int arraySize, int* ptr)
{
    int bRet = 0;

    if (nullptr != ptr && arraySize >= 24 && imgSize > 128)
    {
        const std::vector<int> prime = FourierTransform::prime (imgSize);
        if (0 != prime.size() && arraySize >= prime.size())
        {
            for (const auto& f : prime)
            {
                if (f > 9 && f != 16)
                    return 0;
                bRet++;
            }

            if (arraySize > prime.size())
                std::memcpy(ptr, prime.data(), prime.size() * sizeof(int));
            else
                bRet = 0;
        }
    }

    return bRet;
}


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
