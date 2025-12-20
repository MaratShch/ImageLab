#include "ProcLibExport.hpp"
#include "fft.hpp"
#include "dct.hpp"
#include <vector>

DLL_API_EXPORT int compute_prime (int imgSize, int arraySize, int* ptr)
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


DLL_API_EXPORT void fft_f32 (const float* in, float* out, ptrdiff_t size)
{
    if (nullptr != in && nullptr != out && size >= 2 && in != out)
        FourierTransform::mixed_radix_fft_1D (in, out, size);
    return;
}

DLL_API_EXPORT void fft_f64 (const double* in, double* out, ptrdiff_t size)
{
    if (nullptr != in && nullptr != out && size >= 2 && in != out)
        FourierTransform::mixed_radix_fft_1D(in, out, size);
    return;
}

DLL_API_EXPORT void fft2d_f32 (const float* RESTRICT in, float* RESTRICT scratch, float* RESTRICT out, ptrdiff_t sizeX, ptrdiff_t sizeY)
{
    if (nullptr != in && nullptr != out && nullptr != scratch)
        FourierTransform::mixed_radix_fft_2D(in, scratch, out, sizeX, sizeY);
    return;
}

DLL_API_EXPORT void fft2d_f64 (const double* RESTRICT in, double* RESTRICT scratch, double* RESTRICT out, ptrdiff_t sizeX, ptrdiff_t sizeY)
{
    if (nullptr != in && nullptr != out && nullptr != scratch)
        FourierTransform::mixed_radix_fft_2D(in, scratch, out, sizeX, sizeY);
    return;
}


DLL_API_EXPORT void ifft_f32 (const float* in, float* out, ptrdiff_t size)
{
    if (nullptr != in && nullptr != out && size >= 2 && in != out)
        FourierTransform::mixed_radix_ifft_1D(in, out, size);

    return;
}

DLL_API_EXPORT void ifft_f64 (const double* in, double* out, ptrdiff_t size)
{
    if (nullptr != in && nullptr != out && size >= 2 && in != out)
        FourierTransform::mixed_radix_ifft_1D(in, out, size);
    return;
}


DLL_API_EXPORT void ifft2d_f32 (const float* RESTRICT in, float* RESTRICT scratch, float* RESTRICT out, ptrdiff_t sizeX, ptrdiff_t sizeY)
{
    if (nullptr != in && nullptr != out && nullptr != scratch)
        FourierTransform::mixed_radix_ifft_2D(in, scratch, out, sizeX, sizeY);
    return;
}

DLL_API_EXPORT void ifft2d_f64 (const double* RESTRICT in, double* RESTRICT scratch, double* RESTRICT out, ptrdiff_t sizeX, ptrdiff_t sizeY)
{
    if (nullptr != in && nullptr != out && nullptr != scratch)
        FourierTransform::mixed_radix_ifft_2D(in, scratch, out, sizeX, sizeY);
    return;
}


DLL_API_EXPORT void dct2d_f32 (const float* RESTRICT in, float* RESTRICT scratch, float* RESTRICT out, ptrdiff_t sizeX, ptrdiff_t sizeY)
{
    if (nullptr != in && nullptr != out && nullptr != scratch)
        FourierTransform::dct_2D (in, scratch, out, sizeX, sizeY);
    return;
}

DLL_API_EXPORT void dct2d_f64 (const double* RESTRICT in, double* RESTRICT scratch, double* RESTRICT out, ptrdiff_t sizeX, ptrdiff_t sizeY)
{
    if (nullptr != in && nullptr != out && nullptr != scratch)
        FourierTransform::dct_2D (in, scratch, out, sizeX, sizeY);
    return;
}


DLL_API_EXPORT void idct2d_f32(const float* RESTRICT in, float* RESTRICT scratch, float* RESTRICT out, ptrdiff_t sizeX, ptrdiff_t sizeY)
{
    if (nullptr != in && nullptr != out && nullptr != scratch)
        FourierTransform::idct_2D (in, scratch, out, sizeX, sizeY);
    return;
}

DLL_API_EXPORT void idct2d_f64(const double* RESTRICT in, double* RESTRICT scratch, double* RESTRICT out, ptrdiff_t sizeX, ptrdiff_t sizeY)
{
    if (nullptr != in && nullptr != out && nullptr != scratch)
        FourierTransform::idct_2D (in, scratch, out, sizeX, sizeY);
    return;
}


DLL_API_EXPORT void dct2d_f32_8x8 (const float* RESTRICT in, float* RESTRICT scratch, float* RESTRICT out)
{
    FourierTransform::dct_2D_8x8 (in, scratch, out);
    return;
}

DLL_API_EXPORT void dct2d_f64_8x8 (const double* RESTRICT in, double* RESTRICT scratch, double* RESTRICT out)
{
    FourierTransform::dct_2D_8x8 (in, scratch, out);
    return;
}


DLL_API_EXPORT void idct2d_f32_8x8 (const float* RESTRICT in, float* RESTRICT scratch, float* RESTRICT out)
{
    FourierTransform::idct_2D_8x8 (in, scratch, out);
    return;
}

DLL_API_EXPORT void idct2d_f64_8x8 (const double* RESTRICT in, double* RESTRICT scratch, double* RESTRICT out)
{
    FourierTransform::idct_2D_8x8 (in, scratch, out);
    return;
}
