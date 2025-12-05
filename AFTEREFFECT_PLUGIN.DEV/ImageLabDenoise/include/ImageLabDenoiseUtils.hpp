#ifndef __IMAGE_LAB_DENOISE_UTILS__
#define __IMAGE_LAB_DENOISE_UTILS__

#include <cstddef>
#include <cstdint>
#include "CommonAdobeAE.hpp"

constexpr std::size_t maxTwiddleFactor = 24;

struct strProcFftHandler
{
    std::size_t strSizeOf;
    uint32_t sizeX;
    uint32_t sizeY;
    uint32_t twiddleXSize;
    uint32_t twiddleYSize;
    int32_t  twiddleX[maxTwiddleFactor];
    int32_t  twiddleY[maxTwiddleFactor];
};
using fftTwiddleStr = struct strProcFftHandler;

typedef struct strFftHandle
{
    AEGP_PluginID id;
    A_long valid;
    A_long isFlat;
    fftTwiddleStr twiddleStr;
};

constexpr size_t strProcFftHandlerSize = sizeof(strProcFftHandler);
constexpr size_t strFftHandleSize = sizeof(strFftHandle);

bool LoadProcLibDLL (PF_InData* in_data = nullptr);
void UnloadProcLibDll (void);

// ProcLib API's
int proc_compute_prime (int imgSize, int arraySize, int* ptr);
void proc_fft_f32 (const float* in, float* out, int size);
void proc_ifft_f32 (const float* in, float* out, int size);
void proc_fft2d_f32 (const float* in, float* scratch, float* out, int sizeX, int sizeY);
void proc_ifft2d_f32 (const float* in, float* scratch, float* out, int sizeX, int sizeY);

std::size_t utils_get_compute_fft_buffer_size (A_long sizeX, A_long sizeY) noexcept;



#endif // __IMAGE_LAB_DENOISE_UTILS__