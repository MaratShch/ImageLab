#include <string>
#include <windows.h>
#include "CompileTimeUtils.hpp"
#include "CommonAuxPixFormat.hpp"
#include "ImageLabDenoiseUtils.hpp"

typedef int  (WINAPI *procLib_compute_prime) (int imgSize, int arraySize, int* ptr);
typedef void (WINAPI *procLib_fft_f32)    (const float*  in, float*  out, int size);
typedef void (WINAPI *procLib_fft2d_f32)  (const float*  in, float*  scratch, float*  out, int sizeX, int sizeY);
typedef void (WINAPI *procLib_ifft_f32)   (const float*  in, float*  out, int size);
typedef void (WINAPI* procLib_ifft2d_f32) (const float*  in, float*  scratch, float*  out, int sizeX, int sizeY);


struct ProcLibHandle
{
    procLib_compute_prime compute_prime;
    procLib_fft_f32     fft_f32;
    procLib_ifft_f32    ifft_f32;
    procLib_fft2d_f32   fft2d_f32;
    procLib_ifft2d_f32  ifft2d_f32;
};

static ProcLibHandle fftLibHndl{};
constexpr size_t fftLibHndlSize = sizeof(fftLibHndl);
static HMODULE hLib = nullptr;

bool LoadProcLibDLL (PF_InData* in_data)
{
    A_char pluginFullPath[AEFX_MAX_PATH]{};
    PF_Err extErr = PF_GET_PLATFORM_DATA(PF_PlatData_EXE_FILE_PATH_DEPRECATED, &pluginFullPath);
    bool err = false;

    memset (&fftLibHndl, 0, fftLibHndlSize);

    if (PF_Err_NONE == extErr && 0 != pluginFullPath[0])
    {
        const std::string dllName{ "\\ImageLabProcLib.dll" };
        const std::string aexPath{ pluginFullPath };
        const std::string::size_type pos = aexPath.rfind("\\", aexPath.length());
        const std::string dllPath = aexPath.substr(0, pos) + dllName;

        // Load Memory Management DLL
        hLib = ::LoadLibraryEx (dllPath.c_str(), NULL, LOAD_LIBRARY_SEARCH_DEFAULT_DIRS);
        if (NULL != hLib)
        {
            DisableThreadLibraryCalls(hLib);
            fftLibHndl.compute_prime= reinterpret_cast<procLib_compute_prime>   (GetProcAddress(hLib, __TEXT("compute_prime")));
            fftLibHndl.fft_f32      = reinterpret_cast<procLib_fft_f32>         (GetProcAddress(hLib, __TEXT("fft_f32")));
            fftLibHndl.ifft_f32     = reinterpret_cast<procLib_ifft_f32>        (GetProcAddress(hLib, __TEXT("ifft_f32")));
            fftLibHndl.fft2d_f32    = reinterpret_cast<procLib_fft2d_f32>       (GetProcAddress(hLib, __TEXT("fft2d_f32")));
            fftLibHndl.ifft2d_f32   = reinterpret_cast<procLib_ifft2d_f32>      (GetProcAddress(hLib, __TEXT("ifft2d_f32")));
        }
    }

    return true;
}

void UnloadProcLibDll (void)
{
    memset(&fftLibHndl, 0, fftLibHndlSize);

    ::FreeLibrary(hLib);
    hLib = nullptr;

    return;
}


int proc_compute_prime (int imgSize, int arraySize, int* ptr)
{
    return (nullptr != hLib && nullptr != fftLibHndl.compute_prime) ? fftLibHndl.compute_prime(imgSize, arraySize, ptr) : 0;
}

void proc_fft_f32 (const float* in, float* out, int size)
{
    if (nullptr != hLib && nullptr != fftLibHndl.fft_f32)
        fftLibHndl.fft_f32 (in, out, size);
    return;
}

void proc_ifft_f32 (const float* in, float* out, int size)
{
    if (nullptr != hLib && nullptr != fftLibHndl.ifft_f32)
        fftLibHndl.ifft_f32 (in, out, size);
    return;
}

void proc_fft2d_f32 (const float* in, float* scratch, float* out, int sizeX, int sizeY)
{
    if (nullptr != hLib && nullptr != fftLibHndl.fft2d_f32)
        fftLibHndl.fft2d_f32 (in, scratch, out, sizeX, sizeY);
    return;
}

void proc_ifft2d_f32 (const float* in, float* scratch, float* out, int sizeX, int sizeY)
{
    if (nullptr != hLib && nullptr != fftLibHndl.ifft2d_f32)
        fftLibHndl.ifft2d_f32 (in, scratch, out, sizeX, sizeY);
    return;
}



size_t utils_get_compute_fft_buffer_size (A_long sizeX, A_long sizeY) noexcept
{
    size_t memSize = 0u;

    return memSize;
}
