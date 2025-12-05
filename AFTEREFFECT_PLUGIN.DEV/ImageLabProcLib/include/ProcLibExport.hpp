#ifndef __IMAGE_LAB2_PROCESSING_LIB_EXPORT_APIS__
#define __IMAGE_LAB2_PROCESSING_LIB_EXPORT_APIS__

#include "LibExport.hpp"

// FFT module
DLL_API_EXPORT int compute_prime (int imgSize, int arraySize, int* ptr);

DLL_API_EXPORT void fft_f32  (const float*  in, float*  out, int size);
DLL_API_EXPORT void fft_f64  (const double* in, double* out, int size);
DLL_API_EXPORT void fft2d_f32(const float*  in, float*  scratch, float*  out, int sizeX, int sizeY);
DLL_API_EXPORT void fft2d_f64(const double* in, double* scratch, double* out, int sizeX, int sizeY);

DLL_API_EXPORT void ifft_f32  (const float*  in, float*  out, int size);
DLL_API_EXPORT void ifft_f64  (const double* in, double* out, int size);
DLL_API_EXPORT void ifft2d_f32(const float*  in, float*  scratch, float*  out, int sizeX, int sizeY);
DLL_API_EXPORT void ifft2d_f64(const double* in, double* scratch, double* out, int sizeX, int sizeY);


#endif // __IMAGE_LAB2_PROCESSING_LIB_EXPORT_APIS__
