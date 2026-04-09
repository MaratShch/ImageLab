#ifndef __IMAGE_LAB_ART_POINTILISM_GPU_ACCELERATOR_DEFINITIONS_ALGO__
#define __IMAGE_LAB_ART_POINTILISM_GPU_ACCELERATOR_DEFINITIONS_ALGO__

#include <cuda_runtime.h>
#include "ImageLabCUDA.hpp"
#include "PaintAlgoContols.hpp"

#ifdef __NVCC__
 /* Put here device specific includes files */
 #include <cuda_fp16.h>
#endif

#ifdef __cplusplus
extern "C" {
#endif
 #define CUDA_KERNEL_CALL extern "C"
#ifdef __cplusplus
}
#endif

typedef struct Pixel16
{
    unsigned short x; /* BLUE	*/
    unsigned short y; /* GREEN	*/
    unsigned short z; /* RED	*/
    unsigned short w; /* ALLPHA	*/
}Pixel16, *PPixel16;


CUDA_KERNEL_CALL
void ArtPaint_CUDA
(
    const void* RESTRICT inBuffer, // source (input) buffer
    void* RESTRICT outBuffer,      // destination (output) buffer
    int srcPitch,                   // source buffer pitch in pixels 
    int dstPitch,                   // destination buffer pitch in pixels
    int width,                      // horizontal image size in pixels
    int height,                     // vertical image size in lines
    const AlgoControls* algoGpuParams, // algorithm controls
    int frameCounter,               // frame counter (zero based)
    bool isFloat16,                 // // true - compute in float16, false - compute in float32
    cudaStream_t stream = static_cast<cudaStream_t>(0)
);


#endif // __IMAGE_LAB_ART_POINTILISM_GPU_ACCELERATOR_DEFINITIONS_ALGO__