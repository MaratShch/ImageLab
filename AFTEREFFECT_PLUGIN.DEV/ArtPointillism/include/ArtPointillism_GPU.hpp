#ifndef __IMAGE_LAB_ART_POINTILISM_GPU_ACCELERATOR_DEFINITIONS_ALGO__
#define __IMAGE_LAB_ART_POINTILISM_GPU_ACCELERATOR_DEFINITIONS_ALGO__

#include <cuda_runtime.h>
#include "ImageLabCUDA.hpp"
#include "ArtPointillismControl.hpp"

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
	unsigned short w; /* ALPHA	*/
}Pixel16, *PPixel16;

constexpr size_t Pixel16Size = sizeof(Pixel16);

CUDA_KERNEL_CALL
void ArtPointillism_CUDA
(
    const float* RESTRICT inBuffer, // source (input) buffer
    float* RESTRICT outBuffer,      // destination (output) buffer
    int srcPitch,                   // source buffer pitch in pixels 
    int dstPitch,                   // destination buffer pitch in pixels
    int width,                      // horizontal image size in pixels
    int height,                     // vertical image size in lines
    const PontillismControls* algoGpuParams, // algorithm controls
    cudaStream_t stream = 0
);


#endif // __IMAGE_LAB_ART_POINTILISM_GPU_ACCELERATOR_DEFINITIONS_ALGO__