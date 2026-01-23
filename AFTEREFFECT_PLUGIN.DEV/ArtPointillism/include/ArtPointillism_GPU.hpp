#ifndef __IMAGE_LAB_ART_POINTILISM_GPU_ACCELERATOR_DEFINITIONS_ALFO__
#define __IMAGE_LAB_ART_POINTILISM_GPU_ACCELERATOR_DEFINITIONS_ALFO__

#include <cuda_runtime.h>
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
CUDA_KERNEL_CALL
void ArtPointillism_CUDA
(
    float* inBuffer,
    float* outBuffer,
    int destPitch,
    int srcPitch,
    int is16f,
    int width,
    int height,
    const PontillismControls& algoGpuParams
);


#endif // __IMAGE_LAB_ART_POINTILISM_GPU_ACCELERATOR_DEFINITIONS_ALFO__