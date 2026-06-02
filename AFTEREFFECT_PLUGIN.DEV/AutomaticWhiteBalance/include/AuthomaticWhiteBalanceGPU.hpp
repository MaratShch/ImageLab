#ifndef __IMAGE_LAB_AUTHOMATIC_WHITE_BALANCE_GPU_HANDLERS__
#define __IMAGE_LAB_AUTHOMATIC_WHITE_BALANCE_GPU_HANDLERS__

#include <cuda_runtime.h>
#include "AlgoControl.hpp"
#include "AlgCommonEnums.hpp"
#include "ColorTransformMatrix.hpp"

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

constexpr float algAWBepsilon = 0.00001f;

CUDA_KERNEL_CALL
void ImageLabAWB32_CUDA
(
    const float*  RESTRICT inBuffer,
    float*  RESTRICT       outBuffer,
    int                    srcPitch,    // BGRA pixels/row
    int                    dstPitch,    // BGRA pixels/row
    int                    width,
    int                    height,
    const AlgoControls* RESTRICT algoGpuParams,
    cudaStream_t           stream
);

CUDA_KERNEL_CALL
void ImageLabAWB16_CUDA
(
    const float*  RESTRICT inBuffer,
    float*  RESTRICT       outBuffer,
    int                    srcPitch,
    int                    dstPitch,
    int                    width,
    int                    height,
    const AlgoControls* RESTRICT algoGpuParams,
    cudaStream_t           stream
);


#endif // __IMAGE_LAB_AUTHOMATIC_WHITE_BALANCE_GPU_HANDLERS__