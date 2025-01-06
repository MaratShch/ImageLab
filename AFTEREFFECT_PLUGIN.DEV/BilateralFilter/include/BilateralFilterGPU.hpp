#pragma once

#include <cuda_runtime.h>

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
void BilateralFilter_CUDA
(
	float* inBuf,
	float* outBuf,
	int destPitch,
	int srcPitch,
	int	is16f,
	int width,
	int height,
	int fRadius,
    float fSigma
);

CUDA_KERNEL_CALL
bool LoadGpuMesh_CUDA
(
    const float* hostMesh
);

constexpr int gpuMaxFilterRadius = 10;
constexpr int gpuMaxWindowSize = 2 * gpuMaxFilterRadius + 1;
constexpr int meshCenter = gpuMaxWindowSize * gpuMaxFilterRadius + gpuMaxFilterRadius;
constexpr int gpuMaxMeshSize = gpuMaxWindowSize * gpuMaxWindowSize; /* filter window size with maximal radius equal to 10 */

constexpr float fSigmaMin = 5.f;
constexpr float fSigmaMax = 20.f;