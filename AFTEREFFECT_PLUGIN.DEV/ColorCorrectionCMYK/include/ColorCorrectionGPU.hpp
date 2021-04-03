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
	unsigned short w; /* ALLPHA	*/
}Pixel16, *PPixel16;



#ifdef __cplusplus
extern "C" {
#endif

	void ColorCorrection_CMYK_CUDA
	(
		float* inBuf,
		float* outBuf,
		int destPitch,
		int srcPitch,
		int	is16f,
		int width,
		int height,
		float C,
		float M,
		float Y,
		float K
	) noexcept;

	void ColorCorrection_RGB_CUDA
	(
		float* inBuf,
		float* outBuf,
		int destPitch,
		int srcPitch,
		int	is16f,
		int width,
		int height,
		float R,
		float G,
		float B
	) noexcept;

#ifdef __cplusplus
}
#endif