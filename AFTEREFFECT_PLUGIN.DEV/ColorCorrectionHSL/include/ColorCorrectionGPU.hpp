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

#ifndef CLAMP_VALUE
#define CLAMP_VALUE(val, val_min, val_max) \
	(((val) < (val_min)) ? (val_min) : (((val) > (val_max)) ? (val_max) : (val)))
#endif

#ifdef __cplusplus
extern "C" {
#endif

	void ColorCorrection_HSL_CUDA
	(
		float* inBuf,
		float* outBuf,
		int destPitch,
		int srcPitch,
		int	is16f,
		int width,
		int height
	);

	void ColorCorrection_HSV_CUDA
	(
		float* inBuf,
		float* outBuf,
		int destPitch,
		int srcPitch,
		int	is16f,
		int width,
		int height
	);

	void ColorCorrection_HSI_CUDA
	(
		float* inBuf,
		float* outBuf,
		int destPitch,
		int srcPitch,
		int	is16f,
		int width,
		int height
	);

	void ColorCorrection_HSP_CUDA
	(
		float* inBuf,
		float* outBuf,
		int destPitch,
		int srcPitch,
		int	is16f,
		int width,
		int height
	);

	void ColorCorrection_HSLuv_CUDA
	(
		float* inBuf,
		float* outBuf,
		int destPitch,
		int srcPitch,
		int	is16f,
		int width,
		int height
	);

	void ColorCorrection_HPLuv_CUDA
	(
		float* inBuf,
		float* outBuf,
		int destPitch,
		int srcPitch,
		int	is16f,
		int width,
		int height
	);

#ifdef __cplusplus
}
#endif