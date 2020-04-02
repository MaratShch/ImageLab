#pragma once

#include "PrSDKEffect.h"
#include "PrSDKPixelFormat.h"
#include "PrSDKPPixSuite.h"
#include "PrSDKPixelFormatSuite.h"
#include "PrSDKSequenceInfoSuite.h"
#include "SDK_File.h"

/* color convert from RGB 8 bits per band to HSV (save only V band) */
void convert_rgb_to_hsv_4444_BGRA8u
(
	const csSDK_uint32* __restrict pSrc,
	float* __restrict pDstV,
	csSDK_int32 width,
	csSDK_int32 height,
	csSDK_int32 linePitch
);

void convert_hsv_to_rgb_4444_BGRA8u
(
	const csSDK_uint32* __restrict pSrc, /* original source image used only for get ALPHA value for each pixel */
	const float*  __restrict pHSV, /* buffer layout: H, S, V*/
	csSDK_uint32* __restrict pDst,
	csSDK_int32 width,
	csSDK_int32 height,
	csSDK_int32 linePitch
);