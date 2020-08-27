#pragma once

void ImageGradientVertical_BGRA_4444_8u
(
	const csSDK_uint32* __restrict pSrcBuf,
	float*   __restrict pDstBuf,
	const csSDK_int32&  width,
	const csSDK_int32&  height,
	const csSDK_int32&  linePitch,
	const bool&         isBT709
);

void ImageGradientHorizontal_BGRA_4444_8u
(
	const csSDK_uint32* __restrict pSrcBuf,
	float*   __restrict pDstBuf,
	const csSDK_int32&  width,
	const csSDK_int32&  height,
	const csSDK_int32&  linePitch,
	const bool&         isBT709
);