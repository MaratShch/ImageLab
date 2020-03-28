#include "ImageLabFuzzyMedian.h"
#include <assert.h> 



bool fuzzy_median_filter_BGRA_4444_8u_frame
(
	const csSDK_uint32* __restrict srcBuf,
	csSDK_uint32*       __restrict dstBuf,
	const	csSDK_int32& height,
	const	csSDK_int32& width,
	const	csSDK_int32& linePitch,
	const   csSDK_int16& kernelRadius
)
{
	return true;
}


bool fuzzy_median_filter_ARGB_4444_8u_frame
(
	const csSDK_uint32* __restrict srcPix,
	csSDK_uint32*       __restrict dstPix,
	const	csSDK_int32& height,
	const	csSDK_int32& width,
	const	csSDK_int32& linePitch,
	const   csSDK_int16& kernelRadius
)
{
	return true;
}

