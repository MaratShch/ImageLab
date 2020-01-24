#include "ImageLabFuzzyMedian.h"

bool median_filter_BGRA_4444_8u_frame(
		const	csSDK_uint32* __restrict srcPix,
		    	csSDK_uint32* __restrict dstPix,
		const	csSDK_int32& height,
		const	csSDK_int32& width,
		const	csSDK_int32& linePitch)
{
	if (nullptr == srcPix || nullptr == dstPix)
		return false;

	return true;
}



bool median_filter_ARGB_4444_8u_frame(
	const	csSDK_uint32* __restrict srcPix,
	csSDK_uint32* __restrict dstPix,
	const	csSDK_int32& height,
	const	csSDK_int32& width,
	const	csSDK_int32& linePitch)
{
	if (nullptr == srcPix || nullptr == dstPix)
		return false;

	return true;
}
