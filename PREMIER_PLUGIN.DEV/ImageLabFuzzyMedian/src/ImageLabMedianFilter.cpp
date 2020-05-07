#include "ImageLabFuzzyMedian.h"


bool median_filter_BGRA_4444_8u_frame(
		const	csSDK_uint32* __restrict srcBuf,
		csSDK_uint32*         __restrict dstBuf,
		const	csSDK_int32& height,
		const	csSDK_int32& width,
		const	csSDK_int32& linePitch,
		AlgMemStorage&		 algMem,
		const   csSDK_int16& kernelRadius)
{
	/* get total size of kernel */
	const csSDK_int32 kernelSize = make_odd(kernelRadius * 2);

	/* check if kernel size is good enougth for apply to image */
	if (kernelSize > height || kernelSize > width)
		return false;

	const csSDK_int32 medianSum = (kernelSize * kernelSize) >> 1;


	return false;
}


bool median_filter_ARGB_4444_8u_frame
(
	const csSDK_uint32* __restrict srcPix,
	csSDK_uint32*       __restrict dstPix,
	const	csSDK_int32& height,
	const	csSDK_int32& width,
	const	csSDK_int32& linePitch,
	AlgMemStorage&		 algMem,
	const   csSDK_int16& kernelRadius)
{
	/* get total size of kernel */
	const csSDK_int32 kernelSize = make_odd(kernelRadius * 2);

	/* check if kernel size is good enougth for apply to image */
	if (kernelSize > height || kernelSize > width)
		return false;

	const csSDK_int32 medianSum = (kernelSize * kernelSize) >> 1;

	return false;
}

