#include "ImageLabFuzzyMedian.h"
#include <assert.h> 

bool median_filter_BGRA_4444_8u_frame(
		const	csSDK_uint32* __restrict srcBuf,
		csSDK_uint32* const   __restrict dstBuf,
		const	csSDK_int32& height,
		const	csSDK_int32& width,
		const	csSDK_int32& linePitch,
		AlgMemStorage&		 algMem,
		const   csSDK_int16& kernelRadius)
{
	if (nullptr == srcBuf || nullptr == dstBuf || algMem.strSizeOf != sizeof(algMem))
		return false;

	CACHE_ALIGN HistElem luc[4][16];
	int i, j, k;
	csSDK_int32 R, G, B;
	bool Ret = true;

	/* compute number of stripes for Histogram computation */
	constexpr csSDK_int32 SizeOfHistogram = static_cast<csSDK_int32>(sizeof(algMem.h[0]));

	const csSDK_int32 doubleRadius = kernelRadius << 1;
	const double stripes_d = std::ceil(static_cast<double>(width - doubleRadius) / static_cast<double>(used_mem_size / SizeOfHistogram  - doubleRadius));
	const double stripe_size_d = std::ceil(static_cast<double>(width + stripes_d * doubleRadius - doubleRadius) / stripes_d);
	const csSDK_int32 stripes = algMem.stripeNum = static_cast<csSDK_int32>(stripes_d);
	const csSDK_int32 stripe_size = algMem.stripeSize = static_cast<csSDK_int32>(stripe_size_d);
	const csSDK_int32 stripe_size_minus_dradius = stripe_size - doubleRadius;

	for (i = 0; i < width; i += stripe_size - doubleRadius)
	{
		const csSDK_int32 stripe = ((i + stripe_size_minus_dradius >= width || width - (i + stripe_size_minus_dradius) < doubleRadius + 1) ? (width - i) : stripe_size);
		const csSDK_uint32* __restrict srcPix = srcBuf + i;
		csSDK_uint32* const __restrict dstPix = dstBuf + i;

		const bool padLeft  = (0 == i) ? true : false;
		const bool padRight = ((width - i) == stripe) ? true : false;

		/* check memory storage size [because size_fine lineary dependent from size_coarse -
		   for decrease number of check operations lets check only size_coarse] */
		if (size_coarse < ((stripe << 4) * sizeof(HistElem)))
		{
			assert(size_coarse < ((stripe << 4) * sizeof(HistElem)));
			Ret = false;
			break; /* abnormal exit - memory storage for hold histograms too small */
		}

		// in first cleanup memory storages from data from previous data slice
		memset(algMem.pFine, 0, size_fine);
		memset(algMem.pCoarse, 0, size_coarse);
		memset(algMem.h, 0, sizeof(algMem.h));
		memset(luc, 0, sizeof(luc));

		/* PROCESS IMAGE BUFFER */

		/* First row initialization */
		for (j = 0; j < stripe; j++)
		{
			R = static_cast<csSDK_int32> (srcPix[j] & 0x000000FFu);
			G = static_cast<csSDK_int32>((srcPix[j] & 0x0000FF00u) >> 8);
			B = static_cast<csSDK_int32>((srcPix[j] & 0x00FF0000u) >> 16);

			/* init COARSE level */
			algMem.pCoarse[16 *               j  + (R >> 4)] += kernelRadius + 1;
			algMem.pCoarse[16 * (stripe     + j) + (G >> 4)] += kernelRadius + 1;
			algMem.pCoarse[16 * (stripe * 2 + j) + (B >> 4)] += kernelRadius + 1;

			/* init FINE level */

		} /* for (k = 0; k < stripe; k++) */


		if ((width - i) == stripe)
			break;
	} /* for (i = 0; i < width; i += stripe_size - doubleRadius) */

	return Ret;
}



bool median_filter_ARGB_4444_8u_frame(
	const	csSDK_uint32* __restrict srcPix,
	csSDK_uint32* const   __restrict dstPix,
	const	csSDK_int32& height,
	const	csSDK_int32& width,
	const	csSDK_int32& linePitch,
	AlgMemStorage&		 algMem,
	const   csSDK_int16& kernelRadius)
{
	if (nullptr == srcPix || nullptr == dstPix || algMem.strSizeOf != sizeof(algMem))
		return false;

	CACHE_ALIGN HistElem luc[4][16];
	int i;


	return true;
}


