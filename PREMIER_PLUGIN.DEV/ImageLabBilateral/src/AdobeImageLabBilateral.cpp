#include "ImageLabBilateral.h"
#include <windows.h>

CACHE_ALIGN static float gMesh[maxWinSize][maxWinSize] = { 0 };

// define color space conversion matrix's
CACHE_ALIGN constexpr float coeff_RGB2YUV[2][9] =
{
	// BT.601
	{
		0.299000f,  0.587000f,  0.114000f,
	   -0.168736f, -0.331264f,  0.500000f,
		0.500000f, -0.418688f, -0.081312f
	},

	// BT.709
	{
		0.212600f,   0.715200f,  0.072200f,
	   -0.114570f,  -0.385430f,  0.500000f,
		0.500000f,  -0.454150f, -0.045850f
	}
};

CACHE_ALIGN constexpr float coeff_YUV2RGB[2][9] =
{
	// BT.601
	{
		1.000000f,  0.000000f,  1.407500f,
		1.000000f, -0.344140f, -0.716900f,
		1.000000f,  1.779000f,  0.000000f
	},

	// BT.709
	{
		1.000000f,  0.00000000f,  1.5748021f,
		1.000000f, -0.18732698f, -0.4681240f,
		1.000000f,  1.85559927f,  0.0000000f
	}
};


void gaussian_weights(const float sigma, const int radius /* radius size in range of 3 to 10 */)
{
	int i, j;
	int x, y;

	const float divider = 2.00f * sigma * sigma;

	__VECTOR_ALIGNED__
	for (y = -radius, j = 0; j < maxWinSize; j++, y++)
	{
		for (x = -radius, i = 0; i < maxWinSize; i++, x++)
		{
			const float dSum = static_cast<float>((x * x) + (y * y));
			gMesh[j][i] = aExp(-dSum / divider);
		}
	}

	return;
}


bool process_VUYA_4444_8u_frame (const VideoHandle theData, const int radius)
{
#if !defined __INTEL_COMPILER 
	_MM_SET_FLUSH_ZERO_MODE(_MM_FLUSH_ZERO_ON);
	_MM_SET_DENORMALS_ZERO_MODE(_MM_DENORMALS_ZERO_ON);
#endif

	CACHE_ALIGN float pF[maxWinSize][maxWinSize] = {};
	CACHE_ALIGN float pH[maxWinSize][maxWinSize] = {};

	prRect box = { 0 };

	// Get the frame dimensions
	((*theData)->piSuites->ppixFuncs->ppixGetBounds)((*theData)->destination, &box);

	// Calculate dimensions
	const csSDK_int32 height = box.bottom - box.top;
	const csSDK_int32 width = box.right - box.left;
	const csSDK_int32 rowbytes = ((*theData)->piSuites->ppixFuncs->ppixGetRowbytes)((*theData)->destination);

	const int lastLineIdx  = height - 1;
	const int lastPixelIdx = width - 1;
	const int linePitch = rowbytes >> 2;

	constexpr float sigma = 0.10f;
	constexpr float divider = 2.00f * sigma * sigma;

	// Create copies of pointer to the source, destination frames
	const csSDK_uint32* __restrict srcPix = reinterpret_cast<csSDK_uint32* __restrict>(((*theData)->piSuites->ppixFuncs->ppixGetPixels)((*theData)->source));
	      csSDK_uint32* __restrict dstPix = reinterpret_cast<csSDK_uint32* __restrict>(((*theData)->piSuites->ppixFuncs->ppixGetPixels)((*theData)->destination));


	int i, j;
	int k, l;
	int pixOffset = 0;
	float fY, dY, dot, normF, bSum;
	unsigned int Y;
	unsigned int finalY;
	float Yref;
	csSDK_uint32 dstPixel;

	for (j = 0; j < height; j++)
	{
		for (i = 0; i < width; i++)
		{
			// define processing window coordinates
			const int iMin = MAX(i - radius, 0);
			const int iMax = MIN(i + radius, lastPixelIdx);
			const int jMin = MAX(j - radius, 0);
			const int jMax = MIN(j + radius, lastLineIdx);

			// define process window sizes
			const int iDiff = (iMax - iMin) + 1;
			const int jDiff = (jMax - jMin) + 1;

			pixOffset = j * linePitch + i;
			const csSDK_uint32 srcPixel = srcPix[pixOffset];

			Yref = (static_cast<float>((srcPixel & 0x00FF0000u) >> 16)) / 255.0f;

			// compute Gaussian intensity weights
			for (k = 0; k < jDiff; k++)
			{
				// first pixel position in specified line in filter window
				const int pixPos = (jMin + k) * linePitch + iMin;

				for (l = 0; l < iDiff; l++)
				{
					fY = (static_cast<float>((srcPix[pixPos + l] & 0x00FF0000u) >> 16)) / 255.0f;
					dY = fY - Yref;
					pH[k][l] = aExp(-(dY * dY) / divider);
				} // for (m = 0; m < iDiff; m++)

			} // for (k = 0; k < jDiff; k++)


			// calculate Bilateral Filter responce
			normF = bSum = 0.0f;

			int iIdx = 0;
			int jIdx = jMin - j + radius;

			for (k = 0; k < jDiff; k++)
			{
				iIdx = iMin - i + radius;

				__VECTOR_ALIGNED__
				for (l = 0; l < iDiff; l++)
				{
					pF[k][l] = pH[k][l] * gMesh[jIdx][iIdx];
					normF += pF[k][l];
					iIdx++;
				}
				jIdx++;
			}

			for (k = 0; k < jDiff; k++)
			{
				const int kIdx = (jMin + k) * linePitch + iMin;
				for (l = 0; l < iDiff; l++)
				{
					Y = static_cast<int>((srcPix[kIdx + l] & 0x00FF0000) >> 16);
					fY = (static_cast<float>(Y)) / 255.0f;
					bSum += (pF[k][l] * fY);
				}
			}

			// compute destination pixel
			finalY = CLAMP_U8(static_cast<unsigned int>(255.0f * bSum / normF));

			// pack corrected Luma value
			dstPixel = (srcPixel & 0xFF00FFFFu) | (finalY << 16);

			// save corrected pixel in destination buffer
			dstPix[pixOffset] = dstPixel;

		} // for (i = 0; i < width; i++)

	} // for (j = 0; j < height; j++)

	return true;
}


bool process_VUYA_4444_32f_frame(const VideoHandle theData, const int radius)
{
#if !defined __INTEL_COMPILER 
	_MM_SET_FLUSH_ZERO_MODE(_MM_FLUSH_ZERO_ON);
	_MM_SET_DENORMALS_ZERO_MODE(_MM_DENORMALS_ZERO_ON);
#endif

	CACHE_ALIGN float pF[maxWinSize][maxWinSize] = {};
	CACHE_ALIGN float pH[maxWinSize][maxWinSize] = {};

	prRect box = {};

	// Get the frame dimensions
	((*theData)->piSuites->ppixFuncs->ppixGetBounds)((*theData)->destination, &box);

	// Calculate dimensions
	const csSDK_int32 height = box.bottom - box.top;
	const csSDK_int32 width = box.right - box.left;
	const csSDK_int32 rowbytes = ((*theData)->piSuites->ppixFuncs->ppixGetRowbytes)((*theData)->destination);

	const int lastLineIdx = height - 1;
	const int lastPixelIdx = width - 1;
	const int linePitch = rowbytes >> 2;

	constexpr float sigma = 0.10f;
	constexpr float divider = 2.00f * sigma * sigma;

	// Create copies of pointer to the source, destination frames
	const float* __restrict srcBuf = reinterpret_cast<float* __restrict>(((*theData)->piSuites->ppixFuncs->ppixGetPixels)((*theData)->source));
		  float* __restrict dstBuf = reinterpret_cast<float* __restrict>(((*theData)->piSuites->ppixFuncs->ppixGetPixels)((*theData)->destination));

	int i, j;
	int k, l;
	float Yref = 0.f; /* source (non filtered) Luma value  */
	float dstY = 0.f; /* destination (filtered) Luma value */
	float normF, bSum;

	constexpr int pixelSize = 4; /* 4 float elements per single pixel */
	constexpr int offsetY = 2;   /* skip 2 (U and V) for get Y component */

	float fY, dY;

	for (j = 0; j < height; j++)
	{
		for (i = 0; i < width; i++)
		{
			// define processing window coordinates
			const int iMin = MAX(i - radius, 0);
			const int iMax = MIN(i + radius, lastPixelIdx);
			const int jMin = MAX(j - radius, 0);
			const int jMax = MIN(j + radius, lastLineIdx);

			// define process window sizes
			const int iDiff = (iMax - iMin) + 1;
			const int jDiff = (jMax - jMin) + 1;

			// offset of processed pixel
			const int pixIdx = (j * linePitch) + (i * pixelSize) + offsetY;

			Yref = srcBuf[pixIdx];

			// compute Gaussian intensity weights
			for (k = 0; k < jDiff; k++)
			{
				// first pixel position in specified line in filter window
				const int pixPos = (jMin + k) * linePitch + iMin * pixelSize;

				for (l = 0; l < iDiff; l++)
				{
					const int pixOffset = pixPos + l * pixelSize + offsetY;
					fY = srcBuf[pixOffset];
					dY = fY - Yref;
					pH[k][l] = aExp(-(dY * dY) / divider);
				} // for (m = 0; m < iDiff; m++)

			} // for (k = 0; k < jDiff; k++)

			// calculate Bilateral Filter responce
			normF = bSum = 0.0f;

			int iIdx = 0;
			int jIdx = jMin - j + radius;

			__VECTOR_ALIGNED__
			for (k = 0; k < jDiff; k++)
			{
				iIdx = iMin - i + radius;

				for (l = 0; l < iDiff; l++)
				{
					pF[k][l] = pH[k][l] * gMesh[jIdx][iIdx];
					normF += pF[k][l];
					iIdx++;
				}
				jIdx++;
			}

			for (k = 0; k < jDiff; k++)
			{
				const int kIdx = (jMin + k) * linePitch + iMin * pixelSize;
				for (l = 0; l < iDiff; l++)
				{
					const int lIdx = kIdx + l * pixelSize + offsetY;
					fY = srcBuf[lIdx];
					bSum += (pF[k][l] * fY);
				}
			}

			dstY = bSum / normF;

			__VECTOR_ALIGNED__
			// copy V and U components from source to destination "as is"
			dstBuf[pixIdx - 2] = srcBuf[pixIdx - 2];
			dstBuf[pixIdx - 1] = srcBuf[pixIdx - 1];
			// filter LUMA component
			dstBuf[pixIdx] = dstY;
			// copy ALPHA component from source to destination "as is"
			dstBuf[pixIdx + 1] = srcBuf[pixIdx + 1];
		}
	}

	return true;
}


bool process_BGRA_4444_8u_frame(const VideoHandle theData, const int radius)
{
#if !defined __INTEL_COMPILER 
	_MM_SET_FLUSH_ZERO_MODE(_MM_FLUSH_ZERO_ON);
	_MM_SET_DENORMALS_ZERO_MODE(_MM_DENORMALS_ZERO_ON);
#endif

	CACHE_ALIGN float pF[maxWinSize][maxWinSize] = {};
	CACHE_ALIGN float pH[maxWinSize][maxWinSize] = {};

	prRect box = { 0 };

	// Get the frame dimensions
	((*theData)->piSuites->ppixFuncs->ppixGetBounds)((*theData)->destination, &box);

	// Calculate dimensions
	const csSDK_int32 height = box.bottom - box.top;
	const csSDK_int32 width = box.right - box.left;
	const csSDK_int32 rowbytes = ((*theData)->piSuites->ppixFuncs->ppixGetRowbytes)((*theData)->destination);

	const int lastLineIdx = height - 1;
	const int lastPixelIdx = width - 1;
	const int linePitch = rowbytes >> 2;

	constexpr float sigma = 0.10f;
	constexpr float divider = 2.00f * sigma * sigma;

	// Create copies of pointer to the source, destination frames
	const csSDK_uint32* __restrict srcPix = reinterpret_cast<csSDK_uint32* __restrict>(((*theData)->piSuites->ppixFuncs->ppixGetPixels)((*theData)->source));
	      csSDK_uint32* __restrict dstPix = reinterpret_cast<csSDK_uint32* __restrict>(((*theData)->piSuites->ppixFuncs->ppixGetPixels)((*theData)->destination));
	const float*        __restrict pCoeffRGB2YUV = width < 800 ? coeff_RGB2YUV[0] /* BT601 */ : coeff_RGB2YUV[1] /* BT709 */;
	const float*        __restrict pCoeffYUV2RGB = width < 800 ? coeff_YUV2RGB[0] /* BT601 */ : coeff_YUV2RGB[1] /* BT709 */;

	int i, j;
	int k, l;
	int pixOffset = 0;
	float R, G, B, Y, dY, Yfinal;
	float Yref, Uref, Vref;
	float normF, bSum;

	csSDK_uint32 srcWinPix;
	csSDK_uint32 newR, newB, newG;

	for (j = 0; j < height; j++)
	{
		for (i = 0; i < width; i++)
		{
			// define processing window coordinates
			const int iMin = MAX(i - radius, 0);
			const int iMax = MIN(i + radius, lastPixelIdx);
			const int jMin = MAX(j - radius, 0);
			const int jMax = MIN(j + radius, lastLineIdx);

			// define process window sizes
			const int iDiff = (iMax - iMin) + 1;
			const int jDiff = (jMax - jMin) + 1;

			pixOffset = j * linePitch + i;
			const csSDK_uint32 srcPixel = srcPix[pixOffset];

			B = static_cast<float> (srcPixel & 0x000000FFu);
			G = static_cast<float>((srcPixel & 0x0000FF00u) >> 8);
			R = static_cast<float>((srcPixel & 0x00FF0000u) >> 16);
			const csSDK_uint32 A =  srcPixel & 0xFF000000u;

			Yref = R * pCoeffRGB2YUV[0] + G * pCoeffRGB2YUV[1] + B * pCoeffRGB2YUV[2];
			Uref = R * pCoeffRGB2YUV[3] + G * pCoeffRGB2YUV[4] + B * pCoeffRGB2YUV[5];
			Vref = R * pCoeffRGB2YUV[6] + G * pCoeffRGB2YUV[7] + B * pCoeffRGB2YUV[8];

			Yref /= 255.0f;

			// compute Gaussian intensity weights
			for (k = 0; k < jDiff; k++)
			{
				// first pixel position in specified line in filter window
				const int pixPos = (jMin + k) * linePitch + iMin;

				for (l = 0; l < iDiff; l++)
				{
					srcWinPix = srcPix[pixPos + l];

					B = static_cast<float> (srcWinPix & 0x000000FFu);
					G = static_cast<float>((srcWinPix & 0x0000FF00u) >> 8);
					R = static_cast<float>((srcWinPix & 0x00FF0000u) >> 16);

					Y = (R * pCoeffRGB2YUV[0] + G * pCoeffRGB2YUV[1] + B * pCoeffRGB2YUV[2]) / 255.0f;
					dY = Y - Yref;
					pH[k][l] = aExp(-(dY * dY) / divider);
				} // for (m = 0; m < iDiff; m++)

			} // for (k = 0; k < jDiff; k++)

			// calculate Bilateral Filter responce
			normF = bSum = 0.0f;

			int iIdx = 0;
			int jIdx = jMin - j + radius;

			__VECTOR_ALIGNED__
			for (k = 0; k < jDiff; k++)
			{
				iIdx = iMin - i + radius;

				for (l = 0; l < iDiff; l++)
				{
					pF[k][l] = pH[k][l] * gMesh[jIdx][iIdx];
					normF += pF[k][l];
					iIdx++;
				}
				jIdx++;
			}

			for (k = 0; k < jDiff; k++)
			{
				const int kIdx = (jMin + k) * linePitch + iMin;
				for (l = 0; l < iDiff; l++)
				{
					srcWinPix = srcPix[kIdx + l];

					B = static_cast<float> (srcWinPix & 0x000000FFu);
					G = static_cast<float>((srcWinPix & 0x0000FF00u) >> 8);
					R = static_cast<float>((srcWinPix & 0x00FF0000u) >> 16);

					Y = (R * pCoeffRGB2YUV[0] + G * pCoeffRGB2YUV[1] + B * pCoeffRGB2YUV[2]) / 255.0f;
					bSum += (pF[k][l] * Y);
				}
			}

			// compute destination pixel
			Yfinal = (255.0f * bSum) / normF;

			R = Yfinal * pCoeffYUV2RGB[0] + Uref * pCoeffYUV2RGB[1] + Vref * pCoeffYUV2RGB[2];
			G = Yfinal * pCoeffYUV2RGB[3] + Uref * pCoeffYUV2RGB[4] + Vref * pCoeffYUV2RGB[5];
			B = Yfinal * pCoeffYUV2RGB[6] + Uref * pCoeffYUV2RGB[7] + Vref * pCoeffYUV2RGB[8];

			newR = CLAMP_U8(static_cast<unsigned int>(R));
			newG = CLAMP_U8(static_cast<unsigned int>(G));
			newB = CLAMP_U8(static_cast<unsigned int>(B));

			dstPix[pixOffset] = A    |
						(newR << 16) |
						(newG << 8)  |
						 newB;

		} /* END: for (i = 0; i < width; i++) */

	} /* END: for (j = 0; j < height; j++) */

	return true;
}



bool process_BGRA_4444_16u_frame(const VideoHandle theData, const int radius)
{
#if !defined __INTEL_COMPILER 
	_MM_SET_FLUSH_ZERO_MODE(_MM_FLUSH_ZERO_ON);
	_MM_SET_DENORMALS_ZERO_MODE(_MM_DENORMALS_ZERO_ON);
#endif

	CACHE_ALIGN float pF[maxWinSize][maxWinSize] = {};
	CACHE_ALIGN float pH[maxWinSize][maxWinSize] = {};

	prRect box = { 0 };

	// Get the frame dimensions
	((*theData)->piSuites->ppixFuncs->ppixGetBounds)((*theData)->destination, &box);

	// Calculate dimensions
	const csSDK_int32 height = box.bottom - box.top;
	const csSDK_int32 width = box.right - box.left;
	const csSDK_int32 rowbytes = ((*theData)->piSuites->ppixFuncs->ppixGetRowbytes)((*theData)->destination);

	const int lastLineIdx = height - 1;
	const int lastPixelIdx = width - 1;
	const int linePitch = rowbytes >> 3; // because pixel defined as uint64 - lets add additional right shift to linepitch

	constexpr float sigma = 0.10f;
	constexpr float divider = 2.00f * sigma * sigma;

	// Create copies of pointer to the source, destination frames
	const csSDK_uint64* __restrict srcPix = reinterpret_cast<csSDK_uint64* __restrict>(((*theData)->piSuites->ppixFuncs->ppixGetPixels)((*theData)->source));
	      csSDK_uint64* __restrict dstPix = reinterpret_cast<csSDK_uint64* __restrict>(((*theData)->piSuites->ppixFuncs->ppixGetPixels)((*theData)->destination));
	const float*        __restrict pCoeffRGB2YUV = width < 800 ? coeff_RGB2YUV[0] /* BT601 */ : coeff_RGB2YUV[1] /* BT709 */;
	const float*        __restrict pCoeffYUV2RGB = width < 800 ? coeff_YUV2RGB[0] /* BT601 */ : coeff_YUV2RGB[1] /* BT709 */;

	int i, j;
	int k, l;
	int pixOffset = 0;
	float R, G, B, Y, dY, Yfinal = 0.0f;
	float Yref, Uref, Vref;
	float normF, bSum;

	csSDK_uint64 srcWinPix;
	csSDK_uint32 newR, newB, newG;

	for (j = 0; j < height; j++)
	{
		for (i = 0; i < width; i++)
		{
			// define processing window coordinates
			const int iMin = MAX(i - radius, 0);
			const int iMax = MIN(i + radius, lastPixelIdx);
			const int jMin = MAX(j - radius, 0);
			const int jMax = MIN(j + radius, lastLineIdx);

			// define process window sizes
			const int iDiff = (iMax - iMin) + 1;
			const int jDiff = (jMax - jMin) + 1;

			pixOffset = j * linePitch + i;
			const csSDK_uint64 refPix  = srcPix[pixOffset];

			B = (static_cast<float> (refPix & 0x000000000000FFFFu));
			G = (static_cast<float>((refPix & 0x00000000FFFF0000u) >> 16));
			R = (static_cast<float>((refPix & 0x0000FFFF00000000u) >> 32));
			const csSDK_uint64 A =   refPix & 0xFFFF000000000000u;

			Yref = (R * pCoeffRGB2YUV[0] + G * pCoeffRGB2YUV[1] + B * pCoeffRGB2YUV[2]) / 32768.0f;
			Uref = (R * pCoeffRGB2YUV[3] + G * pCoeffRGB2YUV[4] + B * pCoeffRGB2YUV[5]) / 32768.0f;
			Vref = (R * pCoeffRGB2YUV[6] + G * pCoeffRGB2YUV[7] + B * pCoeffRGB2YUV[8]) / 32768.0f;

			// compute Gaussian intensity weights
			for (k = 0; k < jDiff; k++)
			{
				// first pixel position in specified line in filter window
				const int pixPos = (jMin + k) * linePitch + iMin;

				for (l = 0; l < iDiff; l++)
				{
					srcWinPix = srcPix[pixPos + l];

					B = (static_cast<float> (srcWinPix & 0x000000000000FFFFu));
					G = (static_cast<float>((srcWinPix & 0x00000000FFFF0000u) >> 16));
					R = (static_cast<float>((srcWinPix & 0x0000FFFF00000000u) >> 32));

					Y = (R * pCoeffRGB2YUV[0] + G * pCoeffRGB2YUV[1] + B * pCoeffRGB2YUV[2]) / 32768.0f;
					dY = Y - Yref;
					pH[k][l] = aExp(-(dY * dY) / divider);
				} // for (m = 0; m < iDiff; m++)

			} // for (k = 0; k < jDiff; k++)

				// calculate Bilateral Filter responce
				normF = bSum = 0.0f;

				int iIdx = 0;
				int jIdx = jMin - j + radius;

				__VECTOR_ALIGNED__
				for (k = 0; k < jDiff; k++)
				{
					iIdx = iMin - i + radius;

					for (l = 0; l < iDiff; l++)
					{
						pF[k][l] = pH[k][l] * gMesh[jIdx][iIdx];
						normF += pF[k][l];
						iIdx++;
					}
					jIdx++;
				}

				for (k = 0; k < jDiff; k++)
				{
					const int kIdx = (jMin + k) * linePitch + iMin;
					for (l = 0; l < iDiff; l++)
					{
						srcWinPix = srcPix[kIdx + l];

						B = (static_cast<float> (srcWinPix & 0x000000000000FFFFu));
						G = (static_cast<float>((srcWinPix & 0x00000000FFFF0000u) >> 16));
						R = (static_cast<float>((srcWinPix & 0x0000FFFF00000000u) >> 32));

						Y = (R * pCoeffRGB2YUV[0] + G * pCoeffRGB2YUV[1] + B * pCoeffRGB2YUV[2]) / 32768.0f;
						bSum += (pF[k][l] * Y);
					}
				}

				// compute destination pixel
				Yfinal = bSum / normF;

				R = (Yfinal * pCoeffYUV2RGB[0] + Uref * pCoeffYUV2RGB[1] + Vref * pCoeffYUV2RGB[2]) * 32768.0f;
				G = (Yfinal * pCoeffYUV2RGB[3] + Uref * pCoeffYUV2RGB[4] + Vref * pCoeffYUV2RGB[5]) * 32768.0f;
				B = (Yfinal * pCoeffYUV2RGB[6] + Uref * pCoeffYUV2RGB[7] + Vref * pCoeffYUV2RGB[8]) * 32768.0f;

				newR = CLAMP_U16(static_cast<unsigned int>(R));
				newG = CLAMP_U16(static_cast<unsigned int>(G));
				newB = CLAMP_U16(static_cast<unsigned int>(B));

				dstPix[pixOffset] = A |
					((static_cast<csSDK_uint64>(newR)) << 32) |
					((static_cast<csSDK_uint64>(newG)) << 16) |
					 (static_cast<csSDK_uint64>(newB));

		} /* END: for (i = 0; i < width; i++) */

	} /* END: for (j = 0; j < height; j++) */


	return true;
}


bool process_BGRA_4444_32f_frame(const VideoHandle theData, const int radius)
{
#if !defined __INTEL_COMPILER 
	_MM_SET_FLUSH_ZERO_MODE(_MM_FLUSH_ZERO_ON);
	_MM_SET_DENORMALS_ZERO_MODE(_MM_DENORMALS_ZERO_ON);
#endif

	CACHE_ALIGN float pF[maxWinSize][maxWinSize] = {};
	CACHE_ALIGN float pH[maxWinSize][maxWinSize] = {};

	prRect box = {};

	// Get the frame dimensions
	((*theData)->piSuites->ppixFuncs->ppixGetBounds)((*theData)->destination, &box);

	// Calculate dimensions
	const csSDK_int32 height = box.bottom - box.top;
	const csSDK_int32 width = box.right - box.left;
	const csSDK_int32 rowbytes = ((*theData)->piSuites->ppixFuncs->ppixGetRowbytes)((*theData)->destination);

	const int lastLineIdx = height - 1;
	const int lastPixelIdx = width - 1;
	const int linePitch = rowbytes >> 2;

	constexpr float sigma = 0.10f;
	constexpr float divider = 2.00f * sigma * sigma;
	constexpr int   pixelSize = 4; /* 4 float elements per single pixel */

	// Create copies of pointer to the source, destination frames
	const float* __restrict srcBuf = reinterpret_cast<float* __restrict>(((*theData)->piSuites->ppixFuncs->ppixGetPixels)((*theData)->source));
	      float* __restrict dstBuf = reinterpret_cast<float* __restrict>(((*theData)->piSuites->ppixFuncs->ppixGetPixels)((*theData)->destination));
	const float* __restrict pCoeffRGB2YUV = width < 800 ? coeff_RGB2YUV[0] /* BT601 */ : coeff_RGB2YUV[1] /* BT709 */;
	const float* __restrict pCoeffYUV2RGB = width < 800 ? coeff_YUV2RGB[0] /* BT601 */ : coeff_YUV2RGB[1] /* BT709 */;

	int i, j;
	int k, l;
	float R, G, B, A, newR, newG, newB;
	float Y, U, V;
	float Yref, Uref, Vref;
	float Yfinal = 0.f; /* destination (filtered) Luma value */
	float normF, bSum;
	float fY, dY;

	for (j = 0; j < height; j++)
	{
		for (i = 0; i < width; i++)
		{
			// define processing window coordinates
			const int iMin = MAX(i - radius, 0);
			const int iMax = MIN(i + radius, lastPixelIdx);
			const int jMin = MAX(j - radius, 0);
			const int jMax = MIN(j + radius, lastLineIdx);

			// define process window sizes
			const int iDiff = (iMax - iMin) + 1;
			const int jDiff = (jMax - jMin) + 1;

			// offset of processed pixel
			const int pixIdx = (j * linePitch) + (i * pixelSize);

			__VECTOR_ALIGNED__
			B = srcBuf[pixIdx];
			G = srcBuf[pixIdx + 1];
			R = srcBuf[pixIdx + 2];
			A = srcBuf[pixIdx + 3];

			Yref = R * pCoeffRGB2YUV[0] + G * pCoeffRGB2YUV[1] + B * pCoeffRGB2YUV[2];
			Uref = R * pCoeffRGB2YUV[3] + G * pCoeffRGB2YUV[4] + B * pCoeffRGB2YUV[5];
			Vref = R * pCoeffRGB2YUV[6] + G * pCoeffRGB2YUV[7] + B * pCoeffRGB2YUV[8];

			// compute Gaussian intensity weights
			for (k = 0; k < jDiff; k++)
			{
				// first pixel position in specified line in filter window
				const int pixPos = (jMin + k) * linePitch + (iMin * pixelSize);

				for (l = 0; l < iDiff; l++)
				{
					const int pixOffset = pixPos + l * pixelSize;

					B = srcBuf[pixOffset];
					G = srcBuf[pixOffset + 1];
					R = srcBuf[pixOffset + 2];

					Y = R * pCoeffRGB2YUV[0] + G * pCoeffRGB2YUV[1] + B * pCoeffRGB2YUV[2];

					dY = Y - Yref;
					pH[k][l] = aExp(-(dY * dY) / divider);
				} // for (m = 0; m < iDiff; m++)

			} // for (k = 0; k < jDiff; k++)

			  // calculate Bilateral Filter responce
			normF = bSum = 0.0f;

			int iIdx = 0;
			int jIdx = jMin - j + radius;

			__VECTOR_ALIGNED__
			for (k = 0; k < jDiff; k++)
			{
				iIdx = iMin - i + radius;

				for (l = 0; l < iDiff; l++)
				{
					pF[k][l] = pH[k][l] * gMesh[jIdx][iIdx];
					normF += pF[k][l];
					iIdx++;
				}
				jIdx++;
			}

			for (k = 0; k < jDiff; k++)
			{
				const int kIdx = (jMin + k) * linePitch + iMin * pixelSize;
				for (l = 0; l < iDiff; l++)
				{
					const int lIdx = kIdx + l * pixelSize;

					B = srcBuf[lIdx];
					G = srcBuf[lIdx + 1];
					R = srcBuf[lIdx + 2];

					fY = R * pCoeffRGB2YUV[0] + G * pCoeffRGB2YUV[1] + B * pCoeffRGB2YUV[2];
					bSum += (pF[k][l] * fY);
				}
			}

			// compute destination pixel
			Yfinal = bSum / normF;

			newR = (Yfinal * pCoeffYUV2RGB[0] + Uref * pCoeffYUV2RGB[1] + Vref * pCoeffYUV2RGB[2]);
			newG = (Yfinal * pCoeffYUV2RGB[3] + Uref * pCoeffYUV2RGB[4] + Vref * pCoeffYUV2RGB[5]);
			newB = (Yfinal * pCoeffYUV2RGB[6] + Uref * pCoeffYUV2RGB[7] + Vref * pCoeffYUV2RGB[8]);

			__VECTOR_ALIGNED__
			dstBuf[pixIdx]		= newB;
			dstBuf[pixIdx + 1]	= newG;
			dstBuf[pixIdx + 2]	= newR;
			// copy ALPHA component from source to destination "as is"
			dstBuf[pixIdx + 3]	= A;
		}
	}

	return true;
}


bool process_RGB_444_10u_frame(const VideoHandle theData, const int radius)
{
#if !defined __INTEL_COMPILER 
	_MM_SET_FLUSH_ZERO_MODE(_MM_FLUSH_ZERO_ON);
	_MM_SET_DENORMALS_ZERO_MODE(_MM_DENORMALS_ZERO_ON);
#endif

	CACHE_ALIGN float pF[maxWinSize][maxWinSize] = {};
	CACHE_ALIGN float pH[maxWinSize][maxWinSize] = {};

	prRect box = { 0 };

	// Get the frame dimensions
	((*theData)->piSuites->ppixFuncs->ppixGetBounds)((*theData)->destination, &box);

	// Calculate dimensions
	const csSDK_int32 height = box.bottom - box.top;
	const csSDK_int32 width = box.right - box.left;
	const csSDK_int32 rowbytes = ((*theData)->piSuites->ppixFuncs->ppixGetRowbytes)((*theData)->destination);

	const int lastLineIdx = height - 1;
	const int lastPixelIdx = width - 1;
	const int linePitch = rowbytes >> 2;

	constexpr float sigma = 0.10f;
	constexpr float divider = 2.00f * sigma * sigma;

	// Create copies of pointer to the source, destination frames
	const csSDK_uint32* __restrict srcPix = reinterpret_cast<csSDK_uint32* __restrict>(((*theData)->piSuites->ppixFuncs->ppixGetPixels)((*theData)->source));
	      csSDK_uint32* __restrict dstPix = reinterpret_cast<csSDK_uint32* __restrict>(((*theData)->piSuites->ppixFuncs->ppixGetPixels)((*theData)->destination));
	const float*        __restrict pCoeffRGB2YUV = width < 800 ? coeff_RGB2YUV[0] /* BT601 */ : coeff_RGB2YUV[1] /* BT709 */;
	const float*        __restrict pCoeffYUV2RGB = width < 800 ? coeff_YUV2RGB[0] /* BT601 */ : coeff_YUV2RGB[1] /* BT709 */;

	int i, j;
	int k, l;
	int pixOffset = 0;
	float R, G, B, Y, dY, Yfinal;
	float Yref, Uref, Vref;
	float normF, bSum;

	csSDK_uint32 srcWinPix;
	csSDK_uint32 newR, newB, newG;

	for (j = 0; j < height; j++)
	{
		for (i = 0; i < width; i++)
		{
			// define processing window coordinates
			const int iMin = MAX(i - radius, 0);
			const int iMax = MIN(i + radius, lastPixelIdx);
			const int jMin = MAX(j - radius, 0);
			const int jMax = MIN(j + radius, lastLineIdx);

			// define process window sizes
			const int iDiff = (iMax - iMin) + 1;
			const int jDiff = (jMax - jMin) + 1;

			pixOffset = j * linePitch + i;
			const csSDK_uint32 srcPixel = srcPix[pixOffset];

			R = static_cast<unsigned int>((srcPixel & 0x00000FFCu) >> 2);
			G = static_cast<unsigned int>((srcPixel & 0x003FF000u) >> 12);
			B = static_cast<unsigned int>((srcPixel & 0xFFC00000u) >> 22);

			Yref = (R * pCoeffRGB2YUV[0] + G * pCoeffRGB2YUV[1] + B * pCoeffRGB2YUV[2]);
			Uref = (R * pCoeffRGB2YUV[3] + G * pCoeffRGB2YUV[4] + B * pCoeffRGB2YUV[5]);
			Vref = (R * pCoeffRGB2YUV[6] + G * pCoeffRGB2YUV[7] + B * pCoeffRGB2YUV[8]);

			Yref /= 1024.0f;

			// compute Gaussian intensity weights
			for (k = 0; k < jDiff; k++)
			{
				// first pixel position in specified line in filter window
				const int pixPos = (jMin + k) * linePitch + iMin;

				for (l = 0; l < iDiff; l++)
				{
					srcWinPix = srcPix[pixPos + l];

					R = static_cast<unsigned int>((srcWinPix & 0x00000FFCu) >> 2);
					G = static_cast<unsigned int>((srcWinPix & 0x003FF000u) >> 12);
					B = static_cast<unsigned int>((srcWinPix & 0xFFC00000u) >> 22);

					Y = (R * pCoeffRGB2YUV[0] + G * pCoeffRGB2YUV[1] + B * pCoeffRGB2YUV[2]) / 1024.0f;
					dY = Y - Yref;
					pH[k][l] = aExp(-(dY * dY) / divider);
				} // for (m = 0; m < iDiff; m++)

			} // for (k = 0; k < jDiff; k++)

			  // calculate Bilateral Filter responce
			normF = bSum = 0.0f;

			int iIdx = 0;
			int jIdx = jMin - j + radius;

			__VECTOR_ALIGNED__
				for (k = 0; k < jDiff; k++)
				{
					iIdx = iMin - i + radius;

					for (l = 0; l < iDiff; l++)
					{
						pF[k][l] = pH[k][l] * gMesh[jIdx][iIdx];
						normF += pF[k][l];
						iIdx++;
					}
					jIdx++;
				}

			for (k = 0; k < jDiff; k++)
			{
				const int kIdx = (jMin + k) * linePitch + iMin;
				for (l = 0; l < iDiff; l++)
				{
					srcWinPix = srcPix[kIdx + l];

					R = static_cast<unsigned int>((srcWinPix & 0x00000FFCu) >> 2);
					G = static_cast<unsigned int>((srcWinPix & 0x003FF000u) >> 12);
					B = static_cast<unsigned int>((srcWinPix & 0xFFC00000u) >> 22);

					Y = (R * pCoeffRGB2YUV[0] + G * pCoeffRGB2YUV[1] + B * pCoeffRGB2YUV[2]) / 1024.0f;
					bSum += (pF[k][l] * Y);
				}
			}

			// compute destination pixel
			Yfinal = (1024.0f * bSum) / normF;

			R = Yfinal * pCoeffYUV2RGB[0] + Uref * pCoeffYUV2RGB[1] + Vref * pCoeffYUV2RGB[2];
			G = Yfinal * pCoeffYUV2RGB[3] + Uref * pCoeffYUV2RGB[4] + Vref * pCoeffYUV2RGB[5];
			B = Yfinal * pCoeffYUV2RGB[6] + Uref * pCoeffYUV2RGB[7] + Vref * pCoeffYUV2RGB[8];

			newR = CLAMP_U10(static_cast<unsigned int>(R));
			newG = CLAMP_U10(static_cast<unsigned int>(G));
			newB = CLAMP_U10(static_cast<unsigned int>(B));

			dstPix[pixOffset] = (newR << 2)  |
								(newG << 12) |
								(newB << 22);

		} /* END: for (i = 0; i < width; i++) */

	} /* END: for (j = 0; j < height; j++) */

	return true;
}


bool process_ARGB_4444_8u_frame (const VideoHandle theData, const int radius)
{
#if !defined __INTEL_COMPILER 
	_MM_SET_FLUSH_ZERO_MODE(_MM_FLUSH_ZERO_ON);
	_MM_SET_DENORMALS_ZERO_MODE(_MM_DENORMALS_ZERO_ON);
#endif

	CACHE_ALIGN float pF[maxWinSize][maxWinSize] = {};
	CACHE_ALIGN float pH[maxWinSize][maxWinSize] = {};

	prRect box = { 0 };

	// Get the frame dimensions
	((*theData)->piSuites->ppixFuncs->ppixGetBounds)((*theData)->destination, &box);

	// Calculate dimensions
	const csSDK_int32 height = box.bottom - box.top;
	const csSDK_int32 width = box.right - box.left;
	const csSDK_int32 rowbytes = ((*theData)->piSuites->ppixFuncs->ppixGetRowbytes)((*theData)->destination);

	const int lastLineIdx = height - 1;
	const int lastPixelIdx = width - 1;
	const int linePitch = rowbytes >> 2;

	constexpr float sigma = 0.10f;
	constexpr float divider = 2.00f * sigma * sigma;

	// Create copies of pointer to the source, destination frames
	const csSDK_uint32* __restrict srcPix = reinterpret_cast<csSDK_uint32* __restrict>(((*theData)->piSuites->ppixFuncs->ppixGetPixels)((*theData)->source));
	      csSDK_uint32* __restrict dstPix = reinterpret_cast<csSDK_uint32* __restrict>(((*theData)->piSuites->ppixFuncs->ppixGetPixels)((*theData)->destination));
	const float*        __restrict pCoeffRGB2YUV = width < 800 ? coeff_RGB2YUV[0] /* BT601 */ : coeff_RGB2YUV[1] /* BT709 */;
	const float*        __restrict pCoeffYUV2RGB = width < 800 ? coeff_YUV2RGB[0] /* BT601 */ : coeff_YUV2RGB[1] /* BT709 */;

	int i, j;
	int k, l;
	int pixOffset = 0;
	float R, G, B, Y, dY, Yfinal;
	float Yref, Uref, Vref;
	float normF, bSum;

	csSDK_uint32 srcWinPix;
	csSDK_uint32 newR, newB, newG;

	for (j = 0; j < height; j++)
	{
		for (i = 0; i < width; i++)
		{
			// define processing window coordinates
			const int iMin = MAX(i - radius, 0);
			const int iMax = MIN(i + radius, lastPixelIdx);
			const int jMin = MAX(j - radius, 0);
			const int jMax = MIN(j + radius, lastLineIdx);

			// define process window sizes
			const int iDiff = (iMax - iMin) + 1;
			const int jDiff = (jMax - jMin) + 1;

			pixOffset = j * linePitch + i;
			const csSDK_uint32 srcPixel = srcPix[pixOffset];

			const csSDK_uint32 A =  srcPixel & 0x000000FFu;
			R = static_cast<float>((srcPixel & 0x0000FF00u) >> 8);
			G = static_cast<float>((srcPixel & 0x00FF0000u) >> 16);
			B = static_cast<float>((srcPixel & 0xFF000000u) >> 24);

			Yref = R * pCoeffRGB2YUV[0] + G * pCoeffRGB2YUV[1] + B * pCoeffRGB2YUV[2];
			Uref = R * pCoeffRGB2YUV[3] + G * pCoeffRGB2YUV[4] + B * pCoeffRGB2YUV[5];
			Vref = R * pCoeffRGB2YUV[6] + G * pCoeffRGB2YUV[7] + B * pCoeffRGB2YUV[8];

			Yref /= 255.0f;

			// compute Gaussian intensity weights
			for (k = 0; k < jDiff; k++)
			{
				// first pixel position in specified line in filter window
				const int pixPos = (jMin + k) * linePitch + iMin;

				for (l = 0; l < iDiff; l++)
				{
					srcWinPix = srcPix[pixPos + l];

					R = static_cast<float>((srcWinPix & 0x0000FF00u) >> 8);
					G = static_cast<float>((srcWinPix & 0x00FF0000u) >> 16);
					B = static_cast<float>((srcWinPix & 0xFF000000u) >> 24);

					Y = (R * pCoeffRGB2YUV[0] + G * pCoeffRGB2YUV[1] + B * pCoeffRGB2YUV[2]) / 255.0f;
					dY = Y - Yref;
					pH[k][l] = aExp(-(dY * dY) / divider);
				} // for (m = 0; m < iDiff; m++)

			} // for (k = 0; k < jDiff; k++)

			  // calculate Bilateral Filter responce
			normF = bSum = 0.0f;

			int iIdx = 0;
			int jIdx = jMin - j + radius;

			__VECTOR_ALIGNED__
				for (k = 0; k < jDiff; k++)
				{
					iIdx = iMin - i + radius;

					for (l = 0; l < iDiff; l++)
					{
						pF[k][l] = pH[k][l] * gMesh[jIdx][iIdx];
						normF += pF[k][l];
						iIdx++;
					}
					jIdx++;
				}

			for (k = 0; k < jDiff; k++)
			{
				const int kIdx = (jMin + k) * linePitch + iMin;
				for (l = 0; l < iDiff; l++)
				{
					srcWinPix = srcPix[kIdx + l];

					R = static_cast<float>((srcWinPix & 0x0000FF00u) >> 8);
					G = static_cast<float>((srcWinPix & 0x00FF0000u) >> 16);
					B = static_cast<float>((srcWinPix & 0xFF000000u) >> 24);

					Y = (R * pCoeffRGB2YUV[0] + G * pCoeffRGB2YUV[1] + B * pCoeffRGB2YUV[2]) / 255.0f;
					bSum += (pF[k][l] * Y);
				}
			}

			// compute destination pixel
			Yfinal = (255.0f * bSum) / normF;

			R = Yfinal * pCoeffYUV2RGB[0] + Uref * pCoeffYUV2RGB[1] + Vref * pCoeffYUV2RGB[2];
			G = Yfinal * pCoeffYUV2RGB[3] + Uref * pCoeffYUV2RGB[4] + Vref * pCoeffYUV2RGB[5];
			B = Yfinal * pCoeffYUV2RGB[6] + Uref * pCoeffYUV2RGB[7] + Vref * pCoeffYUV2RGB[8];

			newR = CLAMP_U8(static_cast<unsigned int>(R));
			newG = CLAMP_U8(static_cast<unsigned int>(G));
			newB = CLAMP_U8(static_cast<unsigned int>(B));

			dstPix[pixOffset] = A |
								(newR << 8)  |
								(newG << 16) |
								(newB << 24);

		} /* END: for (i = 0; i < width; i++) */

	} /* END: for (j = 0; j < height; j++) */

	return true;
}


bool process_ARGB_4444_16u_frame (const VideoHandle theData, const int radius)
{
#if !defined __INTEL_COMPILER 
	_MM_SET_FLUSH_ZERO_MODE(_MM_FLUSH_ZERO_ON);
	_MM_SET_DENORMALS_ZERO_MODE(_MM_DENORMALS_ZERO_ON);
#endif

	CACHE_ALIGN float pF[maxWinSize][maxWinSize] = {};
	CACHE_ALIGN float pH[maxWinSize][maxWinSize] = {};

	prRect box = { 0 };

	// Get the frame dimensions
	((*theData)->piSuites->ppixFuncs->ppixGetBounds)((*theData)->destination, &box);

	// Calculate dimensions
	const csSDK_int32 height = box.bottom - box.top;
	const csSDK_int32 width = box.right - box.left;
	const csSDK_int32 rowbytes = ((*theData)->piSuites->ppixFuncs->ppixGetRowbytes)((*theData)->destination);

	const int lastLineIdx = height - 1;
	const int lastPixelIdx = width - 1;
	const int linePitch = rowbytes >> 3; // because pixel defined as uint64 - lets add additional right shift to linepitch

	constexpr float sigma = 0.10f;
	constexpr float divider = 2.00f * sigma * sigma;

	// Create copies of pointer to the source, destination frames
	const csSDK_uint64* __restrict srcPix = reinterpret_cast<csSDK_uint64* __restrict>(((*theData)->piSuites->ppixFuncs->ppixGetPixels)((*theData)->source));
		  csSDK_uint64* __restrict dstPix = reinterpret_cast<csSDK_uint64* __restrict>(((*theData)->piSuites->ppixFuncs->ppixGetPixels)((*theData)->destination));
	const float*        __restrict pCoeffRGB2YUV = width < 800 ? coeff_RGB2YUV[0] /* BT601 */ : coeff_RGB2YUV[1] /* BT709 */;
	const float*        __restrict pCoeffYUV2RGB = width < 800 ? coeff_YUV2RGB[0] /* BT601 */ : coeff_YUV2RGB[1] /* BT709 */;

	int i, j;
	int k, l;
	int pixOffset = 0;
	float R, G, B, Y, dY, Yfinal = 0.0f;
	float Yref, Uref, Vref;
	float normF, bSum;

	csSDK_uint64 srcWinPix, A;
	csSDK_uint32 newR, newB, newG;

	for (j = 0; j < height; j++)
	{
		for (i = 0; i < width; i++)
		{
			// define processing window coordinates
			const int iMin = MAX(i - radius, 0);
			const int iMax = MIN(i + radius, lastPixelIdx);
			const int jMin = MAX(j - radius, 0);
			const int jMax = MIN(j + radius, lastLineIdx);

			// define process window sizes
			const int iDiff = (iMax - iMin) + 1;
			const int jDiff = (jMax - jMin) + 1;

			pixOffset = j * linePitch + i;
			const csSDK_uint64 refPix = srcPix[pixOffset];

			__VECTOR_ALIGNED__
			A = (static_cast<float> (refPix & 0x000000000000FFFFu));
			R = (static_cast<float>((refPix & 0x00000000FFFF0000u) >> 16));
			G = (static_cast<float>((refPix & 0x0000FFFF00000000u) >> 32));
			B = (static_cast<float>((refPix & 0xFFFF000000000000u) >> 48));

			Yref = (R * pCoeffRGB2YUV[0] + G * pCoeffRGB2YUV[1] + B * pCoeffRGB2YUV[2]) / 32768.0f;
			Uref = (R * pCoeffRGB2YUV[3] + G * pCoeffRGB2YUV[4] + B * pCoeffRGB2YUV[5]) / 32768.0f;
			Vref = (R * pCoeffRGB2YUV[6] + G * pCoeffRGB2YUV[7] + B * pCoeffRGB2YUV[8]) / 32768.0f;

			// compute Gaussian intensity weights
			for (k = 0; k < jDiff; k++)
			{
				// first pixel position in specified line in filter window
				const int pixPos = (jMin + k) * linePitch + iMin;

				for (l = 0; l < iDiff; l++)
				{
					srcWinPix = srcPix[pixPos + l];

					__VECTOR_ALIGNED__
					R = (static_cast<float>((srcWinPix & 0x00000000FFFF0000u) >> 16));
					G = (static_cast<float>((srcWinPix & 0x0000FFFF00000000u) >> 32));
					B = (static_cast<float>((srcWinPix & 0xFFFF000000000000u) >> 48));

					Y = (R * pCoeffRGB2YUV[0] + G * pCoeffRGB2YUV[1] + B * pCoeffRGB2YUV[2]) / 32768.0f;
					dY = Y - Yref;
					pH[k][l] = aExp(-(dY * dY) / divider);
				} // for (m = 0; m < iDiff; m++)

			} // for (k = 0; k < jDiff; k++)

			  // calculate Bilateral Filter responce
			normF = bSum = 0.0f;

			int iIdx = 0;
			int jIdx = jMin - j + radius;

			__VECTOR_ALIGNED__
				for (k = 0; k < jDiff; k++)
				{
					iIdx = iMin - i + radius;

					for (l = 0; l < iDiff; l++)
					{
						pF[k][l] = pH[k][l] * gMesh[jIdx][iIdx];
						normF += pF[k][l];
						iIdx++;
					}
					jIdx++;
				}

			for (k = 0; k < jDiff; k++)
			{
				const int kIdx = (jMin + k) * linePitch + iMin;
				for (l = 0; l < iDiff; l++)
				{
					srcWinPix = srcPix[kIdx + l];

					__VECTOR_ALIGNED__
					R = (static_cast<float>((srcWinPix & 0x00000000FFFF0000u) >> 16));
					G = (static_cast<float>((srcWinPix & 0x0000FFFF00000000u) >> 32));
					B = (static_cast<float>((srcWinPix & 0xFFFF000000000000u) >> 48));

					Y = (R * pCoeffRGB2YUV[0] + G * pCoeffRGB2YUV[1] + B * pCoeffRGB2YUV[2]) / 32768.0f;
					bSum += (pF[k][l] * Y);
				}
			}

			// compute destination pixel
			Yfinal = bSum / normF;

			R = (Yfinal * pCoeffYUV2RGB[0] + Uref * pCoeffYUV2RGB[1] + Vref * pCoeffYUV2RGB[2]) * 32768.0f;
			G = (Yfinal * pCoeffYUV2RGB[3] + Uref * pCoeffYUV2RGB[4] + Vref * pCoeffYUV2RGB[5]) * 32768.0f;
			B = (Yfinal * pCoeffYUV2RGB[6] + Uref * pCoeffYUV2RGB[7] + Vref * pCoeffYUV2RGB[8]) * 32768.0f;

			newR = CLAMP_U16(static_cast<unsigned int>(R));
			newG = CLAMP_U16(static_cast<unsigned int>(G));
			newB = CLAMP_U16(static_cast<unsigned int>(B));

			__VECTOR_ALIGNED__
			dstPix[pixOffset] = A |
				((static_cast<csSDK_uint64>(newR)) << 16) |
				((static_cast<csSDK_uint64>(newG)) << 32) |
				((static_cast<csSDK_uint64>(newB)) << 48);

		} /* END: for (i = 0; i < width; i++) */

	} /* END: for (j = 0; j < height; j++) */


	return true;
}


bool process_ARGB_4444_32f_frame (const VideoHandle theData, const int radius)
{
#if !defined __INTEL_COMPILER 
	_MM_SET_FLUSH_ZERO_MODE(_MM_FLUSH_ZERO_ON);
	_MM_SET_DENORMALS_ZERO_MODE(_MM_DENORMALS_ZERO_ON);
#endif

	CACHE_ALIGN float pF[maxWinSize][maxWinSize] = {};
	CACHE_ALIGN float pH[maxWinSize][maxWinSize] = {};

	prRect box = {};

	// Get the frame dimensions
	((*theData)->piSuites->ppixFuncs->ppixGetBounds)((*theData)->destination, &box);

	// Calculate dimensions
	const csSDK_int32 height = box.bottom - box.top;
	const csSDK_int32 width = box.right - box.left;
	const csSDK_int32 rowbytes = ((*theData)->piSuites->ppixFuncs->ppixGetRowbytes)((*theData)->destination);

	const int lastLineIdx = height - 1;
	const int lastPixelIdx = width - 1;
	const int linePitch = rowbytes >> 2;

	constexpr float sigma = 0.10f;
	constexpr float divider = 2.00f * sigma * sigma;
	constexpr int   pixelSize = 4; /* 4 float elements per single pixel */

	// Create copies of pointer to the source, destination frames
	const float* __restrict srcBuf = reinterpret_cast<float* __restrict>(((*theData)->piSuites->ppixFuncs->ppixGetPixels)((*theData)->source));
	float*		 __restrict dstBuf = reinterpret_cast<float* __restrict>(((*theData)->piSuites->ppixFuncs->ppixGetPixels)((*theData)->destination));
	const float* __restrict pCoeffRGB2YUV = width < 800 ? coeff_RGB2YUV[0] /* BT601 */ : coeff_RGB2YUV[1] /* BT709 */;
	const float* __restrict pCoeffYUV2RGB = width < 800 ? coeff_YUV2RGB[0] /* BT601 */ : coeff_YUV2RGB[1] /* BT709 */;

	int i, j;
	int k, l;
	float R, G, B, A, newR, newG, newB;
	float Y, U, V;
	float Yref, Uref, Vref;
	float Yfinal = 0.f; /* destination (filtered) Luma value */
	float normF, bSum;
	float fY, dY;

	for (j = 0; j < height; j++)
	{
		for (i = 0; i < width; i++)
		{
			// define processing window coordinates
			const int iMin = MAX(i - radius, 0);
			const int iMax = MIN(i + radius, lastPixelIdx);
			const int jMin = MAX(j - radius, 0);
			const int jMax = MIN(j + radius, lastLineIdx);

			// define process window sizes
			const int iDiff = (iMax - iMin) + 1;
			const int jDiff = (jMax - jMin) + 1;

			// offset of processed pixel
			const int pixIdx = (j * linePitch) + (i * pixelSize);

			__VECTOR_ALIGNED__
			A = srcBuf[pixIdx];
			R = srcBuf[pixIdx + 1];
			G = srcBuf[pixIdx + 2];
			B = srcBuf[pixIdx + 3];

			Yref = R * pCoeffRGB2YUV[0] + G * pCoeffRGB2YUV[1] + B * pCoeffRGB2YUV[2];
			Uref = R * pCoeffRGB2YUV[3] + G * pCoeffRGB2YUV[4] + B * pCoeffRGB2YUV[5];
			Vref = R * pCoeffRGB2YUV[6] + G * pCoeffRGB2YUV[7] + B * pCoeffRGB2YUV[8];

			// compute Gaussian intensity weights
			for (k = 0; k < jDiff; k++)
			{
				// first pixel position in specified line in filter window
				const int pixPos = (jMin + k) * linePitch + (iMin * pixelSize);

				for (l = 0; l < iDiff; l++)
				{
					const int pixOffset = pixPos + l * pixelSize;

					R = srcBuf[pixOffset + 1];
					G = srcBuf[pixOffset + 2];
					B = srcBuf[pixOffset + 3];

					Y = R * pCoeffRGB2YUV[0] + G * pCoeffRGB2YUV[1] + B * pCoeffRGB2YUV[2];

					dY = Y - Yref;
					pH[k][l] = aExp(-(dY * dY) / divider);
				} // for (m = 0; m < iDiff; m++)

			} // for (k = 0; k < jDiff; k++)

			  // calculate Bilateral Filter responce
			normF = bSum = 0.0f;

			int iIdx = 0;
			int jIdx = jMin - j + radius;

			__VECTOR_ALIGNED__
				for (k = 0; k < jDiff; k++)
				{
					iIdx = iMin - i + radius;

					for (l = 0; l < iDiff; l++)
					{
						pF[k][l] = pH[k][l] * gMesh[jIdx][iIdx];
						normF += pF[k][l];
						iIdx++;
					}
					jIdx++;
				}

			for (k = 0; k < jDiff; k++)
			{
				const int kIdx = (jMin + k) * linePitch + iMin * pixelSize;
				for (l = 0; l < iDiff; l++)
				{
					const int lIdx = kIdx + l * pixelSize;

					R = srcBuf[lIdx + 1];
					G = srcBuf[lIdx + 2];
					B = srcBuf[lIdx + 3];

					fY = R * pCoeffRGB2YUV[0] + G * pCoeffRGB2YUV[1] + B * pCoeffRGB2YUV[2];
					bSum += (pF[k][l] * fY);
				}
			}

			// compute destination pixel
			Yfinal = bSum / normF;

			newR = (Yfinal * pCoeffYUV2RGB[0] + Uref * pCoeffYUV2RGB[1] + Vref * pCoeffYUV2RGB[2]);
			newG = (Yfinal * pCoeffYUV2RGB[3] + Uref * pCoeffYUV2RGB[4] + Vref * pCoeffYUV2RGB[5]);
			newB = (Yfinal * pCoeffYUV2RGB[6] + Uref * pCoeffYUV2RGB[7] + Vref * pCoeffYUV2RGB[8]);

			__VECTOR_ALIGNED__
			dstBuf[pixIdx] = A;
			dstBuf[pixIdx + 1] = newR;
			dstBuf[pixIdx + 2] = newG;
			dstBuf[pixIdx + 3] = newB;
		}
	}

	return true;
}


csSDK_int32 selectProcessFunction (const VideoHandle theData)
{
	static constexpr char* strPpixSuite = "Premiere PPix Suite";
	SPBasicSuite*		   SPBasic = nullptr;
	csSDK_int32 errCode = fsBadFormatIndex;
	bool processSucceed = true;

	// acquire Premier Suites
	if (nullptr != (SPBasic = (*theData)->piSuites->utilFuncs->getSPBasicSuite()))
	{
		PrSDKPPixSuite*			PPixSuite = nullptr;
		const SPErr err = SPBasic->AcquireSuite (strPpixSuite, 1, (const void**)&PPixSuite);

		if (nullptr != PPixSuite && kSPNoError == err)
		{
			PrPixelFormat pixelFormat = PrPixelFormat_Invalid;
			PPixSuite->GetPixelFormat((*theData)->source, &pixelFormat);

			switch (pixelFormat)
			{
				// ============ native AP formats ============================= //
				case PrPixelFormat_BGRA_4444_8u:
					processSucceed = process_BGRA_4444_8u_frame(theData);
				break;

				case PrPixelFormat_VUYA_4444_8u:
				case PrPixelFormat_VUYA_4444_8u_709:
					processSucceed = process_VUYA_4444_8u_frame(theData);
				break;

				case PrPixelFormat_BGRA_4444_16u:
					processSucceed = process_BGRA_4444_16u_frame(theData);
				break;

				case PrPixelFormat_BGRA_4444_32f:
					processSucceed = process_BGRA_4444_32f_frame(theData);
				break;

				case PrPixelFormat_VUYA_4444_32f:
				case PrPixelFormat_VUYA_4444_32f_709:
					processSucceed = process_VUYA_4444_32f_frame(theData);
				break;

				// ============ native AE formats ============================= //
				case PrPixelFormat_ARGB_4444_8u:
					processSucceed = process_ARGB_4444_8u_frame(theData);
				break;

				case PrPixelFormat_ARGB_4444_16u:
					processSucceed = process_ARGB_4444_16u_frame(theData);
				break;

				case PrPixelFormat_ARGB_4444_32f:
					processSucceed = process_ARGB_4444_32f_frame(theData);
				break;

				// =========== miscellanous formats =========================== //
				case PrPixelFormat_RGB_444_10u:
					processSucceed = process_RGB_444_10u_frame(theData);
				break;

				default:
					processSucceed = false;
				break;
			}

			SPBasic->ReleaseSuite(strPpixSuite, 1);
			errCode = (true == processSucceed) ? fsNoErr : errCode;
		}
	}

	return errCode;
}


// Bilateral-RGB filter entry point
PREMPLUGENTRY DllExport xFilter(short selector, VideoHandle theData)
{
	csSDK_int32 errCode = fsNoErr;

	switch (selector)
	{
		case fsExecute:
			errCode = selectProcessFunction(theData);
		break;

		case fsInitSpec:
		break;

		case fsHasSetupDialog:
			errCode = fsHasNoSetupDialog;
		break;

		case fsSetup:
		break;

		case fsDisposeData:
		break;
		
		case fsCanHandlePAR:
			errCode = prEffectCanHandlePAR;
		break;

		case fsGetPixelFormatsSupported:
			errCode = imageLabPixelFormatSupported (theData);
		break;

		case fsCacheOnLoad:
			errCode = fsDoNotCacheOnLoad;
		break;

		default:
			// unhandled case
		break;

	}

	return errCode;
}