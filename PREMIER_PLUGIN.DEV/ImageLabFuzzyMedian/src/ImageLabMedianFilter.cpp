#include "ImageLabFuzzyMedian.h"


bool median_filter_BGRA_4444_8u_frame (const VideoHandle theData, const csSDK_int32& kernelWidth)
{
	// protection on kernel size
	if (MaxKernelWidth < kernelWidth || MinKernelWidth > kernelWidth)
		return false;

	CACHE_ALIGN uint32_t windowR[MaxKernelElemSize];
	CACHE_ALIGN uint32_t windowG[MaxKernelElemSize];
	CACHE_ALIGN uint32_t windowB[MaxKernelElemSize];

	prRect box = { 0 };

	// Get the frame dimensions
	((*theData)->piSuites->ppixFuncs->ppixGetBounds)((*theData)->destination, &box);

	// Calculate dimensions
	const csSDK_int32 height = box.bottom - box.top;
	const csSDK_int32 width = box.right - box.left;
	const csSDK_int32 rowbytes = ((*theData)->piSuites->ppixFuncs->ppixGetRowbytes)((*theData)->destination);
	const csSDK_int32 linePitch = rowbytes >> 2;

	// Create copies of pointer to the source, destination frames
	const csSDK_uint32* __restrict srcPix = reinterpret_cast<const csSDK_uint32* __restrict>(((*theData)->piSuites->ppixFuncs->ppixGetPixels)((*theData)->source));
	      csSDK_uint32* __restrict dstPix = reinterpret_cast<csSDK_uint32* __restrict>(((*theData)->piSuites->ppixFuncs->ppixGetPixels)((*theData)->destination));

	const csSDK_int32 border = kernelWidth >> 1;
	const csSDK_int32 kernelSize = kernelWidth * kernelWidth;
	const csSDK_int32 medianPosition = (kernelSize >> 1) + 1;
	const csSDK_int32 lastLine = height - border;
	const csSDK_int32 lastPixel = width - border;

	csSDK_int32 i, j, k, l, m, n;
	csSDK_int32 idxJ = 0, idxI = 0;
	csSDK_int32 lMin, lMax, mMin, mMax, idxLine, idxPix, idxBuf;

	/* process border pixels from top lines of frame */
	for (j = 0; j < lastLine; j++)
	{
		lMin = j - border;

		/* process pixels from left side */
		for (i = 0; i < border; i++)
		{
			mMin = i - border;
			k = 0;
			for (l = lMin, idxJ = 0; idxJ < kernelWidth; idxJ++, l++)
			{
				idxLine = ((l <= 0) ? 0 : l) * linePitch;
				for (m = mMin, idxI = 0; idxI < kernelWidth; idxI++, m++)
				{
					idxPix = idxLine + ((m <= 0) ? 0 : m);
					windowB[k] = (srcPix[idxPix] & 0x000000FFu);
					windowG[k] = (srcPix[idxPix] & 0x0000FF00u) >> 8;
					windowR[k] = (srcPix[idxPix] & 0x00FF0000u) >> 16;
					k++;
				} /* for (m = mMin; m < kernelWidth - border; m++) */
			} /* for (l = lMin; l < (kernelWidth - border); l++) */

			gnomesort(windowB, windowB + kernelSize);
			gnomesort(windowG, windowG + kernelSize);
			gnomesort(windowR, windowR + kernelSize);

			idxBuf = j * linePitch + i;
			dstPix[idxBuf] = (srcPix[idxBuf] & 0xFF000000u) | /* copy ALPHA channel value as is */
									windowB[medianPosition] |
							 (windowG[medianPosition] << 8) |
							 (windowR[medianPosition] << 16);

		} /* for (i = 0; i < border; i++) */

		/* process pixels after left border, but before right border */
		for (n = i, i = border; i < lastPixel; i++)
		{
			idxPix = ((lMin <= 0) ? 0 : l) * linePitch + i;

			for (m = 0; m < kernelWidth; m++)
			{
				k = (m % i) * kernelWidth + n;
				
				windowB[k] = (srcPix[idxPix] & 0x000000FFu);
				windowG[k] = (srcPix[idxPix] & 0x0000FF00u) >> 8;
				windowR[k] = (srcPix[idxPix] & 0x00FF0000u) >> 16;

				idxPix += ((lMin <= 0) ? 0 : linePitch);
			} /* for (n = 0, l = 0; l < kernelWidth; l++) */

			insertionsort(windowB, windowB + kernelSize);
			insertionsort(windowG, windowG + kernelSize);
			insertionsort(windowR, windowR + kernelSize);

			idxBuf = j * linePitch + i;
			dstPix[idxBuf] = (srcPix[idxBuf] & 0xFF000000u) | /* copy ALPHA channel value as is */
									windowB[medianPosition] |
							 (windowG[medianPosition] << 8) |
							 (windowR[medianPosition] << 16);

			n %= kernelWidth;

		} /* for (n = 0, i = border; i < lastPixel; i++) */

		/* process pixels in end of line */
//		for (i = lastPixel; i < width; i++)
//		{
//			lMax = i + border;
//
//		} /* for (i = lastPixel; i < width; i++) */

	} /* for (j = 0; j < lastLine; j++) */



#if 0
	// copy border lines in top of frame
	__VECTOR_ALIGNED__
	for (j = 0; j < border; j++)
	{
		idxPix = j * linePitch;
		memcpy(&dstPix[idxPix], &srcPix[idxPix], lineBytesSize);
	}

	// perform Median Filter on frame
	for (j = border; j < lastLine; j++)
	{
		// copy border pixels from start of line
		idxPix = j * linePitch;
		for (i = 0; i < border; i++)
			dstPix[idxPix + i] = srcPix[idxPix + i];

		/* ======  first window on line way pass  ======== */
		lMin = j - border;
		lMax = j + border;
		k = 0;

		__VECTOR_ALIGNED__
		for (l = lMin; l <= lMax; l++)
		{
			idxPix = l * linePitch;
			for (m = 0; m < kernelWidth; m++)
			{
				windowB[k] = (srcPix[idxPix + m] & 0x000000FFu);
				windowG[k] = (srcPix[idxPix + m] & 0x0000FF00u) >> 8;
				windowR[k] = (srcPix[idxPix + m] & 0x00FF0000u) >> 16;
				k++;
			} /* for (m = i - border; m = i + border; i++) */
		} /* for (l = j - border; l < j + border; l++) */

		/* Order elements */
		gnomesort(windowB, windowB + kernelSize);
		gnomesort(windowG, windowG + kernelSize);
		gnomesort(windowR, windowR + kernelSize);

		idxPix = (j * linePitch + i);

		dstPix[idxPix] = (srcPix[idxPix] & 0xFF000000u) | /* copy ALPHA channel value as is */
						windowB[medianPosition]         |
					   (windowG[medianPosition] << 8)   |
			           (windowR[medianPosition] << 16);

		/* ======  loop - switch window on next position in line way  ======== */
		for (n = 0, i = border + 1; i < lastPixel; i++, n++)
		{
			// Pick up window elements
			lMax = j + border;
			mMax = i - border;

			idxPix = lMax * linePitch + mMax;

			for (m = 0; m < kernelWidth; m++)
			{
				l = (m % kernelWidth) * kernelWidth + n;
				
				windowB[l] = (srcPix[idxPix] & 0x000000FFu);
				windowG[l] = (srcPix[idxPix] & 0x0000FF00u) >> 8;
				windowR[l] = (srcPix[idxPix] & 0x00FF0000u) >> 16;
			
				idxPix += linePitch;
			} /* for (l = 0; l < kernelWidth; l++) */

			insertionsort (windowB, windowB + kernelSize);
			insertionsort (windowG, windowG + kernelSize);
			insertionsort (windowR, windowR + kernelSize);

			idxPix = (j * linePitch + i);

			dstPix[idxPix] = (srcPix[idxPix] & 0xFF000000u) | /* copy ALPHA channel value as is */
									windowB[medianPosition] |
							 (windowG[medianPosition] << 8) |
							 (windowR[medianPosition] << 16);

			n %= kernelWidth;
		} /* for (i = border + 1; i < lastPixel; i++) */

	  /* copy border pixels from end of line */
		idxPix = j * linePitch;
		for (/* i counter already set in exit from main loop */; i < width; i++)
			dstPix[idxPix + i] = srcPix[idxPix + i];

	} /* for (j = border; j < lastLine; j++) */

	/* copy border lines in bottom of frame */
	__VECTOR_ALIGNED__
	for (j = lastLine; j < height; j++)
	{
		idxPix = j * linePitch;
		memcpy(&dstPix[idxPix], &srcPix[idxPix], lineBytesSize);
	}
#endif

	return true;
}


bool median_filter_BGRA_4444_16u_frame (const VideoHandle theData, const csSDK_int32& kernelWidth)
{
	// protection on kernel size
	if (MaxKernelWidth < kernelWidth || 3 > kernelWidth)
		return false;

	CACHE_ALIGN uint32_t windowR[MaxKernelElemSize];
	CACHE_ALIGN uint32_t windowG[MaxKernelElemSize];
	CACHE_ALIGN uint32_t windowB[MaxKernelElemSize];

	prRect box = { 0 };

	// Get the frame dimensions
	((*theData)->piSuites->ppixFuncs->ppixGetBounds)((*theData)->destination, &box);

	// Calculate dimensions
	const csSDK_int32 height = box.bottom - box.top;
	const csSDK_int32 width = box.right - box.left;
	const csSDK_int32 rowbytes = ((*theData)->piSuites->ppixFuncs->ppixGetRowbytes)((*theData)->destination);
	const csSDK_int32 linePitch = rowbytes >> 3; /* lets add additional shift because we interpret image buffer as int64* */

	// Create copies of pointer to the source, destination frames
	const csSDK_uint64* __restrict srcPix = reinterpret_cast<const csSDK_uint64* __restrict>(((*theData)->piSuites->ppixFuncs->ppixGetPixels)((*theData)->source));
	      csSDK_uint64* __restrict dstPix = reinterpret_cast<csSDK_uint64* __restrict>(((*theData)->piSuites->ppixFuncs->ppixGetPixels)((*theData)->destination));

	const csSDK_int32 border = kernelWidth >> 1;
	const csSDK_int32 medianPosition = ((kernelWidth * kernelWidth) >> 1) + 1;
	const csSDK_int32 lastLine = height - border;
	const csSDK_int32 lastPixel = width - border;

	csSDK_int32 idxPix = 0;
	csSDK_int32 i, j, k, l, m;

	constexpr csSDK_int32 bytesPerBand = 2;
	const size_t lineBytesSize = width * bytesPerBand * sizeof(dstPix[0]);

	// copy border lines in top of frame
	__VECTOR_ALIGNED__
	for (j = 0; j < border; j++)
	{
		idxPix = j * linePitch;
		memcpy(&dstPix[idxPix], &srcPix[idxPix], lineBytesSize);
	}

	// perform Median Filter on frame
	__VECTOR_ALIGNED__
	for (j = border; j < lastLine; j++)
	{
		// copy border pixels from start of line
		idxPix = j * linePitch;
		for (i = 0; i < border; i++)
			dstPix[idxPix + i] = srcPix[idxPix + i];

		for (i = border; i < lastPixel; i++)
		{
			// Pick up window elements
			const csSDK_int32 lMin = j - border;
			const csSDK_int32 lMax = j + border;
			const csSDK_int32 mMin = i - border;
			const csSDK_int32 mMax = i + border;

			k = 0;
			for (l = lMin; l <= lMax; l++)
			{
				idxPix = l * linePitch + mMin;
				for (m = mMin; m <= mMax; m++)
				{
					windowB[k] = static_cast<uint32_t> (srcPix[idxPix] & BGRA16_B_Mask);
					windowG[k] = static_cast<uint32_t>((srcPix[idxPix] & BGRA16_G_Mask) >> 16);
					windowR[k] = static_cast<uint32_t>((srcPix[idxPix] & BGRA16_R_Mask) >> 32);
					k++, idxPix++;
				} /* for (m = i - border; m = i + border; i++) */
			} /* for (l = j - border; l < j + border; l++) */

			// Order elements (only half of them)
			gnomesort(windowB, windowB + k);
			gnomesort(windowG, windowG + k);
			gnomesort(windowR, windowR + k);

			idxPix = (j * linePitch + i);

			dstPix[idxPix] = windowB[medianPosition] |
					 (windowG[medianPosition] << 16) |
				static_cast<csSDK_uint64>(windowR[medianPosition]) << 32 |
				(srcPix[idxPix] & BGRA16_A_Mask); /* copy ALPHA channel value as is */

		} /* for (i = border; i < lastPixel; i++) */

		  // copy border pixels from end of line
		idxPix = j * linePitch;
		for (/* i counter already set in exit from main loop */; i < width; i++)
			dstPix[idxPix + i] = srcPix[idxPix + i];

	} /* for (j = border; j < lastLine; j++) */

	// copy border lines in bottom of frame
	__VECTOR_ALIGNED__
	for (j = lastLine; j < height; j++)
	{
		idxPix = j * linePitch;
		memcpy(&dstPix[idxPix], &srcPix[idxPix], lineBytesSize);
	}

	return true;
}


bool median_filter_BGRA_4444_32f_frame(const VideoHandle theData, const csSDK_int32& kernelWidth)
{
	// protection on kernel size
	if (MaxKernelWidth < kernelWidth || 3 > kernelWidth)
		return false;

	CACHE_ALIGN float windowR[MaxKernelElemSize];
	CACHE_ALIGN float windowG[MaxKernelElemSize];
	CACHE_ALIGN float windowB[MaxKernelElemSize];

	prRect box = { 0 };

	// Get the frame dimensions
	((*theData)->piSuites->ppixFuncs->ppixGetBounds)((*theData)->destination, &box);

	// Calculate dimensions
	const csSDK_int32 height = box.bottom - box.top;
	const csSDK_int32 width = box.right - box.left;
	const csSDK_int32 rowbytes = ((*theData)->piSuites->ppixFuncs->ppixGetRowbytes)((*theData)->destination);
	const csSDK_int32 linePitch = rowbytes >> 2;

	const float* __restrict srcPix = reinterpret_cast<const float* __restrict>(((*theData)->piSuites->ppixFuncs->ppixGetPixels)((*theData)->source));
	      float* __restrict dstPix = reinterpret_cast<float* __restrict>(((*theData)->piSuites->ppixFuncs->ppixGetPixels)((*theData)->destination));

	const csSDK_int32 border = kernelWidth >> 1;
	const csSDK_int32 medianPosition = ((kernelWidth * kernelWidth) >> 1) + 1;
	const csSDK_int32 lastLine = height - border;
	const csSDK_int32 lastPixel = width - border;

	csSDK_int32 idxPix = 0;
	csSDK_int32 i, j, k, l, m;

	constexpr int bytesPerBand = 4;
	const size_t lineBytesSize = width * bytesPerBand * sizeof(dstPix[0]);

	// copy border lines in top of frame
	__VECTOR_ALIGNED__
	for (j = 0; j < border; j++)
	{
		idxPix = j * linePitch;
		memcpy(&dstPix[idxPix], &srcPix[idxPix], lineBytesSize);
	}

	// perform Median Filter on frame
	__VECTOR_ALIGNED__
		for (j = border; j < lastLine; j++)
		{
			// copy border pixels from start of line
			idxPix = j * linePitch;
			for (i = 0; i < border; i++)
			{
				const int pixelLineOffset = i * bytesPerBand;
				dstPix[idxPix + i * bytesPerBand]     = srcPix[idxPix + pixelLineOffset];
				dstPix[idxPix + i * bytesPerBand + 1] = srcPix[idxPix + pixelLineOffset + 1];
				dstPix[idxPix + i * bytesPerBand + 2] = srcPix[idxPix + pixelLineOffset + 2];
				dstPix[idxPix + i * bytesPerBand + 3] = srcPix[idxPix + pixelLineOffset + 3];
			}

			for (i = border; i < lastPixel; i++)
			{
				// Pick up window elements
				const csSDK_int32 lMin = j - border;
				const csSDK_int32 lMax = j + border;
				const csSDK_int32 mMin = i - border;
				const csSDK_int32 mMax = i + border;

				k = 0;
				for (l = lMin; l <= lMax; l++)
				{
					idxPix = l * linePitch + mMin * bytesPerBand;
					for (m = mMin; m <= mMax; m++)
					{
						windowB[k] = srcPix[idxPix];
						windowG[k] = srcPix[idxPix + 1];
						windowR[k] = srcPix[idxPix + 2];
						k++, idxPix += 4;
					} /* for (m = i - border; m = i + border; i++) */
				} /* for (l = j - border; l < j + border; l++) */

				  // Order elements (only half of them)
				gnomesort(windowB, windowB + k);
				gnomesort(windowG, windowG + k);
				gnomesort(windowR, windowR + k);

				idxPix = j * linePitch + i * bytesPerBand;

				dstPix[idxPix] =     windowB[medianPosition];
				dstPix[idxPix + 1] = windowG[medianPosition];
				dstPix[idxPix + 2] = windowR[medianPosition];
				dstPix[idxPix + 3] = srcPix [idxPix + 3];

			} /* for (i = border; i < lastPixel; i++) */

			  // copy border pixels from end of line
			idxPix = j * linePitch;
			for (/* i counter already set in exit from main loop */; i < width; i++)
				dstPix[idxPix + i * bytesPerBand] = srcPix[idxPix + i * bytesPerBand];

		} /* for (j = border; j < lastLine; j++) */

		  // copy border lines in bottom of frame
	__VECTOR_ALIGNED__
		for (j = lastLine; j < height; j++)
		{
			idxPix = j * linePitch;
			memcpy(&dstPix[idxPix], &srcPix[idxPix], lineBytesSize);
		}

	return true;
}


bool median_filter_VUYA_4444_8u_frame (const VideoHandle theData, const csSDK_int32& kernelWidth)
{
	// protection on kernel size
	if (MaxKernelWidth < kernelWidth || 3 > kernelWidth)
		return false;

	CACHE_ALIGN uint32_t windowY[MaxKernelElemSize];

	prRect box = { 0 };

	// Get the frame dimensions
	((*theData)->piSuites->ppixFuncs->ppixGetBounds)((*theData)->destination, &box);

	// Calculate dimensions
	const csSDK_int32 height = box.bottom - box.top;
	const csSDK_int32 width = box.right - box.left;
	const csSDK_int32 rowbytes = ((*theData)->piSuites->ppixFuncs->ppixGetRowbytes)((*theData)->destination);
	const csSDK_int32 linePitch = rowbytes >> 2;

	// Create copies of pointer to the source, destination frames
	const csSDK_uint32* __restrict srcPix = reinterpret_cast<const csSDK_uint32* __restrict>(((*theData)->piSuites->ppixFuncs->ppixGetPixels)((*theData)->source));
	      csSDK_uint32* __restrict dstPix = reinterpret_cast<csSDK_uint32* __restrict>(((*theData)->piSuites->ppixFuncs->ppixGetPixels)((*theData)->destination));

	const csSDK_int32 border = kernelWidth >> 1;
	const csSDK_int32 medianPosition = ((kernelWidth * kernelWidth) >> 1) + 1;
	const csSDK_int32 lastLine = height - border;
	const csSDK_int32 lastPixel = width - border;

	csSDK_int32 idxPix = 0;
	csSDK_int32 i, j, k, l, m;

	const size_t lineBytesSize = width * sizeof(dstPix[0]);

	// copy border lines in top of frame
	__VECTOR_ALIGNED__
	for (j = 0; j < border; j++)
	{
		idxPix = j * linePitch;
		memcpy(&dstPix[idxPix], &srcPix[idxPix], lineBytesSize);
	}

	// perform Median Filter on frame
	__VECTOR_ALIGNED__
	for (j = border; j < lastLine; j++)
	{
		// copy border pixels from start of line
		idxPix = j * linePitch;
		for (i = 0; i < border; i++)
			dstPix[idxPix + i] = srcPix[idxPix + i];

		for (i = border; i < lastPixel; i++)
		{
			// Pick up window elements
			const csSDK_int32 lMin = j - border;
			const csSDK_int32 lMax = j + border;
			const csSDK_int32 mMin = i - border;
			const csSDK_int32 mMax = i + border;

			k = 0;
			for (l = lMin; l <= lMax; l++)
			{
				idxPix = l * linePitch + mMin;
				for (m = mMin; m <= mMax; m++)
				{
					windowY[k] = (srcPix[idxPix] & 0x00FF0000u) >> 16;
					k++, idxPix++;
				} /* for (m = i - border; m = i + border; i++) */
			} /* for (l = j - border; l < j + border; l++) */

			// Order elements (only half of them)
			gnomesort(windowY, windowY + k);

			idxPix = (j * linePitch + i);

			dstPix[idxPix] = (srcPix[idxPix] & 0xFF00FFFFu) | /* copy ALPHA channel value as is */
							(windowY[medianPosition] << 16);

		} /* for (i = border; i < lastPixel; i++) */

		  // copy border pixels from end of line
		idxPix = j * linePitch;
		for (/* i counter already set in exit from main loop */; i < width; i++)
			dstPix[idxPix + i] = srcPix[idxPix + i];

	} /* for (j = border; j < lastLine; j++) */

	  // copy border lines in bottom of frame
	__VECTOR_ALIGNED__
	for (j = lastLine; j < height; j++)
	{
		idxPix = j * linePitch;
		memcpy(&dstPix[idxPix], &srcPix[idxPix], lineBytesSize);
	}

	return true;
}


bool median_filter_VUYA_4444_32f_frame(const VideoHandle theData, const csSDK_int32& kernelWidth)
{
	// protection on kernel size
	if (MaxKernelWidth < kernelWidth || 3 > kernelWidth)
		return false;

	CACHE_ALIGN float windowY[MaxKernelElemSize];
	prRect box = { 0 };

	// Get the frame dimensions
	((*theData)->piSuites->ppixFuncs->ppixGetBounds)((*theData)->destination, &box);

	// Calculate dimensions
	const csSDK_int32 height = box.bottom - box.top;
	const csSDK_int32 width = box.right - box.left;
	const csSDK_int32 rowbytes = ((*theData)->piSuites->ppixFuncs->ppixGetRowbytes)((*theData)->destination);
	const csSDK_int32 linePitch = rowbytes >> 2;

	const float* __restrict srcPix = reinterpret_cast<const float* __restrict>(((*theData)->piSuites->ppixFuncs->ppixGetPixels)((*theData)->source));
	      float* __restrict dstPix = reinterpret_cast<float* __restrict>(((*theData)->piSuites->ppixFuncs->ppixGetPixels)((*theData)->destination));

	const csSDK_int32 border = kernelWidth >> 1;
	const csSDK_int32 medianPosition = ((kernelWidth * kernelWidth) >> 1) + 1;
	const csSDK_int32 lastLine = height - border;
	const csSDK_int32 lastPixel = width - border;

	csSDK_int32 idxPix = 0;
	csSDK_int32 i, j, k, l, m;

	constexpr int bytesPerBand = 4;
	const size_t lineBytesSize = width * bytesPerBand * sizeof(dstPix[0]);

	// copy border lines in top of frame
	__VECTOR_ALIGNED__
		for (j = 0; j < border; j++)
		{
			idxPix = j * linePitch;
			memcpy(&dstPix[idxPix], &srcPix[idxPix], lineBytesSize);
		}

	// perform Median Filter on frame
	__VECTOR_ALIGNED__
		for (j = border; j < lastLine; j++)
		{
			// copy border pixels from start of line
			idxPix = j * linePitch;
			for (i = 0; i < border; i++)
			{
				const int pixelLineOffset = i * bytesPerBand;
				dstPix[idxPix + i * bytesPerBand] = srcPix[idxPix + pixelLineOffset];
				dstPix[idxPix + i * bytesPerBand + 1] = srcPix[idxPix + pixelLineOffset + 1];
				dstPix[idxPix + i * bytesPerBand + 2] = srcPix[idxPix + pixelLineOffset + 2];
				dstPix[idxPix + i * bytesPerBand + 3] = srcPix[idxPix + pixelLineOffset + 3];
			}

			for (i = border; i < lastPixel; i++)
			{
				// Pick up window elements
				const csSDK_int32 lMin = j - border;
				const csSDK_int32 lMax = j + border;
				const csSDK_int32 mMin = i - border;
				const csSDK_int32 mMax = i + border;

				k = 0;
				for (l = lMin; l <= lMax; l++)
				{
					idxPix = l * linePitch + mMin * bytesPerBand;
					for (m = mMin; m <= mMax; m++)
					{
						windowY[k] = srcPix[idxPix + 2];
						k++, idxPix += 4;
					} /* for (m = i - border; m = i + border; i++) */
				} /* for (l = j - border; l < j + border; l++) */

				  // Order elements (only half of them)
				gnomesort(windowY, windowY + k);

				idxPix = j * linePitch + i * bytesPerBand;

				dstPix[idxPix    ] = srcPix[idxPix    ];
				dstPix[idxPix + 1] = srcPix[idxPix + 1];
				dstPix[idxPix + 2] = windowY[medianPosition];
				dstPix[idxPix + 3] = srcPix[idxPix + 3];

			} /* for (i = border; i < lastPixel; i++) */

			  // copy border pixels from end of line
			idxPix = j * linePitch;
			for (/* i counter already set in exit from main loop */; i < width; i++)
				dstPix[idxPix + i * bytesPerBand] = srcPix[idxPix + i * bytesPerBand];

		} /* for (j = border; j < lastLine; j++) */

		  // copy border lines in bottom of frame
	__VECTOR_ALIGNED__
		for (j = lastLine; j < height; j++)
		{
			idxPix = j * linePitch;
			memcpy(&dstPix[idxPix], &srcPix[idxPix], lineBytesSize);
		}

	return true;
}


bool median_filter_ARGB_4444_8u_frame(const VideoHandle theData, const csSDK_int32& kernelWidth)
{
	// protection on kernel size
	if (MaxKernelWidth < kernelWidth || 3 > kernelWidth)
		return false;

	CACHE_ALIGN uint32_t windowR[MaxKernelElemSize];
	CACHE_ALIGN uint32_t windowG[MaxKernelElemSize];
	CACHE_ALIGN uint32_t windowB[MaxKernelElemSize];

	prRect box = { 0 };

	// Get the frame dimensions
	((*theData)->piSuites->ppixFuncs->ppixGetBounds)((*theData)->destination, &box);

	// Calculate dimensions
	const csSDK_int32 height = box.bottom - box.top;
	const csSDK_int32 width = box.right - box.left;
	const csSDK_int32 rowbytes = ((*theData)->piSuites->ppixFuncs->ppixGetRowbytes)((*theData)->destination);
	const csSDK_int32 linePitch = rowbytes >> 2;

	// Create copies of pointer to the source, destination frames
	const csSDK_uint32* __restrict srcPix = reinterpret_cast<const csSDK_uint32* __restrict>(((*theData)->piSuites->ppixFuncs->ppixGetPixels)((*theData)->source));
	csSDK_uint32* __restrict dstPix = reinterpret_cast<csSDK_uint32* __restrict>(((*theData)->piSuites->ppixFuncs->ppixGetPixels)((*theData)->destination));

	const csSDK_int32 border = kernelWidth >> 1;
	const csSDK_int32 medianPosition = ((kernelWidth * kernelWidth) >> 1) + 1;
	const csSDK_int32 lastLine = height - border;
	const csSDK_int32 lastPixel = width - border;

	csSDK_int32 idxPix = 0;
	csSDK_int32 i, j, k, l, m;

	const size_t lineBytesSize = width * sizeof(dstPix[0]);

	// copy border lines in top of frame
	__VECTOR_ALIGNED__
		for (j = 0; j < border; j++)
		{
			idxPix = j * linePitch;
			memcpy(&dstPix[idxPix], &srcPix[idxPix], lineBytesSize);
		}

	// perform Median Filter on frame
	__VECTOR_ALIGNED__
		for (j = border; j < lastLine; j++)
		{
			// copy border pixels from start of line
			idxPix = j * linePitch;
			for (i = 0; i < border; i++)
				dstPix[idxPix + i] = srcPix[idxPix + i];

			for (i = border; i < lastPixel; i++)
			{
				// Pick up window elements
				const csSDK_int32 lMin = j - border;
				const csSDK_int32 lMax = j + border;
				const csSDK_int32 mMin = i - border;
				const csSDK_int32 mMax = i + border;

				k = 0;
				for (l = lMin; l <= lMax; l++)
				{
					idxPix = l * linePitch + mMin;
					for (m = mMin; m <= mMax; m++)
					{
						windowR[k] = (srcPix[idxPix] & 0x0000FF00u) >> 8;
						windowG[k] = (srcPix[idxPix] & 0x00FF0000u) >> 16;
						windowB[k] = (srcPix[idxPix] & 0xFF000000u) >> 24;

						k++, idxPix++;
					} /* for (m = i - border; m = i + border; i++) */
				} /* for (l = j - border; l < j + border; l++) */

				  // Order elements (only half of them)
				gnomesort(windowB, windowB + k);
				gnomesort(windowG, windowG + k);
				gnomesort(windowR, windowR + k);

				idxPix = (j * linePitch + i);

				dstPix[idxPix] = (srcPix[idxPix] & 0xFFu)   | /* copy ALPHA channel value as is */
							(windowR[medianPosition] << 8)  |
							(windowG[medianPosition] << 16) |
							(windowB[medianPosition] << 24);

			} /* for (i = border; i < lastPixel; i++) */

			  // copy border pixels from end of line
			idxPix = j * linePitch;
			for (/* i counter already set in exit from main loop */; i < width; i++)
				dstPix[idxPix + i] = srcPix[idxPix + i];

		} /* for (j = border; j < lastLine; j++) */

		  // copy border lines in bottom of frame
	__VECTOR_ALIGNED__
		for (j = lastLine; j < height; j++)
		{
			idxPix = j * linePitch;
			memcpy(&dstPix[idxPix], &srcPix[idxPix], lineBytesSize);
		}

	return true;
}


bool fuzzy_median_filter_BGRA_4444_8u_frame (const VideoHandle theData) /* kernel size for Fuzzy Algo always - 3x3 */
{
	return true;
}

bool fuzzy_median_filter_BGRA_4444_16u_frame (const VideoHandle theData) /* kernel size for Fuzzy Algo always - 3x3 */
{
	return true;
}

bool fuzzy_median_filter_BGRA_4444_32f_frame (const VideoHandle theData) /* kernel size for Fuzzy Algo always - 3x3 */
{
	return true;
}

bool fuzzy_median_filter_ARGB_4444_8u_frame (const VideoHandle theData) /* kernel size for Fuzzy Algo always - 3x3 */
{
	return true;
}

bool fuzzy_median_filter_VUYA_4444_8u_frame (const VideoHandle theData) /* kernel size for Fuzzy Algo always - 3x3 */
{
	return true;
}

bool fuzzy_median_filter_VUYA_4444_32f_frame (const VideoHandle theData) /* kernel size for Fuzzy Algo always - 3x3 */
{
	return true;
}
