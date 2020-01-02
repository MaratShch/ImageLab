#include "ImageLabMonoTonal.h"

CACHE_ALIGN float constexpr RGB2YUV[2][9] =
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

CACHE_ALIGN float constexpr YUV2RGB[2][9] =
{
	// BT.601
	{
		1.000000f,  0.000000f,  1.407500f,
		1.000000f, -0.344140f, -0.716900f,
		1.000000f,  1.779000f,  0.000000f
	},

	// BT.709
	{
		1.000000f,  0.000000f,  1.57480f,
		1.000000f, -0.187327f, -0.46812f,
		1.000000f,  1.855599f,  0.00000f
	}
};

bool copy_4444_8u_frame(const VideoHandle theData)
{
	prRect box = {};
	// Get the frame dimensions
	((*theData)->piSuites->ppixFuncs->ppixGetBounds)((*theData)->destination, &box);

	// Calculate dimensions
	const csSDK_int32 height = box.bottom - box.top;
	const csSDK_int32 width  = box.right - box.left;
	const csSDK_int32 rowbytes = ((*theData)->piSuites->ppixFuncs->ppixGetRowbytes)((*theData)->destination);
	const csSDK_int32 linePitch = rowbytes >> 2;
	const csSDK_int32 nextLine = linePitch - width;

	const csSDK_uint32* __restrict srcPix = reinterpret_cast<csSDK_uint32* __restrict>(((*theData)->piSuites->ppixFuncs->ppixGetPixels)((*theData)->source));
	      csSDK_uint32* __restrict dstPix = reinterpret_cast<csSDK_uint32* __restrict>(((*theData)->piSuites->ppixFuncs->ppixGetPixels)((*theData)->destination));

	csSDK_int32 i, j;

	for (j = 0; j < height; j++)
	{
		__VECTOR_ALIGNED__
		for (i = 0; i < width; i++)
		{
			*dstPix++ = *srcPix++;
		}

		srcPix += nextLine;
		dstPix += nextLine;
	}

	return true;
}


bool copy_4444_16u_frame(const VideoHandle theData)
{
	prRect box = {};
	// Get the frame dimensions
	((*theData)->piSuites->ppixFuncs->ppixGetBounds)((*theData)->destination, &box);

	// Calculate dimensions
	const csSDK_int32 height = box.bottom - box.top;
	const csSDK_int32 width  = box.right - box.left;
	const csSDK_int32 rowbytes  = ((*theData)->piSuites->ppixFuncs->ppixGetRowbytes)((*theData)->destination);
	const csSDK_int32 linePitch = rowbytes >> 2; 

	const csSDK_uint32* __restrict srcPix = reinterpret_cast<csSDK_uint32* __restrict>(((*theData)->piSuites->ppixFuncs->ppixGetPixels)((*theData)->source));
	      csSDK_uint32* __restrict dstPix = reinterpret_cast<csSDK_uint32* __restrict>(((*theData)->piSuites->ppixFuncs->ppixGetPixels)((*theData)->destination));

	csSDK_int32 i, j;

	for (j = 0; j < height; j++)
	{
		__VECTOR_ALIGNED__
		for (i = 0; i < width; i++)
		{
			*dstPix++ = *srcPix++;
			*dstPix++ = *srcPix++;
		}

		srcPix += (linePitch - width * 2);
		dstPix += (linePitch - width * 2);
	}

	return true;
}


bool copy_4444_32f_frame(const VideoHandle theData)
{
	prRect box = {};
	// Get the frame dimensions
	((*theData)->piSuites->ppixFuncs->ppixGetBounds)((*theData)->destination, &box);

	// Calculate dimensions
	const csSDK_int32 height = box.bottom - box.top;
	const csSDK_int32 width  = box.right - box.left;
	const csSDK_int32 rowbytes = ((*theData)->piSuites->ppixFuncs->ppixGetRowbytes)((*theData)->destination);
	const csSDK_int32 linePitch = rowbytes >> 2;

	const float* __restrict srcPix = reinterpret_cast<float* __restrict>(((*theData)->piSuites->ppixFuncs->ppixGetPixels)((*theData)->source));
	      float* __restrict dstPix = reinterpret_cast<float* __restrict>(((*theData)->piSuites->ppixFuncs->ppixGetPixels)((*theData)->destination));

	csSDK_int32 i, j;

	for (j = 0; j < height; j++)
	{
		__VECTOR_ALIGNED__
		for (i = 0; i < width; i++)
		{
			*dstPix++ = *srcPix++;
			*dstPix++ = *srcPix++;
			*dstPix++ = *srcPix++;
			*dstPix++ = *srcPix++;
		}

		srcPix += (linePitch - width * 4);
		dstPix += (linePitch - width * 4);
	}

	return true;
}

inline bool copy_444_10u_frame(const VideoHandle theData)
{
	return copy_4444_8u_frame(theData);
}


bool process_VUYA_4444_8u_frame (const VideoHandle theData, const prColor color, const CONVERT_MATRIX convertMatrix)
{
#if !defined __INTEL_COMPILER 
	_MM_SET_FLUSH_ZERO_MODE(_MM_FLUSH_ZERO_ON);
	_MM_SET_DENORMALS_ZERO_MODE(_MM_DENORMALS_ZERO_ON);
#endif

	prRect box = {};

	// Get the frame dimensions
	((*theData)->piSuites->ppixFuncs->ppixGetBounds)((*theData)->destination, &box);

	// Calculate dimensions
	const csSDK_int32 height = box.bottom - box.top;
	const csSDK_int32 width  = box.right - box.left;
	const csSDK_int32 rowbytes = ((*theData)->piSuites->ppixFuncs->ppixGetRowbytes)((*theData)->destination);
	const csSDK_int32 linePitch = rowbytes >> 2;
	const csSDK_int32 nextLine = linePitch - width;

	const csSDK_uint32* __restrict srcPix = reinterpret_cast<csSDK_uint32* __restrict>(((*theData)->piSuites->ppixFuncs->ppixGetPixels)((*theData)->source));
	      csSDK_uint32* __restrict dstPix = reinterpret_cast<csSDK_uint32* __restrict>(((*theData)->piSuites->ppixFuncs->ppixGetPixels)((*theData)->destination));
	const float*  __restrict pRGB2YUV = RGB2YUV[convertMatrix];

	const float R = static_cast<float>((color & 0xFFu));
	const float G = static_cast<float>((color >> 8) & 0xFFu);
	const float B = static_cast<float>((color >> 16)& 0xFFu);

	const csSDK_uint32 U = CLAMP_RGB8(static_cast<csSDK_uint32>(pRGB2YUV[3] * R + pRGB2YUV[4] * G + pRGB2YUV[5] * B) + 128);
	const csSDK_uint32 V = CLAMP_RGB8(static_cast<csSDK_uint32>(pRGB2YUV[6] * R + pRGB2YUV[7] * G + pRGB2YUV[8] * B) + 128);

	csSDK_int32 i, j;

	for (j = 0; j < height; j++)
	{
		__VECTOR_ALIGNED__
		for (i = 0; i < width; i++)
		{
			*dstPix = (*srcPix & 0xFFFF0000u) | V | (U << 8);
			srcPix++;
			dstPix++;
		} /* for (i = 0; i < width; i++) */

		srcPix += nextLine;
		dstPix += nextLine;

	} /* for (j = 0; j < height; j++) */


	return true;
}

bool process_VUYA_4444_32f_frame(const VideoHandle theData, const prColor color, const CONVERT_MATRIX convertMatrix)
{
#if !defined __INTEL_COMPILER 
	_MM_SET_FLUSH_ZERO_MODE(_MM_FLUSH_ZERO_ON);
	_MM_SET_DENORMALS_ZERO_MODE(_MM_DENORMALS_ZERO_ON);
#endif

	prRect box = {};

	// Get the frame dimensions
	((*theData)->piSuites->ppixFuncs->ppixGetBounds)((*theData)->destination, &box);

	// Calculate dimensions
	const csSDK_int32 height = box.bottom - box.top;
	const csSDK_int32 width = box.right - box.left;
	const csSDK_int32 rowbytes = ((*theData)->piSuites->ppixFuncs->ppixGetRowbytes)((*theData)->destination);
	const csSDK_int32 linePitch = rowbytes >> 2;

	const float* __restrict srcPix = reinterpret_cast<float* __restrict>(((*theData)->piSuites->ppixFuncs->ppixGetPixels)((*theData)->source));
	      float* __restrict dstPix = reinterpret_cast<float* __restrict>(((*theData)->piSuites->ppixFuncs->ppixGetPixels)((*theData)->destination));
	const float* __restrict pRGB2YUV = RGB2YUV[convertMatrix];

	const float R = static_cast<float>((color & 0xFFu))       / 255.0f;
	const float G = static_cast<float>((color >> 8) & 0xFFu)  / 255.0f;
	const float B = static_cast<float>((color >> 16) & 0xFFu) / 255.0f;

	const float U = pRGB2YUV[3] * R + pRGB2YUV[4] * G + pRGB2YUV[5] * B;
	const float V = pRGB2YUV[6] * R + pRGB2YUV[7] * G + pRGB2YUV[8] * B;

	csSDK_int32 i, j;

	for (j = 0; j < height; j++)
	{
		__VECTOR_ALIGNED__
		for (i = 0; i < width; i++)
		{
			*dstPix++ = V;
			*dstPix++ = U;
			srcPix += 2;
			*dstPix++ = *srcPix++;
			*dstPix++ = *srcPix++;
		} /* for (i = 0; i < width; i++) */

		srcPix += (linePitch - width * 4);
		dstPix += (linePitch - width * 4);

	} /* for (j = 0; j < height; j++) */

	return true;
}


bool process_BGRA_4444_8u_frame(const VideoHandle theData, const prColor color)
{
#if !defined __INTEL_COMPILER 
	_MM_SET_FLUSH_ZERO_MODE(_MM_FLUSH_ZERO_ON);
	_MM_SET_DENORMALS_ZERO_MODE(_MM_DENORMALS_ZERO_ON);
#endif

	prRect box = {};
	// Get the frame dimensions
	((*theData)->piSuites->ppixFuncs->ppixGetBounds)((*theData)->destination, &box);

	// Calculate dimensions
	const csSDK_int32 height = box.bottom - box.top;
	const csSDK_int32 width = box.right - box.left;
	const csSDK_int32 rowbytes = ((*theData)->piSuites->ppixFuncs->ppixGetRowbytes)((*theData)->destination);
	const csSDK_int32 linePitch = rowbytes >> 2;
	const csSDK_int32 nextLine = linePitch - width;

	const csSDK_uint32* __restrict srcPix = reinterpret_cast<csSDK_uint32* __restrict>(((*theData)->piSuites->ppixFuncs->ppixGetPixels)((*theData)->source));
	      csSDK_uint32* __restrict dstPix = reinterpret_cast<csSDK_uint32* __restrict>(((*theData)->piSuites->ppixFuncs->ppixGetPixels)((*theData)->destination));

	const float* __restrict pRGB2YUV = (width < 800) ? RGB2YUV[convertBT601] : RGB2YUV[convertBT709];
	const float* __restrict pYUV2RGB = (width < 800) ? YUV2RGB[convertBT601] : YUV2RGB[convertBT709];

	const float R = static_cast<float>(color & 0xFFu);
	const float G = static_cast<float>((color >> 8) & 0xFFu);
	const float B = static_cast<float>((color >> 16) & 0xFFu);

	const float U = pRGB2YUV[3] * R + pRGB2YUV[4] * G + pRGB2YUV[5] * B;
	const float V = pRGB2YUV[6] * R + pRGB2YUV[7] * G + pRGB2YUV[8] * B;

	csSDK_int32 i, j;
	csSDK_int32 newR, newG, newB;
	csSDK_uint32 A;
	float r, g, b, Y;

	for (j = 0; j < height; j++)
	{
		__VECTOR_ALIGNED__
		for (i = 0; i < width; i++)
		{
			const csSDK_uint32 inPixel = *srcPix++;

			b = static_cast<float>(inPixel & 0xFFu);
			g = static_cast<float>((inPixel >> 8)  & 0xFFu);
			r = static_cast<float>((inPixel >> 16) & 0xFFu);
			A = inPixel & 0xFF000000u;

			Y = pRGB2YUV[0] * r + pRGB2YUV[1] * g + pRGB2YUV[2] * b;

			newR = static_cast<csSDK_int32>(pYUV2RGB[0] * Y + pYUV2RGB[1] * U + pYUV2RGB[2] * V);
			newG = static_cast<csSDK_int32>(pYUV2RGB[3] * Y + pYUV2RGB[4] * U + pYUV2RGB[5] * V);
			newB = static_cast<csSDK_int32>(pYUV2RGB[6] * Y + pYUV2RGB[7] * U + pYUV2RGB[8] * V);

			*dstPix++ = A |
						(CLAMP_RGB8(newB)) |
						(CLAMP_RGB8(newG) << 8) |
						(CLAMP_RGB8(newR) << 16);

		} /* for (j = 0; j < height; j++) */

		srcPix += nextLine;
		dstPix += nextLine;

	} /* for (j = 0; j < height; j++) */

	return true;
}


bool process_BGRA_4444_16u_frame(const VideoHandle theData, const prColor color)
{
#if !defined __INTEL_COMPILER 
	_MM_SET_FLUSH_ZERO_MODE(_MM_FLUSH_ZERO_ON);
	_MM_SET_DENORMALS_ZERO_MODE(_MM_DENORMALS_ZERO_ON);
#endif

	prRect box = {};
	// Get the frame dimensions
	((*theData)->piSuites->ppixFuncs->ppixGetBounds)((*theData)->destination, &box);

	// Calculate dimensions
	const csSDK_int32 height = box.bottom - box.top;
	const csSDK_int32 width  = box.right - box.left;
	const csSDK_int32 rowbytes  = ((*theData)->piSuites->ppixFuncs->ppixGetRowbytes)((*theData)->destination);
	const csSDK_int32 linePitch = rowbytes >>2; 

	const csSDK_uint32* __restrict srcPix = reinterpret_cast<csSDK_uint32* __restrict>(((*theData)->piSuites->ppixFuncs->ppixGetPixels)((*theData)->source));
	      csSDK_uint32* __restrict dstPix = reinterpret_cast<csSDK_uint32* __restrict>(((*theData)->piSuites->ppixFuncs->ppixGetPixels)((*theData)->destination));

	const float* __restrict pRGB2YUV = (width < 800) ? RGB2YUV[convertBT601] : RGB2YUV[convertBT709];
	const float* __restrict pYUV2RGB = (width < 800) ? YUV2RGB[convertBT601] : YUV2RGB[convertBT709];

	const float R = static_cast<float> (color & 0xFFu) * 256.0f;
	const float G = static_cast<float>((color >> 8) & 0xFFu) * 256.0f;
	const float B = static_cast<float>((color >> 16) & 0xFFu) * 256.0f;

	const float U = pRGB2YUV[3] * R + pRGB2YUV[4] * G + pRGB2YUV[5] * B;
	const float V = pRGB2YUV[6] * R + pRGB2YUV[7] * G + pRGB2YUV[8] * B;

	csSDK_int32 i, j;
	csSDK_int32 newR, newG, newB;
	csSDK_uint32 A;
	float r, g, b, Y;

	for (j = 0; j < height; j++)
	{
		__VECTOR_ALIGNED__
		for (i = 0; i < width; i++)
		{
			const csSDK_uint32 inPixel1 = *srcPix++;
			const csSDK_uint32 inPixel2 = *srcPix++;

			b = static_cast<float> (inPixel1 & 0xFFFFu);
			g = static_cast<float>((inPixel1 >> 16) & 0xFFFFu);
			r = static_cast<float>(inPixel2 & 0xFFFFu);
			A = inPixel2 & 0xFFFF0000u;

			Y = pRGB2YUV[0] * r + pRGB2YUV[1] * g + pRGB2YUV[2] * b;

			newR = static_cast<csSDK_int32>(pYUV2RGB[0] * Y + pYUV2RGB[1] * U + pYUV2RGB[2] * V);
			newG = static_cast<csSDK_int32>(pYUV2RGB[3] * Y + pYUV2RGB[4] * U + pYUV2RGB[5] * V);
			newB = static_cast<csSDK_int32>(pYUV2RGB[6] * Y + pYUV2RGB[7] * U + pYUV2RGB[8] * V);

			*dstPix++ = CLAMP_RGB16(newB) |
				(CLAMP_RGB16(newG)) << 16;

			*dstPix++ = A |
				CLAMP_RGB16(newR);
		} /* for (j = 0; j < height; j++) */

		srcPix += (linePitch - width * 2);
		dstPix += (linePitch - width * 2);

	} /* for (j = 0; j < height; j++) */

	return true;
}


bool process_BGRA_4444_32f_frame(const VideoHandle theData, const prColor color)
{
#if !defined __INTEL_COMPILER 
	_MM_SET_FLUSH_ZERO_MODE(_MM_FLUSH_ZERO_ON);
	_MM_SET_DENORMALS_ZERO_MODE(_MM_DENORMALS_ZERO_ON);
#endif

	prRect box = {};
	// Get the frame dimensions
	((*theData)->piSuites->ppixFuncs->ppixGetBounds)((*theData)->destination, &box);

	// Calculate dimensions
	const csSDK_int32 height = box.bottom - box.top;
	const csSDK_int32 width = box.right - box.left;
	const csSDK_int32 rowbytes = ((*theData)->piSuites->ppixFuncs->ppixGetRowbytes)((*theData)->destination);
	const csSDK_int32 linePitch = rowbytes >> 2;

	const float* __restrict srcPix = reinterpret_cast<float* __restrict>(((*theData)->piSuites->ppixFuncs->ppixGetPixels)((*theData)->source));
	      float* __restrict dstPix = reinterpret_cast<float* __restrict>(((*theData)->piSuites->ppixFuncs->ppixGetPixels)((*theData)->destination));

	const float* __restrict pRGB2YUV = (width < 800) ? RGB2YUV[convertBT601] : RGB2YUV[convertBT709];
	const float* __restrict pYUV2RGB = (width < 800) ? YUV2RGB[convertBT601] : YUV2RGB[convertBT709];

	const float R = static_cast<float> (color & 0xFFu);
	const float G = static_cast<float>((color >> 8) & 0xFFu);
	const float B = static_cast<float>((color >> 16) & 0xFFu);

	const float U = (pRGB2YUV[3] * R + pRGB2YUV[4] * G + pRGB2YUV[5] * B) / 256.0f;
	const float V = (pRGB2YUV[6] * R + pRGB2YUV[7] * G + pRGB2YUV[8] * B) / 256.0f;

	csSDK_int32 i, j;
	float newR, newG, newB, A;
	float r, g, b, Y;

	for (j = 0; j < height; j++)
	{
		__VECTOR_ALIGNED__
		for (i = 0; i < width; i++)
		{
			b = *srcPix++;
			g = *srcPix++;
			r = *srcPix++;
			A = *srcPix++;

			Y = pRGB2YUV[0] * r + pRGB2YUV[1] * g + pRGB2YUV[2] * b;

			newR = pYUV2RGB[0] * Y + pYUV2RGB[1] * U + pYUV2RGB[2] * V;
			newG = pYUV2RGB[3] * Y + pYUV2RGB[4] * U + pYUV2RGB[5] * V;
			newB = pYUV2RGB[6] * Y + pYUV2RGB[7] * U + pYUV2RGB[8] * V;

			*dstPix++ = newB;
			*dstPix++ = newG;
			*dstPix++ = newR;
			*dstPix++ = A;
		} /* for (j = 0; j < height; j++) */

		srcPix += (linePitch - width * 4);
		dstPix += (linePitch - width * 4);

	} /* for (j = 0; j < height; j++) */

	return true;
}


bool process_ARGB_4444_16u_frame(const VideoHandle theData, const prColor color)
{
#if !defined __INTEL_COMPILER 
	_MM_SET_FLUSH_ZERO_MODE(_MM_FLUSH_ZERO_ON);
	_MM_SET_DENORMALS_ZERO_MODE(_MM_DENORMALS_ZERO_ON);
#endif

	prRect box = {};
	// Get the frame dimensions
	((*theData)->piSuites->ppixFuncs->ppixGetBounds)((*theData)->destination, &box);

	// Calculate dimensions
	const csSDK_int32 height = box.bottom - box.top;
	const csSDK_int32 width = box.right - box.left;
	const csSDK_int32 rowbytes = ((*theData)->piSuites->ppixFuncs->ppixGetRowbytes)((*theData)->destination);
	const csSDK_int32 linePitch = rowbytes >> 2;

	const csSDK_uint32* __restrict srcPix = reinterpret_cast<csSDK_uint32* __restrict>(((*theData)->piSuites->ppixFuncs->ppixGetPixels)((*theData)->source));
	csSDK_uint32* __restrict dstPix = reinterpret_cast<csSDK_uint32* __restrict>(((*theData)->piSuites->ppixFuncs->ppixGetPixels)((*theData)->destination));

	const float* __restrict pRGB2YUV = (width < 800) ? RGB2YUV[convertBT601] : RGB2YUV[convertBT709];
	const float* __restrict pYUV2RGB = (width < 800) ? YUV2RGB[convertBT601] : YUV2RGB[convertBT709];

	const float R = static_cast<float> (color & 0xFFu) * 256.0f;
	const float G = static_cast<float>((color >> 8) & 0xFFu) * 256.0f;
	const float B = static_cast<float>((color >> 16) & 0xFFu) * 256.0f;

	const float U = pRGB2YUV[3] * R + pRGB2YUV[4] * G + pRGB2YUV[5] * B;
	const float V = pRGB2YUV[6] * R + pRGB2YUV[7] * G + pRGB2YUV[8] * B;

	csSDK_int32 i, j;
	csSDK_int32 newR, newG, newB;
	csSDK_uint32 A;
	float r, g, b, Y;

	for (j = 0; j < height; j++)
	{
		__VECTOR_ALIGNED__
		for (i = 0; i < width; i++)
		{
			const csSDK_uint32 inPixel1 = *srcPix++;
			const csSDK_uint32 inPixel2 = *srcPix++;

			A = static_cast<float> (inPixel1 & 0xFFFFu);
			r = static_cast<float>((inPixel1 >> 16) & 0xFFFFu);
			g = static_cast<float>(inPixel2 & 0xFFFFu);
			b = static_cast<float>((inPixel2 >> 16) & 0xFFFFu);

			Y = pRGB2YUV[0] * r + pRGB2YUV[1] * g + pRGB2YUV[2] * b;

			newR = static_cast<csSDK_int32>(pYUV2RGB[0] * Y + pYUV2RGB[1] * U + pYUV2RGB[2] * V);
			newG = static_cast<csSDK_int32>(pYUV2RGB[3] * Y + pYUV2RGB[4] * U + pYUV2RGB[5] * V);
			newB = static_cast<csSDK_int32>(pYUV2RGB[6] * Y + pYUV2RGB[7] * U + pYUV2RGB[8] * V);

			*dstPix++ = A |
						(CLAMP_RGB16(newR)) << 16;

			*dstPix++ = CLAMP_RGB16(newG) |
						(CLAMP_RGB16(newB)) << 16;

		} /* for (j = 0; j < height; j++) */

		srcPix += (linePitch - width * 2);
		dstPix += (linePitch - width * 2);

	} /* for (j = 0; j < height; j++) */

	return true;
}


bool process_ARGB_4444_32f_frame(const VideoHandle theData, const prColor color)
{
#if !defined __INTEL_COMPILER 
	_MM_SET_FLUSH_ZERO_MODE(_MM_FLUSH_ZERO_ON);
	_MM_SET_DENORMALS_ZERO_MODE(_MM_DENORMALS_ZERO_ON);
#endif

	prRect box = {};
	// Get the frame dimensions
	((*theData)->piSuites->ppixFuncs->ppixGetBounds)((*theData)->destination, &box);

	// Calculate dimensions
	const csSDK_int32 height = box.bottom - box.top;
	const csSDK_int32 width  = box.right - box.left;
	const csSDK_int32 rowbytes  = ((*theData)->piSuites->ppixFuncs->ppixGetRowbytes)((*theData)->destination);
	const csSDK_int32 linePitch = rowbytes >> 2;

	const float* __restrict srcPix = reinterpret_cast<float* __restrict>(((*theData)->piSuites->ppixFuncs->ppixGetPixels)((*theData)->source));
	      float* __restrict dstPix = reinterpret_cast<float *__restrict > (((*theData)->piSuites->ppixFuncs->ppixGetPixels)((*theData)->destination));

	const float* __restrict pRGB2YUV = (width < 800) ? RGB2YUV[convertBT601] : RGB2YUV[convertBT709];
	const float* __restrict pYUV2RGB = (width < 800) ? YUV2RGB[convertBT601] : YUV2RGB[convertBT709];

	const float R = static_cast<float> (color & 0xFFu);
	const float G = static_cast<float>((color >> 8) & 0xFFu);
	const float B = static_cast<float>((color >> 16) & 0xFFu);

	const float U = (pRGB2YUV[3] * R + pRGB2YUV[4] * G + pRGB2YUV[5] * B) / 256.0f;
	const float V = (pRGB2YUV[6] * R + pRGB2YUV[7] * G + pRGB2YUV[8] * B) / 256.0f;

	csSDK_int32 i, j;
	float newR, newG, newB, A;
	float r, g, b, Y;

	for (j = 0; j < height; j++)
	{
		__VECTOR_ALIGNED__
		for (i = 0; i < width; i++)
		{
			A = *srcPix++;
			r = *srcPix++;
			g = *srcPix++;
			b = *srcPix++;

			Y = pRGB2YUV[0] * r + pRGB2YUV[1] * g + pRGB2YUV[2] * b;

			newR = pYUV2RGB[0] * Y + pYUV2RGB[1] * U + pYUV2RGB[2] * V;
			newG = pYUV2RGB[3] * Y + pYUV2RGB[4] * U + pYUV2RGB[5] * V;
			newB = pYUV2RGB[6] * Y + pYUV2RGB[7] * U + pYUV2RGB[8] * V;

			*dstPix++ = A;
			*dstPix++ = newR;
			*dstPix++ = newG;
			*dstPix++ = newB;
		} /* for (j = 0; j < height; j++) */

		srcPix += (linePitch - width * 4);
		dstPix += (linePitch - width * 4);

	} /* for (j = 0; j < height; j++) */

	return true;
}


bool process_ARGB_4444_8u_frame(const VideoHandle theData, const prColor color)
{
#if !defined __INTEL_COMPILER 
	_MM_SET_FLUSH_ZERO_MODE(_MM_FLUSH_ZERO_ON);
	_MM_SET_DENORMALS_ZERO_MODE(_MM_DENORMALS_ZERO_ON);
#endif

	prRect box = {};
	// Get the frame dimensions
	((*theData)->piSuites->ppixFuncs->ppixGetBounds)((*theData)->destination, &box);

	// Calculate dimensions
	const csSDK_int32 height = box.bottom - box.top;
	const csSDK_int32 width = box.right - box.left;
	const csSDK_int32 rowbytes = ((*theData)->piSuites->ppixFuncs->ppixGetRowbytes)((*theData)->destination);
	const csSDK_int32 linePitch = rowbytes >> 2;
	const csSDK_int32 nextLine = linePitch - width;

	const csSDK_uint32* __restrict srcPix = reinterpret_cast<csSDK_uint32* __restrict>(((*theData)->piSuites->ppixFuncs->ppixGetPixels)((*theData)->source));
	      csSDK_uint32* __restrict dstPix = reinterpret_cast<csSDK_uint32* __restrict>(((*theData)->piSuites->ppixFuncs->ppixGetPixels)((*theData)->destination));

	const float* __restrict pRGB2YUV = (width < 800) ? RGB2YUV[convertBT601] : RGB2YUV[convertBT709];
	const float* __restrict pYUV2RGB = (width < 800) ? YUV2RGB[convertBT601] : YUV2RGB[convertBT709];

	const float R = static_cast<float>(color & 0xFFu);
	const float G = static_cast<float>((color >> 8) & 0xFFu);
	const float B = static_cast<float>((color >> 16) & 0xFFu);

	const float U = pRGB2YUV[3] * R + pRGB2YUV[4] * G + pRGB2YUV[5] * B;
	const float V = pRGB2YUV[6] * R + pRGB2YUV[7] * G + pRGB2YUV[8] * B;

	csSDK_int32 i, j;
	csSDK_int32 newR, newG, newB;
	csSDK_uint32 A;
	float r, g, b, Y;

	for (j = 0; j < height; j++)
	{
		__VECTOR_ALIGNED__
		for (i = 0; i < width; i++)
		{
			const csSDK_uint32 inPixel = *srcPix++;

			A = inPixel & 0xFFu;
			r = static_cast<float>((inPixel >> 8)  & 0xFFu);
			g = static_cast<float>((inPixel >> 16) & 0xFFu);
			b = static_cast<float>((inPixel >> 24) & 0xFFu);

			Y = pRGB2YUV[0] * r + pRGB2YUV[1] * g + pRGB2YUV[2] * b;

			newR = static_cast<csSDK_int32>(pYUV2RGB[0] * Y + pYUV2RGB[1] * U + pYUV2RGB[2] * V);
			newG = static_cast<csSDK_int32>(pYUV2RGB[3] * Y + pYUV2RGB[4] * U + pYUV2RGB[5] * V);
			newB = static_cast<csSDK_int32>(pYUV2RGB[6] * Y + pYUV2RGB[7] * U + pYUV2RGB[8] * V);

			*dstPix++ = A |
				(CLAMP_RGB8(newR) << 8)  |
				(CLAMP_RGB8(newG) << 16) |
				(CLAMP_RGB8(newB) << 24);

		} /* for (j = 0; j < height; j++) */

		srcPix += nextLine;
		dstPix += nextLine;

	} /* for (j = 0; j < height; j++) */

	return true;
}


bool process_RGB_444_10u_frame (const VideoHandle theData, const prColor color)
{
#if !defined __INTEL_COMPILER 
	_MM_SET_FLUSH_ZERO_MODE(_MM_FLUSH_ZERO_ON);
	_MM_SET_DENORMALS_ZERO_MODE(_MM_DENORMALS_ZERO_ON);
#endif

	prRect box = {};
	// Get the frame dimensions
	((*theData)->piSuites->ppixFuncs->ppixGetBounds)((*theData)->destination, &box);

	// Calculate dimensions
	const csSDK_int32 height = box.bottom - box.top;
	const csSDK_int32 width = box.right - box.left;
	const csSDK_int32 rowbytes = ((*theData)->piSuites->ppixFuncs->ppixGetRowbytes)((*theData)->destination);
	const csSDK_int32 linePitch = rowbytes >> 2;
	const csSDK_int32 nextLine = linePitch - width;

	const csSDK_uint32* __restrict srcPix = reinterpret_cast<csSDK_uint32* __restrict>(((*theData)->piSuites->ppixFuncs->ppixGetPixels)((*theData)->source));
	      csSDK_uint32* __restrict dstPix = reinterpret_cast<csSDK_uint32* __restrict>(((*theData)->piSuites->ppixFuncs->ppixGetPixels)((*theData)->destination));

	const float* __restrict pRGB2YUV = (width < 800) ? RGB2YUV[convertBT601] : RGB2YUV[convertBT709];
	const float* __restrict pYUV2RGB = (width < 800) ? YUV2RGB[convertBT601] : YUV2RGB[convertBT709];

	const float R = static_cast<float>(color & 0xFFu);
	const float G = static_cast<float>((color >> 8) & 0xFFu);
	const float B = static_cast<float>((color >> 16) & 0xFFu);

	const float U = (pRGB2YUV[3] * R + pRGB2YUV[4] * G + pRGB2YUV[5] * B) * 4.0f;
	const float V = (pRGB2YUV[6] * R + pRGB2YUV[7] * G + pRGB2YUV[8] * B) * 4.0f;

	csSDK_int32 i, j;
	csSDK_int32 newR, newG, newB;
	float r, g, b, Y;

	for (j = 0; j < height; j++)
	{
		__VECTOR_ALIGNED__
			for (i = 0; i < width; i++)
			{
				const csSDK_uint32 inPixel = *srcPix++;

				b = static_cast<float>((inPixel & 0x00000FFC) >> 2);
				g = static_cast<float>((inPixel & 0x003FF000) >> 12);
				r = static_cast<float>((inPixel & 0xFFC00000) >> 22);

				Y = pRGB2YUV[0] * r + pRGB2YUV[1] * g + pRGB2YUV[2] * b;

				newR = static_cast<csSDK_int32>(pYUV2RGB[0] * Y + pYUV2RGB[1] * U + pYUV2RGB[2] * V);
				newG = static_cast<csSDK_int32>(pYUV2RGB[3] * Y + pYUV2RGB[4] * U + pYUV2RGB[5] * V);
				newB = static_cast<csSDK_int32>(pYUV2RGB[6] * Y + pYUV2RGB[7] * U + pYUV2RGB[8] * V);

				*dstPix++ = 
							(CLAMP_RGB10(newB) << 2)  |
							(CLAMP_RGB10(newG) << 12) |
							(CLAMP_RGB10(newR) << 22);

			} /* for (j = 0; j < height; j++) */

		srcPix += nextLine;
		dstPix += nextLine;

	} /* for (j = 0; j < height; j++) */

	return true;
}




inline const prColor getSelectedColor (const FilterParamsHandle pHandle)
{
	prColor color = 0u;
	
	if (nullptr != pHandle)
	{
		color = 0x00FFFFFF & (*pHandle)->Color;
		if (0 != color)
		{
			(*pHandle)->isInitialized |= 0x1u;
		}
	}
	return color;
}

inline const uint32_t isColorSelected(const FilterParamsHandle pHandle)
{
	return (nullptr != pHandle) ? (*pHandle)->isInitialized : 0x0u;
}



csSDK_int32 selectProcessFunction(const VideoHandle theData)
{
	static constexpr char* strPpixSuite = "Premiere PPix Suite";
	SPBasicSuite*		   SPBasic = nullptr;
	csSDK_int32 errCode = fsBadFormatIndex;
	bool processSucceed = true;

	const FilterParamsHandle filterParamH = reinterpret_cast<FilterParamsHandle>((*theData)->specsHandle);;

	// acquire Premier Suites
	if (nullptr != (SPBasic = (*theData)->piSuites->utilFuncs->getSPBasicSuite()))
	{
		PrSDKPPixSuite*			PPixSuite = nullptr;
		const SPErr err = SPBasic->AcquireSuite(strPpixSuite, 1, (const void**)&PPixSuite);

		if (nullptr != PPixSuite && kSPNoError == err)
		{
			PrPixelFormat pixelFormat = PrPixelFormat_Invalid;
			PPixSuite->GetPixelFormat((*theData)->source, &pixelFormat);

			const prColor color = getSelectedColor(filterParamH);
			const uint32_t isSelected = isColorSelected(filterParamH);

			switch (pixelFormat)
			{
					// ============ native AP formats ============================= //
				case PrPixelFormat_BGRA_4444_8u:
					processSucceed = (isSelected != 0u) ?
						process_BGRA_4444_8u_frame(theData, color) : copy_4444_8u_frame(theData);
				break;

				case PrPixelFormat_VUYA_4444_8u:
					processSucceed = (isSelected != 0u) ?
						process_VUYA_4444_8u_frame (theData, color, convertBT601) : copy_4444_8u_frame (theData);
				break;

				case PrPixelFormat_VUYA_4444_8u_709:
					processSucceed = (isSelected != 0u) ?
						process_VUYA_4444_8u_frame(theData, color, convertBT709) : copy_4444_8u_frame(theData);
				break;

				case PrPixelFormat_BGRA_4444_16u:
					processSucceed = (isSelected != 0u) ?
						process_BGRA_4444_16u_frame(theData, color) : copy_4444_16u_frame(theData);
				break;

				case PrPixelFormat_BGRA_4444_32f:
					processSucceed = (isSelected != 0u) ?
						process_BGRA_4444_32f_frame(theData, color) : copy_4444_32f_frame(theData);
				break;

				case PrPixelFormat_VUYA_4444_32f:
					processSucceed = (isSelected != 0u) ?
						process_VUYA_4444_32f_frame(theData, color, convertBT601) : copy_4444_32f_frame(theData);
				break;

				case PrPixelFormat_VUYA_4444_32f_709:
					processSucceed = (isSelected != 0u) ?
						process_VUYA_4444_32f_frame(theData, color, convertBT709) : copy_4444_32f_frame(theData);
				break;

					// ============ native AE formats ============================= //
				case PrPixelFormat_ARGB_4444_8u:
					processSucceed = (isSelected != 0u) ?
						process_ARGB_4444_8u_frame(theData, color) : copy_4444_8u_frame(theData);
				break;

				case PrPixelFormat_ARGB_4444_16u:
					processSucceed = (isSelected != 0u) ?
						process_ARGB_4444_16u_frame(theData, color) : copy_4444_16u_frame(theData);
				break;

				case PrPixelFormat_ARGB_4444_32f:
					processSucceed = (isSelected != 0u) ?
						process_ARGB_4444_32f_frame(theData, color) : copy_4444_32f_frame(theData);
				break;

					// =========== miscellanous formats =========================== //
				case PrPixelFormat_RGB_444_10u:
					processSucceed = (isSelected != 0u) ?
						process_RGB_444_10u_frame(theData, color) : copy_444_10u_frame(theData);
				break;

					// =========== Packed uncompressed formats ==================== //
				case PrPixelFormat_YUYV_422_8u_601:
				break;
				case PrPixelFormat_YUYV_422_8u_709:
				break;
				case PrPixelFormat_UYVY_422_8u_601:
				break;
				case PrPixelFormat_UYVY_422_8u_709:
				break;
				case PrPixelFormat_UYVY_422_32f_601:
				break;
				case PrPixelFormat_UYVY_422_32f_709:
				break;

				default:
					processSucceed = false;
				break;
			}

			errCode = (true == processSucceed) ? fsNoErr : errCode;
		}
	}

	return errCode;
}



// Bilateral-RGB filter entry point
PREMPLUGENTRY DllExport xFilter (short selector, VideoHandle theData)
{
	csSDK_int32 errCode = fsNoErr;
	FilterParamsHandle filterParamH = nullptr;

	switch (selector)
	{
		case fsExecute:
			errCode = selectProcessFunction (theData);
		break;

		case fsInitSpec:
			filterParamH = reinterpret_cast<FilterParamsHandle>(((*theData)->piSuites->memFuncs->newHandle)(sizeof(SFilterParams)));
			if (nullptr != filterParamH)
			{
				IMAGE_LAB_FILTER_PARAM_HANDLE_INIT(filterParamH);
				// save the filter parameters inside of Premier handler
				(*theData)->specsHandle = reinterpret_cast<char**>(filterParamH);
			}
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