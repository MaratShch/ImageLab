#include "AdobeImageLabSepia.h"
#include <math.h>

CACHE_ALIGN constexpr float SepiaMatrix[9] = 
{
	0.3930f, 0.7690f, 0.1890f,
	0.3490f, 0.6860f, 0.1680f,
	0.2720f, 0.5340f, 0.1310f
};

// define color space conversion matrix's
CACHE_ALIGN constexpr float RGB2YUV[2][9] =
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

CACHE_ALIGN constexpr float YUV2RGB[2][9] =
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



bool processSepiaBGRA4444_8u_slice (VideoHandle theData)
{
#if !defined __INTEL_COMPILER 
	_MM_SET_FLUSH_ZERO_MODE(_MM_FLUSH_ZERO_ON);
	_MM_SET_DENORMALS_ZERO_MODE(_MM_DENORMALS_ZERO_ON);
#endif

	prRect box = { 0 };

	// Get the frame dimensions
	((*theData)->piSuites->ppixFuncs->ppixGetBounds)((*theData)->destination, &box);

	// Calculate dimensions
	const csSDK_int32 height = box.bottom - box.top;
	const csSDK_int32 width = box.right - box.left;
	const csSDK_int32 rowbytes = ((*theData)->piSuites->ppixFuncs->ppixGetRowbytes)((*theData)->destination);
	const int linePitch = rowbytes >> 2;

	// Create copies of pointer to the source, destination frames
	csSDK_uint32* __restrict srcImg = reinterpret_cast<csSDK_uint32* __restrict>(((*theData)->piSuites->ppixFuncs->ppixGetPixels)((*theData)->source));
	csSDK_uint32* __restrict dstImg = reinterpret_cast<csSDK_uint32* __restrict>(((*theData)->piSuites->ppixFuncs->ppixGetPixels)((*theData)->destination));

	unsigned int alpha;
	float R, G, B;
	int newR, newG, newB;

	for (int j = 0; j < height; j++)
	{
		__VECTOR_ALIGNED__
			for (int i = 0; i < width; i++)
			{
				const csSDK_uint32 BGRAPixel = *srcImg++;

				alpha = BGRAPixel & 0xFF000000u;
				R = static_cast<float>((BGRAPixel & 0x00FF0000) >> 16);
				G = static_cast<float>((BGRAPixel & 0x0000FF00) >> 8);
				B = static_cast<float> (BGRAPixel & 0x000000FF);

				newR = static_cast<int>(R * SepiaMatrix[0] + G * SepiaMatrix[1] + B * SepiaMatrix[2]);
				newG = static_cast<int>(R * SepiaMatrix[3] + G * SepiaMatrix[4] + B * SepiaMatrix[5]);
				newB = static_cast<int>(R * SepiaMatrix[6] + G * SepiaMatrix[7] + B * SepiaMatrix[8]);

				const csSDK_uint32 OutPixel = alpha |
					(CLAMP_RGB8(newR)) << 16 |
					(CLAMP_RGB8(newG)) << 8  |
					(CLAMP_RGB8(newB));

				*dstImg++ = OutPixel;
			}

		srcImg += linePitch - width;
		dstImg += linePitch - width;
	}

	return true;
}


bool processSepiaARGB4444_8u_slice(VideoHandle theData)
{
#if !defined __INTEL_COMPILER 
	_MM_SET_FLUSH_ZERO_MODE(_MM_FLUSH_ZERO_ON);
	_MM_SET_DENORMALS_ZERO_MODE(_MM_DENORMALS_ZERO_ON);
#endif

	prRect box = { 0 };

	// Get the frame dimensions
	((*theData)->piSuites->ppixFuncs->ppixGetBounds)((*theData)->destination, &box);

	// Calculate dimensions
	const csSDK_int32 height = box.bottom - box.top;
	const csSDK_int32 width = box.right - box.left;
	const csSDK_int32 rowbytes = ((*theData)->piSuites->ppixFuncs->ppixGetRowbytes)((*theData)->destination);
	const int linePitch = rowbytes >> 2;

	// Create copies of pointer to the source, destination frames
	csSDK_uint32* __restrict srcImg = reinterpret_cast<csSDK_uint32* __restrict>(((*theData)->piSuites->ppixFuncs->ppixGetPixels)((*theData)->source));
	csSDK_uint32* __restrict dstImg = reinterpret_cast<csSDK_uint32* __restrict>(((*theData)->piSuites->ppixFuncs->ppixGetPixels)((*theData)->destination));

	unsigned int alpha;
	float R, G, B;
	int newR, newG, newB;

	for (int j = 0; j < height; j++)
	{
		__VECTOR_ALIGNED__
			for (int i = 0; i < width; i++)
			{
				const csSDK_uint32 BGRAPixel = *srcImg++;

				alpha = BGRAPixel & 0x000000FFu;
				R = static_cast<float>((BGRAPixel & 0x0000FF00) >> 8);
				G = static_cast<float>((BGRAPixel & 0x00FF0000) >> 16);
				B = static_cast<float>((BGRAPixel & 0xFF000000) >> 24);

				newR = static_cast<int>(R * SepiaMatrix[0] + G * SepiaMatrix[1] + B * SepiaMatrix[2]);
				newG = static_cast<int>(R * SepiaMatrix[3] + G * SepiaMatrix[4] + B * SepiaMatrix[5]);
				newB = static_cast<int>(R * SepiaMatrix[6] + G * SepiaMatrix[7] + B * SepiaMatrix[8]);

				const csSDK_uint32 OutPixel = alpha |
					(CLAMP_RGB8(newR)) << 8  |      
					(CLAMP_RGB8(newG)) << 16 |
					(CLAMP_RGB8(newB)) << 24;

				*dstImg++ = OutPixel;
			}

		srcImg += linePitch - width;
		dstImg += linePitch - width;
	}

	return true;
}

bool processSepiaBGRA4444_16u_slice (VideoHandle theData)
{
#if !defined __INTEL_COMPILER 
	_MM_SET_FLUSH_ZERO_MODE(_MM_FLUSH_ZERO_ON);
	_MM_SET_DENORMALS_ZERO_MODE(_MM_DENORMALS_ZERO_ON);
#endif

	prRect box = { 0 };

	// Get the frame dimensions
	((*theData)->piSuites->ppixFuncs->ppixGetBounds)((*theData)->destination, &box);

	// Calculate dimensions
	const csSDK_int32 height = box.bottom - box.top;
	const csSDK_int32 width = box.right - box.left;
	const csSDK_int32 rowbytes = ((*theData)->piSuites->ppixFuncs->ppixGetRowbytes)((*theData)->destination);
	const csSDK_int32 linePitch = rowbytes >> 2;

	// Create copies of pointer to the source, destination frames
	csSDK_uint32* __restrict srcImg = reinterpret_cast<csSDK_uint32* __restrict>(((*theData)->piSuites->ppixFuncs->ppixGetPixels)((*theData)->source));
	csSDK_uint32* __restrict dstImg = reinterpret_cast<csSDK_uint32* __restrict>(((*theData)->piSuites->ppixFuncs->ppixGetPixels)((*theData)->destination));

	unsigned int alpha;
	float R, G, B;
	int newR, newG, newB, A;

	for (int j = 0; j < height; j++)
	{
		__VECTOR_ALIGNED__
			for (int i = 0; i < width; i++)
			{
				const csSDK_uint32 first  = *srcImg++;
				const csSDK_uint32 second = *srcImg++;

				B = static_cast<float> (first & 0x0000FFFFu);
				G = static_cast<float>((first & 0xFFFF0000u) >> 16);
				R = static_cast<float> (second & 0x0000FFFFu);
				A = second & 0xFFFF0000u;

				newR = static_cast<int>(R * SepiaMatrix[0] + G * SepiaMatrix[1] + B * SepiaMatrix[2]);
				newG = static_cast<int>(R * SepiaMatrix[3] + G * SepiaMatrix[4] + B * SepiaMatrix[5]);
				newB = static_cast<int>(R * SepiaMatrix[6] + G * SepiaMatrix[7] + B * SepiaMatrix[8]);

				*dstImg++ =  CLAMP_RGB16(newB) | 
							(CLAMP_RGB16(newG)) << 16;

				*dstImg++ =  A |
							 CLAMP_RGB16(newR);
				;
			}

		srcImg += (linePitch - width * 2);
		dstImg += (linePitch - width * 2);
	}

	return true;
}


bool processSepiaARGB4444_16u_slice(VideoHandle theData)
{
#if !defined __INTEL_COMPILER 
	_MM_SET_FLUSH_ZERO_MODE(_MM_FLUSH_ZERO_ON);
	_MM_SET_DENORMALS_ZERO_MODE(_MM_DENORMALS_ZERO_ON);
#endif

	prRect box = { 0 };

	// Get the frame dimensions
	((*theData)->piSuites->ppixFuncs->ppixGetBounds)((*theData)->destination, &box);

	// Calculate dimensions
	const csSDK_int32 height = box.bottom - box.top;
	const csSDK_int32 width = box.right - box.left;
	const csSDK_int32 rowbytes = ((*theData)->piSuites->ppixFuncs->ppixGetRowbytes)((*theData)->destination);
	const csSDK_int32 linePitch = rowbytes >> 2;

	// Create copies of pointer to the source, destination frames
	csSDK_uint32* __restrict srcImg = reinterpret_cast<csSDK_uint32* __restrict>(((*theData)->piSuites->ppixFuncs->ppixGetPixels)((*theData)->source));
	csSDK_uint32* __restrict dstImg = reinterpret_cast<csSDK_uint32* __restrict>(((*theData)->piSuites->ppixFuncs->ppixGetPixels)((*theData)->destination));

	unsigned int alpha;
	float R, G, B;
	int newR, newG, newB, A;

	for (int j = 0; j < height; j++)
	{
		__VECTOR_ALIGNED__
			for (int i = 0; i < width; i++)
			{
				const csSDK_uint32 first = *srcImg++;
				const csSDK_uint32 second = *srcImg++;

				A = first & 0x0000FFFFu;
				R = static_cast<float>((first & 0xFFFF0000u) >> 16);
				G = static_cast<float> (second & 0x0000FFFFu);
				B = static_cast<float>((second & 0xFFFF0000u) >> 16);

				newR = static_cast<int>(R * SepiaMatrix[0] + G * SepiaMatrix[1] + B * SepiaMatrix[2]);
				newG = static_cast<int>(R * SepiaMatrix[3] + G * SepiaMatrix[4] + B * SepiaMatrix[5]);
				newB = static_cast<int>(R * SepiaMatrix[6] + G * SepiaMatrix[7] + B * SepiaMatrix[8]);

				*dstImg++ = A |
					(CLAMP_RGB16(newR)) << 16;

				*dstImg++ = CLAMP_RGB16(newG) |
					(CLAMP_RGB16(newB)) << 16;
			}

		srcImg += (linePitch - width * 2);
		dstImg += (linePitch - width * 2);
	}

	return true;
}


bool processSepiaBGRA4444_32f_slice (VideoHandle theData)
{
#if !defined __INTEL_COMPILER 
	_MM_SET_FLUSH_ZERO_MODE(_MM_FLUSH_ZERO_ON);
	_MM_SET_DENORMALS_ZERO_MODE(_MM_DENORMALS_ZERO_ON);
#endif

	prRect box = { 0 };

	// Get the frame dimensions
	((*theData)->piSuites->ppixFuncs->ppixGetBounds)((*theData)->destination, &box);

	// Calculate dimensions
	const csSDK_int32 height = box.bottom - box.top;
	const csSDK_int32 width = box.right - box.left;
	const csSDK_int32 rowbytes = ((*theData)->piSuites->ppixFuncs->ppixGetRowbytes)((*theData)->destination);
	const csSDK_int32 linePitch = rowbytes >> 2;

	// Create copies of pointer to the source, destination frames
	float* __restrict srcImg = reinterpret_cast<float* __restrict>(((*theData)->piSuites->ppixFuncs->ppixGetPixels)((*theData)->source));
	float* __restrict dstImg = reinterpret_cast<float* __restrict>(((*theData)->piSuites->ppixFuncs->ppixGetPixels)((*theData)->destination));

	float R, G, B;
	float newR, newG, newB, A;

	for (int j = 0; j < height; j++)
	{
		__VECTOR_ALIGNED__
			for (int i = 0; i < width; i++)
			{
				B = *srcImg++;
				G = *srcImg++;
				R = *srcImg++;
				A = *srcImg++;

				newR = R * SepiaMatrix[0] + G * SepiaMatrix[1] + B * SepiaMatrix[2];
				newG = R * SepiaMatrix[3] + G * SepiaMatrix[4] + B * SepiaMatrix[5];
				newB = R * SepiaMatrix[6] + G * SepiaMatrix[7] + B * SepiaMatrix[8];

				*dstImg++ = newB;
				*dstImg++ = newG;
				*dstImg++ = newR;
				*dstImg++ = A;
			}

		srcImg += (linePitch - width * 4);
		dstImg += (linePitch - width * 4);

	}

	return true;
}

bool processSepiaARGB4444_32f_slice(VideoHandle theData)
{
#if !defined __INTEL_COMPILER 
	_MM_SET_FLUSH_ZERO_MODE(_MM_FLUSH_ZERO_ON);
	_MM_SET_DENORMALS_ZERO_MODE(_MM_DENORMALS_ZERO_ON);
#endif

	prRect box = { 0 };

	// Get the frame dimensions
	((*theData)->piSuites->ppixFuncs->ppixGetBounds)((*theData)->destination, &box);

	// Calculate dimensions
	const csSDK_int32 height = box.bottom - box.top;
	const csSDK_int32 width = box.right - box.left;
	const csSDK_int32 rowbytes = ((*theData)->piSuites->ppixFuncs->ppixGetRowbytes)((*theData)->destination);
	const csSDK_int32 linePitch = rowbytes >> 2;

	// Create copies of pointer to the source, destination frames
	float* __restrict srcImg = reinterpret_cast<float* __restrict>(((*theData)->piSuites->ppixFuncs->ppixGetPixels)((*theData)->source));
	float* __restrict dstImg = reinterpret_cast<float* __restrict>(((*theData)->piSuites->ppixFuncs->ppixGetPixels)((*theData)->destination));

	float R, G, B;
	float newR, newG, newB, A;

	for (int j = 0; j < height; j++)
	{
		__VECTOR_ALIGNED__
			for (int i = 0; i < width; i++)
			{
				A = *srcImg++;
				R = *srcImg++;
				G = *srcImg++;
				B = *srcImg++;

				newR = R * SepiaMatrix[0] + G * SepiaMatrix[1] + B * SepiaMatrix[2];
				newG = R * SepiaMatrix[3] + G * SepiaMatrix[4] + B * SepiaMatrix[5];
				newB = R * SepiaMatrix[6] + G * SepiaMatrix[7] + B * SepiaMatrix[8];

				*dstImg++ = A;
				*dstImg++ = newR;
				*dstImg++ = newG;
				*dstImg++ = newB;
			}

		srcImg += (linePitch - width * 4);
		dstImg += (linePitch - width * 4);

	}

	return true;
}

bool processSepiaRGB444_10u_slice (VideoHandle theData)
{
#if !defined __INTEL_COMPILER 
	_MM_SET_FLUSH_ZERO_MODE(_MM_FLUSH_ZERO_ON);
	_MM_SET_DENORMALS_ZERO_MODE(_MM_DENORMALS_ZERO_ON);
#endif

	prRect box = { 0 };

	// Get the frame dimensions
	((*theData)->piSuites->ppixFuncs->ppixGetBounds)((*theData)->destination, &box);

	// Calculate dimensions
	const csSDK_int32 height = box.bottom - box.top;
	const csSDK_int32 width = box.right - box.left;
	const csSDK_int32 rowbytes = ((*theData)->piSuites->ppixFuncs->ppixGetRowbytes)((*theData)->destination);
	const int linePitch = rowbytes >> 2;

	// Create copies of pointer to the source, destination frames
	csSDK_uint32* __restrict srcImg = reinterpret_cast<csSDK_uint32* __restrict>(((*theData)->piSuites->ppixFuncs->ppixGetPixels)((*theData)->source));
	csSDK_uint32* __restrict dstImg = reinterpret_cast<csSDK_uint32* __restrict>(((*theData)->piSuites->ppixFuncs->ppixGetPixels)((*theData)->destination));

	float R, G, B;
	int newR, newG, newB;

	for (int j = 0; j < height; j++)
	{
		__VECTOR_ALIGNED__
			for (int i = 0; i < width; i++)
			{
				const csSDK_uint32 BGRAPixel = *srcImg++;

				B = static_cast<float>((BGRAPixel & 0x00000FFC) >> 2);
				G = static_cast<float>((BGRAPixel & 0x003FF000) >> 12);
				R = static_cast<float>((BGRAPixel & 0xFFC00000) >> 22);

				newR = static_cast<int>(R * SepiaMatrix[0] + G * SepiaMatrix[1] + B * SepiaMatrix[2]);
				newG = static_cast<int>(R * SepiaMatrix[3] + G * SepiaMatrix[4] + B * SepiaMatrix[5]);
				newB = static_cast<int>(R * SepiaMatrix[6] + G * SepiaMatrix[7] + B * SepiaMatrix[8]);

				const csSDK_uint32 SepiaPixel =
										CLAMP_RGB10(newB) << 2  |
										CLAMP_RGB10(newG) << 12 |
										CLAMP_RGB10(newR) << 22;

				*dstImg++ = SepiaPixel;
			}

		srcImg += linePitch - width;
		dstImg += linePitch - width;
	}

	return true;
}


bool processSepiaVUYA4444_8u_BT601_slice (VideoHandle theData)
{
#if !defined __INTEL_COMPILER 
	_MM_SET_FLUSH_ZERO_MODE(_MM_FLUSH_ZERO_ON);
	_MM_SET_DENORMALS_ZERO_MODE(_MM_DENORMALS_ZERO_ON);
#endif

	prRect box = { 0 };

	// Get the frame dimensions
	((*theData)->piSuites->ppixFuncs->ppixGetBounds)((*theData)->destination, &box);

	// Calculate dimensions
	const csSDK_int32 height = box.bottom - box.top;
	const csSDK_int32 width = box.right - box.left;
	const csSDK_int32 rowbytes = ((*theData)->piSuites->ppixFuncs->ppixGetRowbytes)((*theData)->destination);
	const int linePitch = rowbytes >> 2;

	// Create copies of pointer to the source, destination frames
	csSDK_uint32* __restrict srcImg = reinterpret_cast<csSDK_uint32* __restrict>(((*theData)->piSuites->ppixFuncs->ppixGetPixels)((*theData)->source));
	csSDK_uint32* __restrict dstImg = reinterpret_cast<csSDK_uint32* __restrict>(((*theData)->piSuites->ppixFuncs->ppixGetPixels)((*theData)->destination));

	unsigned int alpha;
	float R, G, B;
	float Y, U, V;
	float newR, newG, newB;
	int newY, newU, newV;

	for (int j = 0; j < height; j++)
	{
		__VECTOR_ALIGNED__
			for (int i = 0; i < width; i++)
			{
				alpha = *srcImg & 0xFF000000u;
				Y = static_cast<float>(static_cast<int>((*srcImg & 0x00FF0000) >> 16));
				U = static_cast<float>(static_cast<int>((*srcImg & 0x0000FF00) >> 8) - 128);
				V = static_cast<float>(static_cast<int>(*srcImg & 0x000000FF) - 128);
				srcImg++;

				R = Y * YUV2RGB[0][0] + U * YUV2RGB[0][1] + V * YUV2RGB[0][2];
				G = Y * YUV2RGB[0][3] + U * YUV2RGB[0][4] + V * YUV2RGB[0][5];
				B = Y * YUV2RGB[0][6] + U * YUV2RGB[0][7] + V * YUV2RGB[0][8];

				newR = R * SepiaMatrix[0] + G * SepiaMatrix[1] + B * SepiaMatrix[2];
				newG = R * SepiaMatrix[3] + G * SepiaMatrix[4] + B * SepiaMatrix[5];
				newB = R * SepiaMatrix[6] + G * SepiaMatrix[7] + B * SepiaMatrix[8];

				newY = static_cast<int>(newR * RGB2YUV[0][0] + newG * RGB2YUV[0][1] + newB * RGB2YUV[0][2]);
				newU = static_cast<int>(newR * RGB2YUV[0][3] + newG * RGB2YUV[0][4] + newB * RGB2YUV[0][5]) + 128;
				newV = static_cast<int>(newR * RGB2YUV[0][6] + newG * RGB2YUV[0][7] + newB * RGB2YUV[0][8]) + 128;

				*dstImg++ = alpha			|
					CLAMP_RGB8(newY) << 16  |
					CLAMP_RGB8(newU) << 8   |
					CLAMP_RGB8(newV);
			}

		srcImg += linePitch - width;
		dstImg += linePitch - width;
	}

	return true;
}

bool processSepiaVUYA4444_8u_BT709_slice(VideoHandle theData)
{
#if !defined __INTEL_COMPILER 
	_MM_SET_FLUSH_ZERO_MODE(_MM_FLUSH_ZERO_ON);
	_MM_SET_DENORMALS_ZERO_MODE(_MM_DENORMALS_ZERO_ON);
#endif

	prRect box = { 0 };

	// Get the frame dimensions
	((*theData)->piSuites->ppixFuncs->ppixGetBounds)((*theData)->destination, &box);

	// Calculate dimensions
	const csSDK_int32 height = box.bottom - box.top;
	const csSDK_int32 width = box.right - box.left;
	const csSDK_int32 rowbytes = ((*theData)->piSuites->ppixFuncs->ppixGetRowbytes)((*theData)->destination);
	const int linePitch = rowbytes >> 2;

	// Create copies of pointer to the source, destination frames
	csSDK_uint32* __restrict srcImg = reinterpret_cast<csSDK_uint32* __restrict>(((*theData)->piSuites->ppixFuncs->ppixGetPixels)((*theData)->source));
	csSDK_uint32* __restrict dstImg = reinterpret_cast<csSDK_uint32* __restrict>(((*theData)->piSuites->ppixFuncs->ppixGetPixels)((*theData)->destination));

	unsigned int alpha;
	float R, G, B;
	float Y, U, V;
	float newR, newG, newB;
	int newY, newU, newV;

	for (int j = 0; j < height; j++)
	{
		__VECTOR_ALIGNED__
			for (int i = 0; i < width; i++)
			{
				alpha = *srcImg & 0xFF000000u;
				Y = static_cast<float>(static_cast<int>((*srcImg & 0x00FF0000) >> 16));
				U = static_cast<float>(static_cast<int>((*srcImg & 0x0000FF00) >> 8) - 128);
				V = static_cast<float>(static_cast<int>(*srcImg & 0x000000FF) - 128);
				srcImg++;

				R = Y * YUV2RGB[1][0] + U * YUV2RGB[1][1] + V * YUV2RGB[1][2];
				G = Y * YUV2RGB[1][3] + U * YUV2RGB[1][4] + V * YUV2RGB[1][5];
				B = Y * YUV2RGB[1][6] + U * YUV2RGB[1][7] + V * YUV2RGB[1][8];

				newR = R * SepiaMatrix[0] + G * SepiaMatrix[1] + B * SepiaMatrix[2];
				newG = R * SepiaMatrix[3] + G * SepiaMatrix[4] + B * SepiaMatrix[5];
				newB = R * SepiaMatrix[6] + G * SepiaMatrix[7] + B * SepiaMatrix[8];

				newY = static_cast<int>(newR * RGB2YUV[1][0] + newG * RGB2YUV[1][1] + newB * RGB2YUV[1][2]);
				newU = static_cast<int>(newR * RGB2YUV[1][3] + newG * RGB2YUV[1][4] + newB * RGB2YUV[1][5]) + 128;
				newV = static_cast<int>(newR * RGB2YUV[1][6] + newG * RGB2YUV[1][7] + newB * RGB2YUV[1][8]) + 128;

				*dstImg++ = alpha |
					CLAMP_RGB8(newY) << 16 |
					CLAMP_RGB8(newU) << 8 |
					CLAMP_RGB8(newV);
			}

		srcImg += linePitch - width;
		dstImg += linePitch - width;
	}

	return true;
}

bool processSepiaVUYA4444_32f_BT601_slice(VideoHandle theData)
{
#if !defined __INTEL_COMPILER 
	_MM_SET_FLUSH_ZERO_MODE(_MM_FLUSH_ZERO_ON);
	_MM_SET_DENORMALS_ZERO_MODE(_MM_DENORMALS_ZERO_ON);
#endif

	prRect box = { 0 };

	// Get the frame dimensions
	((*theData)->piSuites->ppixFuncs->ppixGetBounds)((*theData)->destination, &box);

	// Calculate dimensions
	const csSDK_int32 height = box.bottom - box.top;
	const csSDK_int32 width = box.right - box.left;
	const csSDK_int32 rowbytes = ((*theData)->piSuites->ppixFuncs->ppixGetRowbytes)((*theData)->destination);
	const csSDK_int32 linePitch = rowbytes >> 2;

	// Create copies of pointer to the source, destination frames
	float* __restrict srcImg = reinterpret_cast<float* __restrict>(((*theData)->piSuites->ppixFuncs->ppixGetPixels)((*theData)->source));
	float* __restrict dstImg = reinterpret_cast<float* __restrict>(((*theData)->piSuites->ppixFuncs->ppixGetPixels)((*theData)->destination));

	float Y, U, V, A;
	float R, G, B;
	float newR, newG, newB;
	float newY, newU, newV;

	for (int j = 0; j < height; j++)
	{
		__VECTOR_ALIGNED__
		for (int i = 0; i < width; i++)
		{
			// get YUV components as 32f values
			V = *srcImg++;
			U = *srcImg++;
			Y = *srcImg++;
			// get ALPHA as 32f value
			A = *srcImg++;

			R = Y * YUV2RGB[0][0] + U * YUV2RGB[0][1] + V * YUV2RGB[0][2];
			G = Y * YUV2RGB[0][3] + U * YUV2RGB[0][4] + V * YUV2RGB[0][5];
			B = Y * YUV2RGB[0][6] + U * YUV2RGB[0][7] + V * YUV2RGB[0][8];

			newR = CLAMP_RGB8(R * SepiaMatrix[0] + G * SepiaMatrix[1] + B * SepiaMatrix[2]);
			newG = CLAMP_RGB8(R * SepiaMatrix[3] + G * SepiaMatrix[4] + B * SepiaMatrix[5]);
			newB = CLAMP_RGB8(R * SepiaMatrix[6] + G * SepiaMatrix[7] + B * SepiaMatrix[8]);

			newY = newR * RGB2YUV[0][0] + newG * RGB2YUV[0][1] + newB * RGB2YUV[0][2];
			newU = newR * RGB2YUV[0][3] + newG * RGB2YUV[0][4] + newB * RGB2YUV[0][5];
			newV = newR * RGB2YUV[0][6] + newG * RGB2YUV[0][7] + newB * RGB2YUV[0][8];

			*dstImg++ = newV;
			*dstImg++ = newU;
			*dstImg++ = newY;
			*dstImg++ = A;
		}

		srcImg += (linePitch - width * 4);
		dstImg += (linePitch - width * 4);

	}

	return true;
}


bool processSepiaVUYA4444_32f_BT709_slice(VideoHandle theData)
{
#if !defined __INTEL_COMPILER 
	_MM_SET_FLUSH_ZERO_MODE(_MM_FLUSH_ZERO_ON);
	_MM_SET_DENORMALS_ZERO_MODE(_MM_DENORMALS_ZERO_ON);
#endif

	prRect box = { 0 };

	// Get the frame dimensions
	((*theData)->piSuites->ppixFuncs->ppixGetBounds)((*theData)->destination, &box);

	// Calculate dimensions
	const csSDK_int32 height = box.bottom - box.top;
	const csSDK_int32 width = box.right - box.left;
	const csSDK_int32 rowbytes = ((*theData)->piSuites->ppixFuncs->ppixGetRowbytes)((*theData)->destination);
	const csSDK_int32 linePitch = rowbytes >> 2;

	// Create copies of pointer to the source, destination frames
	float* __restrict srcImg = reinterpret_cast<float* __restrict>(((*theData)->piSuites->ppixFuncs->ppixGetPixels)((*theData)->source));
	float* __restrict dstImg = reinterpret_cast<float* __restrict>(((*theData)->piSuites->ppixFuncs->ppixGetPixels)((*theData)->destination));

	float Y, U, V, A;
	float R, G, B;
	float newR, newG, newB;
	float newY, newU, newV;

	for (int j = 0; j < height; j++)
	{
		__VECTOR_ALIGNED__
			for (int i = 0; i < width; i++)
			{
				// get YUV components as 32f values
				V = *srcImg++;
				U = *srcImg++;
				Y = *srcImg++;
				// get ALPHA as 32f value
				A = *srcImg++;

				R = Y * YUV2RGB[1][0] + U * YUV2RGB[1][1] + V * YUV2RGB[1][2];
				G = Y * YUV2RGB[1][3] + U * YUV2RGB[1][4] + V * YUV2RGB[1][5];
				B = Y * YUV2RGB[1][6] + U * YUV2RGB[1][7] + V * YUV2RGB[1][8];

				newR = CLAMP_RGB8(R * SepiaMatrix[0] + G * SepiaMatrix[1] + B * SepiaMatrix[2]);
				newG = CLAMP_RGB8(R * SepiaMatrix[3] + G * SepiaMatrix[4] + B * SepiaMatrix[5]);
				newB = CLAMP_RGB8(R * SepiaMatrix[6] + G * SepiaMatrix[7] + B * SepiaMatrix[8]);

				newY = newR * RGB2YUV[1][0] + newG * RGB2YUV[1][1] + newB * RGB2YUV[1][2];
				newU = newR * RGB2YUV[1][3] + newG * RGB2YUV[1][4] + newB * RGB2YUV[1][5];
				newV = newR * RGB2YUV[1][6] + newG * RGB2YUV[1][7] + newB * RGB2YUV[1][8];

				*dstImg++ = newV;
				*dstImg++ = newU;
				*dstImg++ = newY;
				*dstImg++ = A;
			}

		srcImg += (linePitch - width * 4);
		dstImg += (linePitch - width * 4);

	}

	return true;
}



csSDK_int32 selectProcessFunction(VideoHandle theData)
{
	static constexpr char* strPpixSuite = "Premiere PPix Suite";
	SPBasicSuite*		   SPBasic = nullptr;
	csSDK_int32 errCode = fsBadFormatIndex;
	bool processSucceed = true;

	// acquire Premier Suites
	if (nullptr != (SPBasic = (*theData)->piSuites->utilFuncs->getSPBasicSuite()))
	{
		PrSDKPPixSuite*			PPixSuite = nullptr;
		SPBasic->AcquireSuite(strPpixSuite, 1, (const void**)&PPixSuite);

		if (nullptr != PPixSuite)
		{
			PrPixelFormat pixelFormat = PrPixelFormat_Invalid;
			PPixSuite->GetPixelFormat((*theData)->source, &pixelFormat);

			switch (pixelFormat)
			{
				// ============ native AP formats ============================= //
				case PrPixelFormat_BGRA_4444_8u:
					processSucceed = processSepiaBGRA4444_8u_slice (theData);
				break;

				case PrPixelFormat_VUYA_4444_8u:
					processSucceed = processSepiaVUYA4444_8u_BT601_slice (theData);
				break;

				case PrPixelFormat_VUYA_4444_8u_709:
					processSucceed = processSepiaVUYA4444_8u_BT709_slice(theData);
				break;

				case PrPixelFormat_BGRA_4444_16u:
					processSucceed = processSepiaBGRA4444_16u_slice (theData);
				break;

				case PrPixelFormat_BGRA_4444_32f:
					processSucceed = processSepiaBGRA4444_32f_slice (theData);
				break;

				case PrPixelFormat_VUYA_4444_32f:
					processSucceed = processSepiaVUYA4444_32f_BT601_slice(theData);
				break;

				case PrPixelFormat_VUYA_4444_32f_709:
					processSucceed = processSepiaVUYA4444_32f_BT709_slice(theData);
				break;

				// ============ native AE formats ============================= //
				case PrPixelFormat_ARGB_4444_8u:
					processSucceed = processSepiaARGB4444_8u_slice(theData);
				break;

				case PrPixelFormat_ARGB_4444_16u:
					processSucceed = processSepiaARGB4444_16u_slice(theData);
				break;

				case PrPixelFormat_ARGB_4444_32f:
					processSucceed = processSepiaARGB4444_32f_slice(theData);
				break;

				// =========== miscellanous formats =========================== //
				case PrPixelFormat_RGB_444_10u:
					processSucceed = processSepiaRGB444_10u_slice(theData);
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

// ImageLabHDR filter entry point
PREMPLUGENTRY DllExport xFilter(short selector, VideoHandle theData)
{
	csSDK_int32		errCode = fsNoErr;

	switch (selector)
	{
		case fsInitSpec:
		break;

		case fsSetup:
		break;

		case fsExecute:
			errCode = selectProcessFunction(theData);
		break;

		case fsDisposeData:
		break;

		case fsCanHandlePAR:
			errCode = prEffectCanHandlePAR;
		break;
			
		case fsGetPixelFormatsSupported:
			errCode = imageLabPixelFormatSupported(theData);
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
