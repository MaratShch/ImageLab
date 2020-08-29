#include "AdobeImageLabBW.h"
#include <math.h>

constexpr unsigned int BT601  = 0u;
constexpr unsigned int BT709  = 1u;
constexpr unsigned int BT2020 = 2u;

constexpr float fLumaExp = 1.0f / 2.20f;

CACHE_ALIGN constexpr float coeff [][3] = 
{
	// BT.601
	{
		0.2990f,  0.5870f,  0.1140f
	},

	// BT.709
	{
		0.2126f,   0.7152f,  0.0722f
	}
};

// in future todo: for minimize memory usage - dynamically load this tables from file (?!?)
CACHE_ALIGN float R_coeff[256] = {};
CACHE_ALIGN float G_coeff[256] = {};
CACHE_ALIGN float B_coeff[256] = {};

CACHE_ALIGN float R_U10coeff[1024] = {};
CACHE_ALIGN float G_U10coeff[1024] = {};
CACHE_ALIGN float B_U10coeff[1024] = {};

CACHE_ALIGN float R_U16coeff[65536] = {};
CACHE_ALIGN float G_U16coeff[65536] = {};
CACHE_ALIGN float B_U16coeff[65536] = {};


void initCompCoeffcients(void)
{
	int i;

	// prepare 8-bits tables
	__VECTOR_ALIGNED__
	for (i = 0; i < 256; i++)
		R_coeff[i] = 0.2126f * pow(static_cast<float>(i), 2.20f);

	__VECTOR_ALIGNED__
	for (i = 0; i < 256; i++)
		G_coeff[i] = 0.7152f * pow(static_cast<float>(i), 2.20f);

	__VECTOR_ALIGNED__
	for (i = 0; i < 256; i++)
		B_coeff[i] = 0.0722f * pow(static_cast<float>(i), 2.20f);


	// prepare 10-bits tables
	__VECTOR_ALIGNED__
	for (i = 0; i < 1024; i++)
		R_U10coeff[i] = 0.2126f * pow(static_cast<float>(i), 2.20f);

	__VECTOR_ALIGNED__
	for (i = 0; i < 1024; i++)
		G_U10coeff[i] = 0.7152f * pow(static_cast<float>(i), 2.20f);

	__VECTOR_ALIGNED__
	for (i = 0; i < 1024; i++)
		B_U10coeff[i] = 0.0722f * pow(static_cast<float>(i), 2.20f);


	// prepare 16-bits tables
	__VECTOR_ALIGNED__
	for (i = 0; i < 65536; i++)
		R_U16coeff[i] = 0.2126f * pow(static_cast<float>(i), 2.20f);

	__VECTOR_ALIGNED__
	for (i = 0; i < 65536; i++)
		G_U16coeff[i] = 0.7152f * pow(static_cast<float>(i), 2.20f);

	__VECTOR_ALIGNED__
	for (i = 0; i < 65536; i++)
		B_U16coeff[i] = 0.0722f * pow(static_cast<float>(i), 2.20f);

}


bool processVUYA_4444_8u_slice(VideoHandle theData)
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
	
	for (int j = 0; j < height; j++)
	{
		__VECTOR_ALIGNED__
			for (int i = 0; i < width; i++)
			{
				*dstImg++ = (*srcImg++ & 0xFFFF0000u) | 0x00008080u;
			}

		srcImg += linePitch - width;
		dstImg += linePitch - width;
	}

	return true;
}



bool processBGRA4444_8u_slice (VideoHandle theData)
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

	const float* __restrict lpCoeff = (width > 800) ? coeff[BT709] : coeff[BT601];

	unsigned int alpha;
	unsigned int Luma;
	float R, G, B;

	for (int j = 0; j < height; j++)
	{
		__VECTOR_ALIGNED__
		for (int i = 0; i < width; i++)
		{
			const csSDK_uint32 BGRAPixel = *srcImg++;

			alpha = BGRAPixel & 0xFF000000u;
			R = static_cast<float>((BGRAPixel & 0x00FF0000) >> 16);
			G = static_cast<float>((BGRAPixel & 0x0000FF00) >> 8);
			B = static_cast<float>( BGRAPixel & 0x000000FF);

			Luma = static_cast<unsigned int>(R * lpCoeff[0] + G * lpCoeff[1] + B * lpCoeff[2]);

			const csSDK_uint32 BWPixel = alpha | 
									Luma << 16 |
									Luma << 8  |
									Luma;

			*dstImg++ = BWPixel;
		}

		srcImg += linePitch - width;
		dstImg += linePitch - width;
	}

	return true;
}


bool processAdvancedBGRA4444_8u_slice(VideoHandle theData)
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

	float fLuma;
	unsigned int Luma;
	unsigned int alpha;
	int R, G, B;

	for (int j = 0; j < height; j++)
	{
		__VECTOR_ALIGNED__
			for (int i = 0; i < width; i++)
			{
				const csSDK_uint32 BGRAPixel = *srcImg++;

				alpha = BGRAPixel & 0xFF000000u;
				R = static_cast<int>((BGRAPixel & 0x00FF0000) >> 16);
				G = static_cast<int>((BGRAPixel & 0x0000FF00) >> 8);
				B = static_cast<int>(BGRAPixel & 0x000000FF);

				fLuma = pow((R_coeff[R] + G_coeff[G] + B_coeff[B]), fLumaExp);
				Luma = static_cast<unsigned int>(fLuma);

				const csSDK_uint32 BWPixel = alpha |
					Luma << 16 |
					Luma << 8 |
					Luma;

				*dstImg++ = BWPixel;
			}

		srcImg += linePitch - width;
		dstImg += linePitch - width;
	}

	return true;
}


bool processARGB4444_8u_slice(VideoHandle theData)
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

	const float* __restrict lpCoeff = (width > 800) ? coeff[BT709] : coeff[BT601];

	unsigned int alpha;
	unsigned int Luma;
	float R, G, B;

	for (int j = 0; j < height; j++)
	{
		__VECTOR_ALIGNED__
			for (int i = 0; i < width; i++)
			{
				const csSDK_uint32 BGRAPixel = *srcImg++;

				B = static_cast<float>((BGRAPixel & 0xFF000000u)>> 24);
				G = static_cast<float>((BGRAPixel & 0x00FF0000) >> 16);
				R = static_cast<float>((BGRAPixel & 0x0000FF00) >> 8);
				alpha = static_cast<int>(BGRAPixel & 0x000000FF);

				Luma = static_cast<unsigned int>(R * lpCoeff[0] + G * lpCoeff[1] + B * lpCoeff[2]);

				const csSDK_uint32 BWPixel = alpha |
					Luma << 24 |
					Luma << 16 |
					Luma << 8;

				*dstImg++ = BWPixel;
			}

		srcImg += linePitch - width;
		dstImg += linePitch - width;
	}

	return true;
}

bool processAdvancedARGB4444_8u_slice(VideoHandle theData)
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

	float fLuma;
	unsigned int Luma;
	unsigned int alpha;
	int R, G, B;

	for (int j = 0; j < height; j++)
	{
		__VECTOR_ALIGNED__
			for (int i = 0; i < width; i++)
			{
				const csSDK_uint32 BGRAPixel = *srcImg++;

				B = static_cast<int>((BGRAPixel & 0xFF000000u) >> 24);
				G = static_cast<int>((BGRAPixel & 0x00FF0000u) >> 16);
				R = static_cast<int>((BGRAPixel & 0x0000FF00u) >> 8);
				alpha = static_cast<int>(BGRAPixel & 0x000000FF);

				fLuma = pow((R_coeff[R] + G_coeff[G] + B_coeff[B]), fLumaExp);
				Luma = static_cast<unsigned int>(fLuma);

				const csSDK_uint32 BWPixel = alpha |
					Luma << 24 |
					Luma << 16 |
					Luma << 8;

				*dstImg++ = BWPixel;
			}

		srcImg += linePitch - width;
		dstImg += linePitch - width;
	}

	return true;
}

bool processBGRA4444_16u_slice(VideoHandle theData)
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

	const float* __restrict lpCoeff = (width > 800) ? coeff[BT709] : coeff[BT601];
	
	float R, G, B;
	unsigned int Luma;
	unsigned int A;

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

				Luma = static_cast<unsigned int>(R * lpCoeff[0] + G * lpCoeff[1] + B * lpCoeff[2]);

				*dstImg++ = (Luma << 16) | Luma;
				*dstImg++ = A | Luma;
			}

		srcImg += (linePitch - width * 2);
		dstImg += (linePitch - width * 2);
	}

	return true;
}


bool processAdvancedBGRA4444_16u_slice(VideoHandle theData)
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

	unsigned int R, G, B;
	unsigned int Luma;
	float fLuma;
	unsigned int A;

	for (int j = 0; j < height; j++)
	{
		__VECTOR_ALIGNED__
			for (int i = 0; i < width; i++)
			{
				const csSDK_uint32 first = *srcImg++;
				const csSDK_uint32 second = *srcImg++;

				B = static_cast<unsigned int> (first & 0x0000FFFFu);
				G = static_cast<unsigned int>((first & 0xFFFF0000u) >> 16);
				R = static_cast<unsigned int> (second & 0x0000FFFFu);
				A = second & 0xFFFF0000u;

				fLuma = pow((R_U16coeff[R] + G_U16coeff[G] + B_U16coeff[B]), fLumaExp);

				Luma = static_cast<unsigned int>(fLuma);

				*dstImg++ = (Luma << 16) | Luma;
				*dstImg++ = A | Luma;
			}

		srcImg += (linePitch - width * 2);
		dstImg += (linePitch - width * 2);
	}

	return true;
}


bool processBGRA4444_32f_slice(VideoHandle theData)
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
	const float* __restrict lpCoeff = (width > 800) ? coeff[BT709] : coeff[BT601];

	float R, G, B, Luma;

	for (int j = 0; j < height; j++)
	{
		__VECTOR_ALIGNED__
			for (int i = 0; i < width; i++)
			{
				B = *srcImg++;
				G = *srcImg++;
				R = *srcImg++;

				Luma = R * lpCoeff[0] + G * lpCoeff[1] + B * lpCoeff[2];

				*dstImg++ = Luma;
				*dstImg++ = Luma;
				*dstImg++ = Luma;

				// put alpha channel to destination
				*dstImg++ = *srcImg++;
			}

		srcImg += (linePitch - width * 4);
		dstImg += (linePitch - width * 4);

	}

	return true;
}


bool processARGB4444_32f_slice(VideoHandle theData)
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
	const float* __restrict lpCoeff = (width > 800) ? coeff[BT709] : coeff[BT601];

	float R, G, B, Luma;

	for (int j = 0; j < height; j++)
	{
		__VECTOR_ALIGNED__
			for (int i = 0; i < width; i++)
			{
				// put alpha channel to destination
				*dstImg++ = *srcImg++;

				B = *srcImg++;
				G = *srcImg++;
				R = *srcImg++;

				Luma = R * lpCoeff[0] + G * lpCoeff[1] + B * lpCoeff[2];

				*dstImg++ = Luma;
				*dstImg++ = Luma;
				*dstImg++ = Luma;
			}

		srcImg += (linePitch - width * 4);
		dstImg += (linePitch - width * 4);

	}

	return true;
}

bool processVUYA4444_32f_slice(VideoHandle theData)
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

	for (int j = 0; j < height; j++)
	{
		__VECTOR_ALIGNED__
			for (int i = 0; i < width; i++)
			{
				srcImg += 2; // skip first and second (V and U) elements from source		
				*dstImg++ = 0.0f; // put V to destination
				*dstImg++ = 0.0f; // put U to destination
				*dstImg++ = *srcImg++;   // put Y to destination
				*dstImg++ = *srcImg++;   // put ALPHA to destination
			}

		srcImg += (linePitch - width * 4);
		dstImg += (linePitch - width * 4);

	}

	return true;
}


bool processARGB4444_16u_slice(VideoHandle theData)
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

	const float* __restrict lpCoeff = (width > 800) ? coeff[BT709] : coeff[BT601];

	float R, G, B;
	unsigned int Luma;
	unsigned int A;

	for (int j = 0; j < height; j++)
	{
		__VECTOR_ALIGNED__
			for (int i = 0; i < width; i++)
			{
				const csSDK_uint32 first = *srcImg++;
				const csSDK_uint32 second = *srcImg++;

				A = static_cast<unsigned int>(first & 0x0000FFFFu);
				R = static_cast<float>((first & 0xFFFF0000u) >> 16);
				G = static_cast<float> (second & 0x0000FFFFu);
				B = static_cast<float>((second & 0xFFFF0000u) >> 16);

				Luma = static_cast<unsigned int>(R * lpCoeff[0] + G * lpCoeff[1] + B * lpCoeff[2]);

				*dstImg++ = (Luma << 16) | A;
				*dstImg++ = (Luma << 16) | Luma;
			}

		srcImg += (linePitch - width * 2);
		dstImg += (linePitch - width * 2);
	}

	return true;
}


bool processAdvancedARGB4444_16u_slice(VideoHandle theData)
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

	unsigned int R, G, B;
	unsigned int Luma;
	float fLuma;
	unsigned int A;

	for (int j = 0; j < height; j++)
	{
		__VECTOR_ALIGNED__
			for (int i = 0; i < width; i++)
			{
				const csSDK_uint32 first = *srcImg++;
				const csSDK_uint32 second = *srcImg++;

				A = static_cast<unsigned int> (first & 0x0000FFFFu);
				R = static_cast<unsigned int>((first & 0xFFFF0000u) >> 16);
				G = static_cast<unsigned int> (second & 0x0000FFFFu);
				B = static_cast<unsigned int>((second & 0xFFFF0000u) >> 16);

				fLuma = pow((R_U16coeff[R] + G_U16coeff[G] + B_U16coeff[B]), fLumaExp);

				Luma = static_cast<unsigned int>(fLuma);

				*dstImg++ = (Luma << 16) | A;
				*dstImg++ = (Luma << 16) | Luma;
			}

		srcImg += (linePitch - width * 2);
		dstImg += (linePitch - width * 2);
	}

	return true;
}


bool processRGB444_10u_slice(VideoHandle theData)
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

	const float* __restrict lpCoeff = coeff[BT709];

	unsigned int Luma;
	float R, G, B;

	for (int j = 0; j < height; j++)
	{
		__VECTOR_ALIGNED__
			for (int i = 0; i < width; i++)
			{
				const csSDK_uint32 BGRAPixel = *srcImg++;

				B = static_cast<float>((BGRAPixel & 0x00000FFC) >> 2);
				G = static_cast<float>((BGRAPixel & 0x003FF000) >> 12);
				R = static_cast<float>((BGRAPixel & 0xFFC00000) >> 22);

				Luma = 0x000003FFu & static_cast<unsigned int>(R * lpCoeff[0] + G * lpCoeff[1] + B * lpCoeff[2]);

				const csSDK_uint32 BWPixel = (Luma << 22) | (Luma << 12) | (Luma << 2);

				*dstImg++ = BWPixel;
			}

		srcImg += linePitch - width;
		dstImg += linePitch - width;
	}

	return true;
}


bool processAdvancedRGB444_10u_slice(VideoHandle theData)
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

	float fLuma;
	unsigned int Luma;
	unsigned int R, G, B;

	for (int j = 0; j < height; j++)
	{
		__VECTOR_ALIGNED__
			for (int i = 0; i < width; i++)
			{
				const csSDK_uint32 BGRAPixel = *srcImg++;

				R = static_cast<unsigned int>((BGRAPixel & 0x00000FFCu) >> 2);
				G = static_cast<unsigned int>((BGRAPixel & 0x003FF000u) >> 12);
				B = static_cast<unsigned int>((BGRAPixel & 0xFFC00000u) >> 22);

				fLuma = pow((R_U10coeff[R] + G_U10coeff[G] + B_U10coeff[B]), fLumaExp);
				Luma = static_cast<unsigned int>(fLuma);

				const csSDK_uint32 BWPixel = (Luma << 22) | (Luma << 12) | (Luma << 2);

				*dstImg++ = BWPixel;
			}

		srcImg += linePitch - width;
		dstImg += linePitch - width;
	}

	return true;
}



csSDK_int32 selectProcessFunction(VideoHandle theData, bool advancedAlg)
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
					processSucceed = (true == advancedAlg ? processAdvancedBGRA4444_8u_slice(theData) : processBGRA4444_8u_slice(theData));
				break;

				case PrPixelFormat_VUYA_4444_8u:
				case PrPixelFormat_VUYA_4444_8u_709:
					processSucceed = processVUYA_4444_8u_slice(theData);
				break;

				case PrPixelFormat_BGRA_4444_16u:
					processSucceed = (true == advancedAlg ? processAdvancedBGRA4444_16u_slice(theData) : processBGRA4444_16u_slice(theData));
				break;

				case PrPixelFormat_BGRA_4444_32f:
					processSucceed = processBGRA4444_32f_slice(theData);
				break;

				case PrPixelFormat_VUYA_4444_32f:
				case PrPixelFormat_VUYA_4444_32f_709:
					processSucceed = processVUYA4444_32f_slice(theData);
				break;

				// ============ native AE formats ============================= //
				case PrPixelFormat_ARGB_4444_8u:
					processSucceed = (true == advancedAlg ? processAdvancedARGB4444_8u_slice(theData) : processARGB4444_8u_slice(theData));
				break;

				case PrPixelFormat_ARGB_4444_16u:
					processSucceed = (true == advancedAlg ? processAdvancedARGB4444_16u_slice(theData) : processARGB4444_16u_slice(theData));
				break;

				case PrPixelFormat_ARGB_4444_32f:
					processSucceed = processARGB4444_32f_slice(theData);
				break;

				// =========== miscellanous formats =========================== //
				case PrPixelFormat_RGB_444_10u:
					processSucceed = (true == advancedAlg ? processAdvancedRGB444_10u_slice(theData) : processRGB444_10u_slice(theData));
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



// ImageLabHDR filter entry point
PREMPLUGENTRY DllExport xFilter(short selector, VideoHandle theData)
{
	filterParamsH	paramsH = nullptr;
	csSDK_int32		errCode = fsNoErr;

	switch (selector)
	{
		case fsInitSpec:
			if ((*theData)->specsHandle)
			{
				// In a filter that has a need for a more complex setup dialog
				// you would present your platform specific user interface here,
				// storing results in the specsHandle (which you've allocated).
			}
			else
			{
				paramsH = reinterpret_cast<filterParamsH>(((*theData)->piSuites->memFuncs->newHandle)(sizeof(filterParams)));

				// Memory allocation failed, no need to continue
				if (nullptr != paramsH)
				{
					(*paramsH)->checkbox = 0;
				}
				(*theData)->specsHandle = reinterpret_cast<char**>(paramsH);
			}
		break;

		case fsSetup:
		break;

		case fsExecute:
		{
			// Get the data from specsHandle
			paramsH = (filterParamsH)(*theData)->specsHandle;
			const bool advFlag = (nullptr != paramsH ? ((*paramsH)->checkbox ? true : false) : false);
			errCode = selectProcessFunction(theData, advFlag);
		}
		break;

		case fsDisposeData:
			if (nullptr != (*theData)->specsHandle)
			{
				(*theData)->piSuites->memFuncs->disposeHandle((*theData)->specsHandle);
				(*theData)->specsHandle = nullptr;
			}
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


