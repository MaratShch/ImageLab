#pragma once

#include "PrSDKEffect.h"
#include "PrSDKPixelFormat.h"
#include "PrSDKPPixSuite.h"
#include "PrSDKPixelFormatSuite.h"
#include "PrSDKSequenceInfoSuite.h"
#include "SDK_File.h"

#define CACHE_LINE  64
#define CACHE_ALIGN __declspec(align(CACHE_LINE))

#define AVX2_ALIGN __declspec(align(32))
#define AVX512_ALIGN __declspec(align(64))

#if defined __INTEL_COMPILER 
#define __VECTOR_ALIGNED__ __pragma(vector aligned)
#else
#define __VECTOR_ALIGNED__
#endif

static constexpr int CIELabBufferbands = 3;
static constexpr size_t CIELabBufferPixSize = 2048 * 1080; /* components order: L, a, b */
static constexpr size_t CIELabBufferSize = CIELabBufferbands * CIELabBufferPixSize * sizeof(float); /* components order: L, a, b */
static constexpr size_t CIELabBufferAlign = CACHE_LINE;

static constexpr int maxRadiusSize = 10;
static constexpr float Exp = 2.7182818f;


template<typename T>
T MIN(T a, T b) { return ((a < b) ? a : b); }

template<typename T>
T MAX(T a, T b) { return ((a > b) ? a : b); }

template<typename T>
T EXP(T val) {
	return powf(Exp, val); // powf for floating
}


// Declare plug-in entry point with C linkage
#ifdef __cplusplus
extern "C" {
#endif

PREMPLUGENTRY DllExport xFilter(short selector, VideoHandle theData);
BOOL APIENTRY DllMain(HMODULE /* hModule */, DWORD ul_reason_for_call, LPVOID /* lpReserved */);

#ifdef __cplusplus
}
#endif

csSDK_int32 imageLabPixelFormatSupported(const VideoHandle theData);

void BGRA_convert_to_CIELab(
	const csSDK_uint32* __restrict pBGRA,   /* format B, G, R, A (each band as unsigned char) */
	float*		        __restrict pCEILab, /* format: L, a, b (each band as double) */
	const int                      sizeX,
	const int                      sizeY,
	const int                      rowBytes);

void CIELab_convert_to_BGRA(const float*       __restrict pCIELab,
	const unsigned int* __restrict pSrcBGRA, /* original image required only for take data from alpha channel */
	unsigned int*		__restrict pDstBGRA,
	const int                      sizeX,
	const int                      sizeY,
	const int                      rowBytes);


void CreateColorConvertTable(void);
void DeleteColorConvertTable(void);

void* allocCIELabBuffer(const size_t& size);
void freeCIELabBuffer(void* pMem);

void gaussian_weights(const float sigma = 3.0f, const int radius = 5 /* radius size in range of 5 to 10 */);
void bilateral_filter_color(const float* __restrict pCIELab, float* __restrict pFiltered, const int sizeX, const int sizeY, const int radius, const float sigmaR);

static csSDK_int32 processFrame(VideoHandle theData);

inline float acbrt (float x0)
{
	union { int ix; float x; };
	int sign = 0x0;

	if (x0 < 0)
	{
		x0 = abs(x0);
		sign = 0x80000000;
	}
	x = x0;
	ix = (ix >> 2) + (ix >> 4);           // Approximate divide by 3.
	ix = ix + (ix >> 4);
	ix = ix + (ix >> 8);
	ix = 0x2a5137a0 + ix;        // Initial guess.
	x = 0.33333333f * (2.0f * x + x0 / (x * x));  // Newton step.
	x = 0.33333333f * (2.0f * x + x0 / (x * x));  // Newton step again.
	ix |= sign;
	return x;
}

inline float aExpFast(const float& fVal) {
	float x = 1.0f + fVal / 256.0f;
	x *= x; x *= x; x *= x; x *= x;
	x *= x; x *= x; x *= x; x *= x;
	return x;
}


inline float aExp(const float & fVal)
{
	float y = 1.0f + fVal / 1024.0f;
	y *= y; y *= y; y *= y; y *= y;
	y *= y; y *= y; y *= y; y *= y;
	y *= y; y *= y;
	return y;
}


inline uint32_t __float_as_int(const float& x0) {
	union { int ix; float x; };
	x = x0;
	return ix;
}


inline float __int_as_float(const int32_t& x0) {
	union { int ix; float x; };
	ix = x0;
	return x;
}

// natural log on [0x1.f7a5ecp-127, 0x1.fffffep127]. Maximum relative error 9.4529e-5 
inline float aLog(const float& a)
{
	float m, r, s, t, i, f;
	int32_t e;

	e = (__float_as_int(a) - 0x3f2aaaab) & 0xff800000;
	m = __int_as_float(__float_as_int(a) - e);
	i = (float)e * 1.19209290e-7f; // 0x1.0p-23
								   // m in [2/3, 4/3] 
	f = m - 1.0f;
	s = f * f;
	// Compute log1p(f) for f in [-1/3, 1/3] 
	r = fmaf(0.230836749f, f, -0.279208571f); // 0x1.d8c0f0p-3, -0x1.1de8dap-2
	t = fmaf(0.331826031f, f, -0.498910338f); // 0x1.53ca34p-2, -0x1.fee25ap-2
	r = fmaf(r, s, t);
	r = fmaf(r, s, f);
	r = fmaf(i, 0.693147182f, r); // 0x1.62e430p-1 // log(2) 
	return r;
}

