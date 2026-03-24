#include "FastAriphmetics.hpp"
#include "ImageMosaicUtils.hpp"
#ifdef _DEBUG
#include <cassert>
#endif


void ArtMosaic::fillProcBuf (Color* pBuf, const A_long pixNumber, const float val) noexcept
{
	constexpr A_long elemInStruct = sizeof(pBuf[0]) / sizeof(pBuf[0].r);
	const A_long rawSize    = pixNumber * elemInStruct;
	const A_long rawSize24  = rawSize / 24;
	const A_long rawFract24 = rawSize % 24;

	float* pBufF = reinterpret_cast<float*>(pBuf);

	// Use set1_ps for cleaner broadcasting
	const __m256 fPattern = _mm256_set1_ps(val); 
	
	for (A_long i = 0; i < rawSize24; i++)
	{
		_mm256_storeu_ps (pBufF, fPattern), pBufF += 8;
		_mm256_storeu_ps (pBufF, fPattern), pBufF += 8;
		_mm256_storeu_ps (pBufF, fPattern), pBufF += 8;
	}

	for (A_long i = 0; i < rawFract24; i++)
		pBufF[i] = val;

	return;
}

void ArtMosaic::fillProcBuf (std::unique_ptr<Color[]>& pBuf, const A_long pixNumber, const float val) noexcept
{
	ArtMosaic::fillProcBuf (pBuf.get(), pixNumber, val);
}

void ArtMosaic::fillProcBuf (A_long* pBuf, const A_long pixNumber, const A_long val) noexcept
{
	const A_long rawSize16  = pixNumber / 16;
	const A_long rawFract16 = pixNumber % 16;
	
	// Use set1_epi32
	const __m256i iPattern = _mm256_set1_epi32(val); 
	__m256i* pBufAvxPtr = reinterpret_cast<__m256i*>(pBuf);

	for (A_long i = 0; i < rawSize16; i++)
	{
		// FIX: Changed to storeu (unaligned)
		_mm256_storeu_si256 (pBufAvxPtr, iPattern), pBufAvxPtr++;
		_mm256_storeu_si256 (pBufAvxPtr, iPattern), pBufAvxPtr++;
	}

	if (0 != rawFract16)
	{
		A_long* pBufPtr = reinterpret_cast<A_long*>(pBufAvxPtr);
		for (A_long i = 0; i < rawFract16; i++)
			pBufPtr[i] = val;
	}

	return;
}

void ArtMosaic::fillProcBuf(std::unique_ptr<A_long[]>& pBuf, const A_long pixNumber, const A_long val) noexcept
{
	ArtMosaic::fillProcBuf (pBuf.get(), pixNumber, val);
}


void ArtMosaic::fillProcBuf (float* pBuf, const A_long pixNumber, const float val) noexcept
{
	const A_long rawSize16 = pixNumber / 16;
	const A_long rawFract16 = pixNumber % 16;
	
	// Use set1_ps
	const __m256 fPattern = _mm256_set1_ps(val); 
	
	// Better to cast to float* for the pointer arithmetic and storing
	float* pBufAvxPtr = pBuf;

	for (A_long i = 0; i < rawSize16; i++)
	{
		// FIX: Changed to storeu_ps (unaligned)
		_mm256_storeu_ps (pBufAvxPtr, fPattern), pBufAvxPtr += 8;
		_mm256_storeu_ps (pBufAvxPtr, fPattern), pBufAvxPtr += 8;
	}

	if (0 != rawFract16)
	{
		for (A_long i = 0; i < rawFract16; i++)
			pBufAvxPtr[i] = val;
	}

	return;
}


void ArtMosaic::fillProcBuf(std::unique_ptr<float[]>& pBuf, const A_long pixNumber, const float val) noexcept
{
	ArtMosaic::fillProcBuf(pBuf.get(), pixNumber, val);
}


