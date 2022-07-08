#include "ImageStylization.hpp"
#include "StylizationStructs.hpp"
#include "PrSDKAESupport.h"
#include "ColorTransformMatrix.hpp"
#include "FastAriphmetics.hpp"
#include "ImageAuxPixFormat.hpp"
#include "ImageMosaicUtils.hpp"


void ArtMosaic::fillProcBuf (ProcPixel* pBuf, const A_long& pixNumber, const float& val) noexcept
{
	constexpr A_long elemInStruct = sizeof(pBuf[0]) / sizeof(pBuf[0].R);
	const A_long rawSize    = pixNumber * elemInStruct;
	const A_long rawSize24  = rawSize / 24;
	const A_long rawFract24 = rawSize % 24;

	float* pBufF = reinterpret_cast<float*>(pBuf);

	const __m256 fPattern = _mm256_set_ps(val, val, val, val, val, val, val, val);
	for (A_long i = 0; i < rawSize24; i++)
	{
		_mm256_storeu_ps(pBufF, fPattern), pBufF += 8;
		_mm256_storeu_ps(pBufF, fPattern), pBufF += 8;
		_mm256_storeu_ps(pBufF, fPattern), pBufF += 8;
	}

	for (A_long i = 0; i < rawFract24; i++)
		pBufF[i] = val;

	return;
}

void ArtMosaic::fillProcBuf(std::unique_ptr<ProcPixel[]>& pBuf, const A_long& pixNumber, const float& val) noexcept
{
	ArtMosaic::fillProcBuf (pBuf.get(), pixNumber, val);
}


