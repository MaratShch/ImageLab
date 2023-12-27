#ifndef __NOISE_CLEAN_ALGO_NON_LOCAL_BAYES_PARAMETERS__
#define __NOISE_CLEAN_ALGO_NON_LOCAL_BAYES_PARAMETERS__

#include <cstdint>
#include "CompileTimeUtils.hpp"
#include "CommonPixFormat.hpp"
#include "CommonAuxPixFormat.hpp"

struct AlgoBayesParams
{
	float fSigma;
	float fGamma;
	float fBeta;
	float fTau;
	uint32_t iSizePatch;
	uint32_t iSimilarPatches;
	uint32_t iSizeSearchWindow;
	uint32_t iBoundary;
	uint32_t iOffSet;
	uint32_t iIsFirstStep;
};


inline AlgoBayesParams AlgoBayesInitParametersSet1 (const float& fSigma) noexcept
{
	AlgoBayesParams algoParams;

	algoParams.fSigma = fSigma;
	algoParams.fGamma = 1.050f;
	algoParams.fBeta  = 1.0f;
	algoParams.fTau   = 0.0f;
	algoParams.iSizePatch = (fSigma < 20.0f ? 3u : (fSigma < 50.0f ? 5u : 7u));
	algoParams.iSimilarPatches = algoParams.iSizePatch * algoParams.iSizePatch * 3u;
	algoParams.iSizeSearchWindow = ODD_VALUE(HALF(algoParams.iSimilarPatches));
	algoParams.iBoundary = static_cast<uint32_t>(1.50f * static_cast<float>(algoParams.iSizeSearchWindow));
	algoParams.iOffSet = HALF(algoParams.iSizePatch);
	algoParams.iIsFirstStep = 1u;

	return algoParams;
}


inline AlgoBayesParams AlgoBayesInitParametersSet2 (const float& fSigma) noexcept
{
	AlgoBayesParams algoParams;

	algoParams.fSigma = fSigma;
	algoParams.fGamma = 1.050f;
	algoParams.fBeta  = fSigma < 50.0f ? 1.20f : 1.0f;
	algoParams.iSizePatch = (fSigma < 50.0f ? 3u : (fSigma < 70.0f ? 5u : 7u));
	algoParams.iSimilarPatches = algoParams.iSizePatch * algoParams.iSizePatch * 3u;
	algoParams.iSizeSearchWindow = ODD_VALUE(HALF(algoParams.iSimilarPatches));
	algoParams.iBoundary = static_cast<uint32_t>(1.50f * static_cast<float>(algoParams.iSizeSearchWindow));
	algoParams.fTau = 48.0f * static_cast<float>(algoParams.iSizePatch * algoParams.iSizePatch);
	algoParams.iOffSet = HALF(algoParams.iSizePatch);
	algoParams.iIsFirstStep = 0u;

	return algoParams;
}

inline void Convert2fYUV
(
	const PF_Pixel_VUYA_8u* __restrict pSrc,
	fYUV*                   __restrict pDst,
	const A_long&       sizeX,
	const A_long&       sizeY,
	const A_long&       srcPitch,
	const A_long&       dstPitch,
	const float&        fCoeff
) noexcept
{
	(void)fCoeff;
	for (A_long j = 0; j < sizeY; j++)
	{
		const PF_Pixel_VUYA_8u* __restrict pSrcLine = pSrc + j * srcPitch;
		      fYUV*             __restrict pDstLine = pDst + j * dstPitch;
		for (A_long i = 0; i < sizeX; i++)
		{
			pDstLine[i].Y = static_cast<float>(pSrcLine[i].Y);
			pDstLine[i].U = static_cast<float>(pSrcLine[i].U);
			pDstLine[i].V = static_cast<float>(pSrcLine[i].V);
		}
	}
}


template <typename T, std::enable_if_t<is_YUV_proc<T>::value>* = nullptr>
inline void Convert2fYUV
(
	const T* __restrict pSrc,
	fYUV*    __restrict pDst,
	const A_long&       sizeX,
	const A_long&       sizeY,
	const A_long&       srcPitch,
	const A_long&       dstPitch,
	const float&        fCoeff
) noexcept
{
	for (A_long j = 0; j < sizeY; j++)
	{
		const T* __restrict pSrcLine = pSrc + j * srcPitch;
		fYUV*    __restrict pDstLine = pDst + j * dstPitch;
		for (A_long i = 0; i < sizeX; i++)
		{
			pDstLine[i].Y = pSrcLine[i].Y * fCoeff;
			pDstLine[i].U = pSrcLine[i].U * fCoeff;
			pDstLine[i].V = pSrcLine[i].V * fCoeff;
		}
	}
}


#endif // __NOISE_CLEAN_ALGO_NON_LOCAL_BAYES_PARAMETERS__