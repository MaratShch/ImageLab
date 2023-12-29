#include <vector>
#include "NoiseCleanNonLocalBayesAlgo.hpp"
#include "NoiseCleanAlgoMemory.hpp"


void NonLocalBayes_fYUV_Processing
(
	const fYUV* __restrict pSrc,
	      fYUV* __restrict pDst,
	A_long                 sizeX,
	A_long                 sizeY,
	const AlgoBayesParams& algoParams,
	bool                   isFirstStep
) noexcept
{
	const auto sW = algoParams.iSizeSearchWindow;
	const auto sP = algoParams.iSizePatch;
	const auto sP2 = sP  * sP;
	const auto sPC = sP2 * 3;
	const auto nSP = algoParams.iSimilarPatches;
	      auto nInverseFailed = 0;
	const auto fSigma = algoParams.fSigma;
	const auto fGamma = algoParams.fGamma;
	const auto threshold = fSigma * fSigma * fGamma * (isFirstStep ? 3.f : 1.f);

	/* define matrices using during Bayes' estimation */
	const size_t vecSize_group3d = nSP * sP2;
	std::vector<std::vector<float>> group3d(3, std::vector<float>(vecSize_group3d));

	const size_t vecSize_group3dNoisy = sW * sW * sPC;
	std::vector<float>  group3dNoisy(vecSize_group3dNoisy);

	const size_t vecSize_group3dBasic = vecSize_group3dNoisy;
	std::vector<float>  group3dBasic(vecSize_group3dBasic);

	const size_t vecSize_index = (isFirstStep ? nSP : sW * sW);
	std::vector<A_long> index(vecSize_index);

	const size_t vecSize_group3dTranspose = (isFirstStep ? nSP * sP2 : sW * sW * sPC);
	std::vector<float>  group3dTranspose(vecSize_group3dTranspose);

	const size_t vecSize_tmpMat = (isFirstStep ? sP2 * sP2 : sPC * sPC);
	std::vector<float>  tmpMat(vecSize_tmpMat);

	const size_t vecSize_BariCenter = (isFirstStep ? sP2 : sPC);
	std::vector<float>  BariCenter(vecSize_BariCenter);

	const size_t vecSize_CovMat = (isFirstStep ? sP2 * sP2 : sPC * sPC);
	std::vector<float>  CovMat(vecSize_CovMat);

	const size_t vecSize_CovMatTmp = vecSize_CovMat;
	std::vector<float>  CovMatTmp (vecSize_CovMatTmp);

	const size_t imgSize = sizeX * sizeY;
	std::vector<float> weight (imgSize, 0.f);
	std::vector<bool>  mask (imgSize, false);


	return;
}

