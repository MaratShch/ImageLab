#include "NoiseClean.hpp"
#include "PrSDKAESupport.h"
#include "NoiseCleanNonLocalBayesAlgo.hpp"
#include "NoiseCleanAlgoMemory.hpp"
#include "FastAriphmetics.hpp"

template <typename T, std::enable_if_t<is_YUV_proc<T>::value>* = nullptr>
PF_Err NonLocalBayes_YUV_Processing
(
	const T* __restrict    pSrc,
	      T* __restrict    pDst,
	A_long                 sizeX,
	A_long                 sizeY,
	A_long                 srcPitch,
	A_long                 dstPitch,
	const AlgoBayesParams& algoParams1,
	const AlgoBayesParams& algoParams2,
	const float            maxColorVal
) noexcept
{
	fYUV* pTmp[2]{};
	const float fCoeff1 = 255.f / maxColorVal;
	const float fCoeff2 = maxColorVal / 255.f;

	/* compute ImageBoundary for left/right side and for top/botom size */
	const auto imgMaxBoundary = 2u * FastCompute::Max (algoParams1.iBoundary, algoParams2.iBoundary);
	const auto boundarySizeX = sizeX + imgMaxBoundary;
	const auto boundarySizeY = sizeY + imgMaxBoundary;

	/* allocate temporary buffers for processing */
	A_long memBlockId = MemoryBufferAlloc (boundarySizeX, boundarySizeY, &pTmp[0], &pTmp[1]);
	if (nullptr != pTmp[0] && nullptr != pTmp[1] && -1 != memBlockId)
	{
		/* convert to fYUV with value range from 0 to 255 */
		YUV2fYUV (pSrc, pTmp[0], sizeX, sizeY, srcPitch, boundarySizeX, fCoeff1);

		/* first ALGO Step */
		NonLocalBayes_fYUV_Processing (pTmp[0], pTmp[1], sizeX, sizeY, algoParams1, true);

		/* second ALGO step */
		NonLocalBayes_fYUV_Processing (pTmp[1], pTmp[0], sizeX, sizeY, algoParams2, false);

		/* back convert to YUV after denoising, original source (pSrc) used only for save ALPHA channel values to destination */
		fYUV2YUV (pSrc, pTmp[0], pDst, sizeX, sizeY, srcPitch, boundarySizeX, dstPitch, fCoeff2);

		/* release temporary memory buffer after processing complete */
		MemoryBufferRelease (memBlockId);
		memBlockId = -1;
		pTmp[0] = pTmp[1] = nullptr;
	}

	return PF_Err_NONE;
}


PF_Err NoiseClean_AlgoNonLocalBayes
(
	PF_InData*   __restrict in_data,
	PF_OutData*  __restrict out_data,
	PF_ParamDef* __restrict params[],
	PF_LayerDef* __restrict output
) noexcept
{
	PF_Err err = PF_Err_NONE;
	PF_Err errFormat = PF_Err_INVALID_INDEX;
	PrPixelFormat destinationPixelFormat = PrPixelFormat_Invalid;

	constexpr float reciproc10 = 1.f / 10.f;
	const float fSigma = static_cast<const float>(params[eNOISE_CLEAN_NL_BAYES_SIGMA]->u.sd.value) * reciproc10;

	AlgoBayesParams algoParamSet1 = AlgoBayesInitParametersSet1 (fSigma);
	AlgoBayesParams algoParamSet2 = AlgoBayesInitParametersSet2 (fSigma);

	/* This plugin called frop PR - check video fomat */
	if (PF_Err_NONE == (AEFX_SuiteScoper<PF_PixelFormatSuite1>(in_data, kPFPixelFormatSuite, kPFPixelFormatSuiteVersion1, out_data)->GetPixelFormat(output, &destinationPixelFormat)))
	{
		const PF_LayerDef* pfLayer = reinterpret_cast<const PF_LayerDef*>(&params[eNOISE_CLEAN_INPUT]->u.ld);

		switch (destinationPixelFormat)
		{
			case PrPixelFormat_BGRA_4444_8u:
			break;

			case PrPixelFormat_VUYA_4444_8u_709:
			case PrPixelFormat_VUYA_4444_8u:
			{
				const PF_Pixel_VUYA_8u* __restrict localSrc = reinterpret_cast<const PF_Pixel_VUYA_8u* __restrict>(pfLayer->data);
				      PF_Pixel_VUYA_8u* __restrict localDst = reinterpret_cast<      PF_Pixel_VUYA_8u* __restrict>(output->data);
				const A_long sizeX = pfLayer->extent_hint.right  - pfLayer->extent_hint.left;
			    const A_long sizeY = pfLayer->extent_hint.bottom - pfLayer->extent_hint.top;
				const A_long linePitch = pfLayer->rowbytes / static_cast<A_long>(PF_Pixel_BGRA_8u_size);
				constexpr float fMaxColorValue = 255.0f;

				err = NonLocalBayes_YUV_Processing (localSrc, localDst, sizeX, sizeY, linePitch, linePitch, algoParamSet1, algoParamSet2, fMaxColorValue);
			}
			break;

			case PrPixelFormat_VUYA_4444_32f_709:
			case PrPixelFormat_VUYA_4444_32f:
			{
				const PF_Pixel_VUYA_32f* __restrict localSrc = reinterpret_cast<const PF_Pixel_VUYA_32f* __restrict>(pfLayer->data);
				      PF_Pixel_VUYA_32f* __restrict localDst = reinterpret_cast<      PF_Pixel_VUYA_32f* __restrict>(output->data);
				const A_long sizeX = pfLayer->extent_hint.right  - pfLayer->extent_hint.left;
				const A_long sizeY = pfLayer->extent_hint.bottom - pfLayer->extent_hint.top;
				const A_long linePitch = pfLayer->rowbytes / static_cast<A_long>(PF_Pixel_BGRA_8u_size);
				constexpr float fMaxColorValue = 1.0f;

				err = NonLocalBayes_YUV_Processing(localSrc, localDst, sizeX, sizeY, linePitch, linePitch, algoParamSet1, algoParamSet2, fMaxColorValue);
			}
			break;

			case PrPixelFormat_BGRA_4444_16u:
			break;

			case PrPixelFormat_BGRA_4444_32f:
			break;

			default:
				err = PF_Err_INVALID_INDEX;
			break;
		}
	} /* if (PF_Err_NONE == (errFormat = pixelFormatSuite->GetPixelFormat(output, &destinationPixelFormat))) */
	else
	{
		err = PF_Err_UNRECOGNIZED_PARAM_TYPE;
	}

	return err;
}



PF_Err NoiseClean_AlgoNonLocalBayesAe8
(
	PF_InData*   __restrict in_data,
	PF_OutData*  __restrict out_data,
	PF_ParamDef* __restrict params[],
	PF_LayerDef* __restrict output
) noexcept
{
	const PF_EffectWorld*   __restrict input    = reinterpret_cast<const PF_EffectWorld*   __restrict>(&params[eNOISE_CLEAN_INPUT]->u.ld);
	const PF_Pixel_ARGB_8u* __restrict localSrc = reinterpret_cast<const PF_Pixel_ARGB_8u* __restrict>(input->data);
	      PF_Pixel_ARGB_8u* __restrict localDst = reinterpret_cast<      PF_Pixel_ARGB_8u* __restrict>(output->data);

	auto const src_pitch = input->rowbytes  / static_cast<A_long>(PF_Pixel_ARGB_8u_size);
	auto const dst_pitch = output->rowbytes / static_cast<A_long>(PF_Pixel_ARGB_8u_size);
	auto const sizeY = output->height;
	auto const sizeX = output->width;

	constexpr float reciproc10 = 1.f / 10.f;
	const float fSigma = static_cast<const float>(params[eNOISE_CLEAN_NL_BAYES_SIGMA]->u.sd.value) * reciproc10;

	AlgoBayesParams algoParamSet1 = AlgoBayesInitParametersSet1 (fSigma);
	AlgoBayesParams algoParamSet2 = AlgoBayesInitParametersSet2 (fSigma);


	return  PF_Err_NONE;
}


PF_Err NoiseClean_AlgoNonLocalBayesAe16
(
	PF_InData*   __restrict in_data,
	PF_OutData*  __restrict out_data,
	PF_ParamDef* __restrict params[],
	PF_LayerDef* __restrict output
) noexcept
{
	const PF_EffectWorld*    __restrict input    = reinterpret_cast<const PF_EffectWorld*    __restrict>(&params[eNOISE_CLEAN_INPUT]->u.ld);
	const PF_Pixel_ARGB_16u* __restrict localSrc = reinterpret_cast<const PF_Pixel_ARGB_16u* __restrict>(input->data);
	      PF_Pixel_ARGB_16u* __restrict localDst = reinterpret_cast<      PF_Pixel_ARGB_16u* __restrict>(output->data);

	auto const src_pitch = input->rowbytes  / static_cast<A_long>(PF_Pixel_ARGB_16u_size);
	auto const dst_pitch = output->rowbytes / static_cast<A_long>(PF_Pixel_ARGB_16u_size);
	auto const sizeY = output->height;
	auto const sizeX = output->width;

	constexpr float reciproc10 = 1.f / 10.f;
	const float fSigma = static_cast<const float>(params[eNOISE_CLEAN_NL_BAYES_SIGMA]->u.sd.value) * reciproc10;

	AlgoBayesParams algoParamSet1 = AlgoBayesInitParametersSet1 (fSigma);
	AlgoBayesParams algoParamSet2 = AlgoBayesInitParametersSet2 (fSigma);


	return PF_Err_NONE;
}