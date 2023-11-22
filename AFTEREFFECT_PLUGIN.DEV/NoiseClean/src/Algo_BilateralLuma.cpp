#include "NoiseClean.hpp"
#include "FastAriphmetics.hpp"
#include "PrSDKAESupport.h"
#include "ImageLabMemInterface.hpp"

template <typename T, std::enable_if_t<is_YUV_proc<T>::value>* = nullptr>
PF_Err NoiseClean_AlgoBilateralLuma
(
	const T* __restrict pSrc,
	      T* __restrict pDst,
	 const A_long       sizeX,
	 const A_long       sizeY,
	 const A_long       srcPitch,
	 const A_long       dstPitch,
	 const A_long       windowSize,
	 const float        whiteValue = static_cast<float>(u8_value_white)
) noexcept
{
	CACHE_ALIGN float gMesh[cBilateralWindowMax][cBilateralWindowMax]{};
	CACHE_ALIGN float pH[cBilateralWindowMax * cBilateralWindowMax]{};
	CACHE_ALIGN float pF[cBilateralWindowMax * cBilateralWindowMax]{};
	
	constexpr float sigma = cBilateralSigma;
	constexpr float reciProcSigma = 1.f / (2.f * sigma * sigma);
	const float  reciprocWhite = 1.f / whiteValue;
	const A_long filterRadius = windowSize >> 1;
	const A_long lastLine  = sizeY - 1;
	const A_long lastPixel = sizeX - 1;
	A_long idx = 0;

	/* compute gaussian weights */
	gaussian_weights (filterRadius, gMesh);

	for (A_long j = 0; j < sizeY; j++)
	{
		const A_long jMin = FastCompute::Max(j - filterRadius, 0);
		const A_long jMax = FastCompute::Min(j + filterRadius, lastLine);
		const A_long jDif = jMax - jMin + 1;

		for (A_long i = 0; i < sizeX; i++)
		{
			const A_long iMin = FastCompute::Max(i - filterRadius, 0);
			const A_long iMax = FastCompute::Min(i + filterRadius, lastPixel);
			const A_long iDif = iMax - iMin + 1;

			/* reference (current) pixel */
			const T srcPixel = pSrc[j * srcPitch + i];
			const float yRef = static_cast<float>(srcPixel.Y) * reciprocWhite;
			A_long k, l;

			/* compute Gaussian intencity weights */
			__VECTORIZATION__
			for (idx = k = 0; k < jDif; k++)
			{
				const int pixPos = (jMin + k) * srcPitch + iMin;
				for (l = 0; l < iDif; l++)
				{
					const float dY = (static_cast<float>(pSrc[pixPos + i].Y) * reciprocWhite) - yRef;
					pH[idx] = FastCompute::Exp(-(dY * dY) * reciProcSigma);
					idx++;
				}
			}

			/* calculate Bilateral Filter responce */
			float fNorm = 0.f;
			float bSum = 0.f;
			A_long jIdx = jMin - j + filterRadius;

			for (idx = k = 0; k < jDif; k++)
			{
				A_long iIdx = iMin - i + filterRadius;
				__LOOP_UNROLL(3)
				for (l = 0; l < iDif; l++)
				{
					pF[idx] = pH[idx] * gMesh[jIdx][iIdx];
					fNorm += pF[idx];
					idx++, iIdx++;
				}
				jIdx++;
			} /* for (k = 0; k < jDif; k++) */

			for (idx = k = 0; k < jDif; k++)
			{
				const int kIdx = (jMin + k) * srcPitch + iMin;
				for (l = 0; l < iDif; l++)
				{
					const float fY = static_cast<float>(pSrc[kIdx + l].Y) * reciprocWhite;
					bSum += (pF[idx] * fY);
					idx++;
				}
			}

			T dstPixel;
			dstPixel.A = srcPixel.A;
			dstPixel.U = srcPixel.U;
			dstPixel.V = srcPixel.V;
			dstPixel.Y = (CLAMP_VALUE(whiteValue * bSum / fNorm, 0.f, whiteValue));
			pDst[j * dstPitch + i] = dstPixel;

		} /* for (A_long i = 0; i < sizeX; i++) */
	} /* for (j = 0; j < sizeY; j++) */

	return PF_Err_NONE;
}


template <typename T, std::enable_if_t<is_RGB_proc<T>::value>* = nullptr>
PF_Err NoiseClean_AlgoBilateralLuma
(
	const T* __restrict pSrc,
	      T* __restrict pDst,
	const A_long        sizeX,
	const A_long        sizeY,
	const A_long        srcPitch,
	const A_long        dstPitch,
	const A_long        windowSize,
	const float         whiteValue = static_cast<float>(u8_value_white)
) noexcept
{
	CACHE_ALIGN float gMesh[cBilateralWindowMax][cBilateralWindowMax]{};
	CACHE_ALIGN float pH[cBilateralWindowMax * cBilateralWindowMax]{};
	CACHE_ALIGN float pF[cBilateralWindowMax * cBilateralWindowMax]{};

	constexpr float sigma = cBilateralSigma;
	constexpr float sigmaDiv = 2.f * sigma * sigma;
	constexpr float reciProcSigma = 1.f / sigmaDiv;
	const float  reciprocWhite = 1.f / whiteValue;
	const A_long filterRadius = windowSize >> 1;
	const A_long lastLine  = sizeY - 1;
	const A_long lastPixel = sizeX - 1;
	A_long idx = 0;

	constexpr size_t doubleBuffer = 2;
	const size_t frameSize = sizeX * sizeY;
	const size_t requiredMemSize = frameSize * (sizeof(T) * doubleBuffer);
	const size_t memSizeForTmpBuffers = CreateAlignment (requiredMemSize, static_cast<size_t>(CACHE_LINE));

	void* pMemoryBlock = nullptr;
	A_long memBlockId = -1;

	memBlockId = ::GetMemoryBlock (memSizeForTmpBuffers, 0, &pMemoryBlock);
	if (-1 != memBlockId && nullptr != pMemoryBlock)
	{
		/* compute gaussian weights */
		gaussian_weights(filterRadius, gMesh);

		/* convert RGB buffer to YUV color space */

		/* release memory block */
		::FreeMemoryBlock (memBlockId);
	}

	return PF_Err_NONE;
}


PF_Err NoiseClean_AlgoBilateral
(
	PF_InData*   __restrict in_data,
	PF_OutData*  __restrict out_data,
	PF_ParamDef* __restrict params[],
	PF_LayerDef* __restrict output
) noexcept
{
	PF_Err err = PF_Err_NONE;
	A_long sizeX = 0, sizeY = 0, linePitch = 0;

	const PF_LayerDef* pfLayer = reinterpret_cast<const PF_LayerDef*>(&params[eNOISE_CLEAN_INPUT]->u.ld);
	/* get "Bilateral Window size" from slider */
	const int32_t bilateralWindowSize = ODD_VALUE(CLAMP_VALUE (static_cast<int32_t>(params[eNOISE_CLEAN_BILATERAL_WINDOW_SLIDER]->u.sd.value), cBilateralWindowMin, cBilateralWindowMax));

	/* This plugin called frop PR - check video fomat */
	PrPixelFormat destinationPixelFormat = PrPixelFormat_Invalid;
	if (PF_Err_NONE == (AEFX_SuiteScoper<PF_PixelFormatSuite1>(in_data, kPFPixelFormatSuite, kPFPixelFormatSuiteVersion1, out_data)->GetPixelFormat(output, &destinationPixelFormat)))
	{
		switch (destinationPixelFormat)
		{
			case PrPixelFormat_BGRA_4444_8u:
			{
				const PF_Pixel_BGRA_8u* __restrict localSrc = reinterpret_cast<const PF_Pixel_BGRA_8u* __restrict>(pfLayer->data);
				      PF_Pixel_BGRA_8u* __restrict localDst = reinterpret_cast<      PF_Pixel_BGRA_8u* __restrict>(output->data);
				sizeY = pfLayer->extent_hint.bottom - pfLayer->extent_hint.top;
				sizeX = pfLayer->extent_hint.right  - pfLayer->extent_hint.left;
				linePitch = pfLayer->rowbytes / static_cast<A_long>(PF_Pixel_BGRA_8u_size);

				err = NoiseClean_AlgoBilateralLuma (localSrc, localDst, sizeX, sizeY, linePitch, linePitch, bilateralWindowSize, static_cast<float>(u8_value_white));
			}
			break;

			case PrPixelFormat_VUYA_4444_8u_709:
			case PrPixelFormat_VUYA_4444_8u:
			{
				const PF_Pixel_VUYA_8u* __restrict localSrc = reinterpret_cast<const PF_Pixel_VUYA_8u* __restrict>(pfLayer->data);
				      PF_Pixel_VUYA_8u* __restrict localDst = reinterpret_cast<      PF_Pixel_VUYA_8u* __restrict>(output->data);
				sizeY = pfLayer->extent_hint.bottom - pfLayer->extent_hint.top;
				sizeX = pfLayer->extent_hint.right  - pfLayer->extent_hint.left;
				linePitch = pfLayer->rowbytes / static_cast<A_long>(PF_Pixel_VUYA_8u_size);

				err = NoiseClean_AlgoBilateralLuma (localSrc, localDst, sizeX, sizeY, linePitch, linePitch, bilateralWindowSize, static_cast<float>(u8_value_white));
			}
			break;

			case PrPixelFormat_VUYA_4444_32f_709:
			case PrPixelFormat_VUYA_4444_32f:
			{
				const PF_Pixel_VUYA_32f* __restrict localSrc = reinterpret_cast<const PF_Pixel_VUYA_32f* __restrict>(pfLayer->data);
				      PF_Pixel_VUYA_32f* __restrict localDst = reinterpret_cast<      PF_Pixel_VUYA_32f* __restrict>(output->data);
				sizeY = pfLayer->extent_hint.bottom - pfLayer->extent_hint.top;
				sizeX = pfLayer->extent_hint.right  - pfLayer->extent_hint.left;
				linePitch = pfLayer->rowbytes / static_cast<A_long>(PF_Pixel_VUYA_32f_size);

				err = NoiseClean_AlgoBilateralLuma (localSrc, localDst, sizeX, sizeY, linePitch, linePitch, bilateralWindowSize, f32_value_white);
			}
			break;

			case PrPixelFormat_BGRA_4444_16u:
			{
				const PF_Pixel_BGRA_16u* __restrict localSrc = reinterpret_cast<const PF_Pixel_BGRA_16u* __restrict>(pfLayer->data);
				      PF_Pixel_BGRA_16u* __restrict localDst = reinterpret_cast<      PF_Pixel_BGRA_16u* __restrict>(output->data);
				sizeY = pfLayer->extent_hint.bottom - pfLayer->extent_hint.top;
				sizeX = pfLayer->extent_hint.right  - pfLayer->extent_hint.left;
				linePitch = pfLayer->rowbytes / static_cast<A_long>(PF_Pixel_BGRA_16u_size);

				err = NoiseClean_AlgoBilateralLuma (localSrc, localDst, sizeX, sizeY, linePitch, linePitch, bilateralWindowSize, static_cast<float>(u16_value_white));
			}
			break;
	
			case PrPixelFormat_BGRA_4444_32f:
			{
				const PF_Pixel_BGRA_32f* __restrict localSrc = reinterpret_cast<const PF_Pixel_BGRA_32f* __restrict>(pfLayer->data);
				      PF_Pixel_BGRA_32f* __restrict localDst = reinterpret_cast<      PF_Pixel_BGRA_32f* __restrict>(output->data);
				sizeY = pfLayer->extent_hint.bottom - pfLayer->extent_hint.top;
				sizeX = pfLayer->extent_hint.right  - pfLayer->extent_hint.left;
				linePitch = pfLayer->rowbytes / static_cast<A_long>(PF_Pixel_BGRA_32f_size);

				err = NoiseClean_AlgoBilateralLuma (localSrc, localDst, sizeX, sizeY, linePitch, linePitch, bilateralWindowSize, f32_value_white);
			}
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


PF_Err NoiseClean_AlgoBilateralAe8
(
	PF_InData*   __restrict in_data,
	PF_OutData*  __restrict out_data,
	PF_ParamDef* __restrict params[],
	PF_LayerDef* __restrict output
) noexcept
{
	const PF_EffectWorld*   __restrict input = reinterpret_cast<const PF_EffectWorld* __restrict>(&params[eNOISE_CLEAN_INPUT]->u.ld);
	const PF_Pixel_ARGB_8u* __restrict localSrc = reinterpret_cast<const PF_Pixel_ARGB_8u* __restrict>(input->data);
	      PF_Pixel_ARGB_8u* __restrict localDst = reinterpret_cast<      PF_Pixel_ARGB_8u* __restrict>(output->data);

	/* get "Bilateral Window size" from slider */
	const int32_t bilateralWindowSize = ODD_VALUE(CLAMP_VALUE(static_cast<int32_t>(params[eNOISE_CLEAN_BILATERAL_WINDOW_SLIDER]->u.sd.value), cBilateralWindowMin, cBilateralWindowMax));

	auto const src_pitch = input->rowbytes  / static_cast<A_long>(PF_Pixel_ARGB_8u_size);
	auto const dst_pitch = output->rowbytes / static_cast<A_long>(PF_Pixel_ARGB_8u_size);
	auto const sizeY  = output->height;
	auto const sizeX  = output->width;

	return NoiseClean_AlgoBilateralLuma (localSrc, localDst, sizeX, sizeY, src_pitch, dst_pitch, bilateralWindowSize, static_cast<float>(u8_value_white));
}


PF_Err NoiseClean_AlgoBilateralAe16
(
	PF_InData*   __restrict in_data,
	PF_OutData*  __restrict out_data,
	PF_ParamDef* __restrict params[],
	PF_LayerDef* __restrict output
) noexcept
{
	const PF_EffectWorld*   __restrict input = reinterpret_cast<const PF_EffectWorld* __restrict>(&params[eNOISE_CLEAN_INPUT]->u.ld);
	const PF_Pixel_ARGB_16u* __restrict localSrc = reinterpret_cast<const PF_Pixel_ARGB_16u* __restrict>(input->data);
	      PF_Pixel_ARGB_16u* __restrict localDst = reinterpret_cast<      PF_Pixel_ARGB_16u* __restrict>(output->data);

	  /* get "Bilateral Window size" from slider */
	const int32_t bilateralWindowSize = ODD_VALUE(CLAMP_VALUE(static_cast<int32_t>(params[eNOISE_CLEAN_BILATERAL_WINDOW_SLIDER]->u.sd.value), cBilateralWindowMin, cBilateralWindowMax));

	auto const src_pitch = input->rowbytes  / static_cast<A_long>(PF_Pixel_ARGB_16u_size);
	auto const dst_pitch = output->rowbytes / static_cast<A_long>(PF_Pixel_ARGB_16u_size);
	auto const sizeY = output->height;
	auto const sizeX = output->width;

	return NoiseClean_AlgoBilateralLuma (localSrc, localDst, sizeX, sizeY, src_pitch, dst_pitch, bilateralWindowSize, static_cast<float>(u16_value_white));
}