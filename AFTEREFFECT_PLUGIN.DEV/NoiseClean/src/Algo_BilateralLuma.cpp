#include "NoiseCleanAlgoMemory.hpp"
#include "FastAriphmetics.hpp"
#include "PrSDKAESupport.h"
#include "ColorTransform.hpp"

template <typename T, std::enable_if_t<is_RGB_proc<T>::value>* = nullptr>
void imgRGB2YUV
(
	const T* __restrict srcImage,
	PF_Pixel_VUYA_32f* __restrict dstImage,
	eCOLOR_SPACE transformSpace,
	A_long sizeX,
	A_long sizeY,
	A_long src_line_pitch,
	A_long dst_line_pitch,
	float   div = 1.f
) noexcept
{
	const float* const __restrict colorMatrix = RGB2YUV[transformSpace];
	const float reciproc = 1.f / div;
	for (A_long j = 0; j < sizeY; j++)
	{
		const T*           __restrict pSrcLine = srcImage + j * src_line_pitch;
		PF_Pixel_VUYA_32f* __restrict pDstLine = dstImage + j * dst_line_pitch;

		__VECTOR_ALIGNED__
		for (A_long i = 0; i < sizeX; i++)
		{
			const float R = static_cast<float>(pSrcLine[i].R) * reciproc;
			const float G = static_cast<float>(pSrcLine[i].G) * reciproc;
			const float B = static_cast<float>(pSrcLine[i].B) * reciproc;

			pDstLine[i].A = pSrcLine[i].A;
			pDstLine[i].Y = R * colorMatrix[0] + G * colorMatrix[1] + B * colorMatrix[2];
			pDstLine[i].U = R * colorMatrix[3] + G * colorMatrix[4] + B * colorMatrix[5];
			pDstLine[i].V = R * colorMatrix[6] + G * colorMatrix[7] + B * colorMatrix[8];
		}
	}
	return;
}

template <typename T, std::enable_if_t<is_RGB_proc<T>::value>* = nullptr>
void imgYUV2RGB
(
	const PF_Pixel_VUYA_32f* __restrict srcImage,
	T* __restrict dstImage,
	eCOLOR_SPACE transformSpace,
	A_long sizeX,
	A_long sizeY,
	A_long src_line_pitch,
	A_long dst_line_pitch,
	float  mult = 1.f
) noexcept
{
	const float* __restrict colorMatrix = YUV2RGB[transformSpace];
	for (A_long j = 0; j < sizeY; j++)
	{
		const PF_Pixel_VUYA_32f* __restrict pSrcLine = srcImage + j * src_line_pitch;
		T*                       __restrict pDstLine = dstImage + j * dst_line_pitch;

		__VECTOR_ALIGNED__
		for (A_long i = 0; i < sizeX; i++)
		{
			pDstLine[i].A = pSrcLine[i].A;
			pDstLine[i].R = CLAMP_VALUE (mult * (pSrcLine[i].Y * colorMatrix[0] + pSrcLine[i].U * colorMatrix[1] + pSrcLine[i].V * colorMatrix[2]), 0.f, mult);
			pDstLine[i].G = CLAMP_VALUE (mult * (pSrcLine[i].Y * colorMatrix[3] + pSrcLine[i].U * colorMatrix[4] + pSrcLine[i].V * colorMatrix[5]), 0.f, mult);
			pDstLine[i].B = CLAMP_VALUE (mult * (pSrcLine[i].Y * colorMatrix[6] + pSrcLine[i].U * colorMatrix[7] + pSrcLine[i].V * colorMatrix[8]), 0.f, mult);
		}
	}
	return;
}


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
	 const float        whiteValue
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


PF_Err NoiseClean_AlgoBilateralLuma
(
	const PF_Pixel_BGRA_8u* __restrict pSrc,
	      PF_Pixel_BGRA_8u* __restrict pDst,
	const A_long        sizeX,
	const A_long        sizeY,
	const A_long        srcPitch,
	const A_long        dstPitch,
	const A_long        windowSize,
	const float         whiteValue
) noexcept
{
	PF_Pixel_VUYA_32f* pTmp1 = nullptr; /* Because u8 pixel defined as unsigned char we can't normally convert to YUV with possible U or V negative values */
	PF_Pixel_VUYA_32f* pTmp2 = nullptr; /* Let's convert this buffer from unsigned char RGB to float YUV                                                   */

	A_long memBlockId = MemoryBufferAlloc (sizeX, sizeY, &pTmp1, &pTmp2);

	if (nullptr != pTmp1 && nullptr != pTmp2 && -1 != memBlockId)
	{
		constexpr int32_t convert_addendum{ 0 };
		constexpr eCOLOR_SPACE colorSpace { BT709 };

		/* convert RGB image to YUV */
		imgRGB2YUV (pSrc, pTmp1, colorSpace, sizeX, sizeY, srcPitch, sizeX, whiteValue);
		/* call templated function for VUYA variant */
		NoiseClean_AlgoBilateralLuma (pTmp1, pTmp2, sizeX, sizeY, sizeX, sizeX, windowSize, whiteValue);
		/* convert processed YUV back to RGB */
		imgYUV2RGB (pTmp2, pDst, colorSpace, sizeX, sizeY, sizeX, dstPitch, whiteValue);

		/* Release temporary memory */
		MemoryBufferRelease(memBlockId);
		memBlockId = -1;
		pTmp1 = pTmp2 = nullptr;
	}

	return PF_Err_NONE;
}


PF_Err NoiseClean_AlgoBilateralLuma
(
	const PF_Pixel_ARGB_8u* __restrict pSrc,
	      PF_Pixel_ARGB_8u* __restrict pDst,
	const A_long        sizeX,
	const A_long        sizeY,
	const A_long        srcPitch,
	const A_long        dstPitch,
	const A_long        windowSize,
	const float         whiteValue
) noexcept
{
	PF_Pixel_VUYA_32f* pTmp1 = nullptr; /* Because u8 pixel defined as unsigned char we can't normally convert to YUV with possible U or V negative values */
	PF_Pixel_VUYA_32f* pTmp2 = nullptr; /* Let's convert this buffer from unsigned char RGB to float YUV                                                   */

	A_long memBlockId = MemoryBufferAlloc (sizeX, sizeY, &pTmp1, &pTmp2);

	if (nullptr != pTmp1 && nullptr != pTmp2 && -1 != memBlockId)
	{
		constexpr eCOLOR_SPACE colorSpace { BT709 };

		/* convert RGB image to YUV */
		imgRGB2YUV (pSrc, pTmp1, colorSpace, sizeX, sizeY, srcPitch, sizeX, whiteValue);
		/* call templated function for VUYA variant */
		NoiseClean_AlgoBilateralLuma (pTmp1, pTmp2, sizeX, sizeY, sizeX, sizeX, windowSize, whiteValue);
		/* convert processed YUV back to RGB */
		imgYUV2RGB (pTmp2, pDst, colorSpace, sizeX, sizeY, sizeX, dstPitch, whiteValue);

		/* Release temporary memory */
		MemoryBufferRelease (memBlockId);
		memBlockId = -1;
		pTmp1 = pTmp2 = nullptr;
	}

	return PF_Err_NONE;
}


PF_Err NoiseClean_AlgoBilateralLuma
(
	const PF_Pixel_BGRA_16u* __restrict pSrc,
	      PF_Pixel_BGRA_16u* __restrict pDst,
	const A_long        sizeX,
	const A_long        sizeY,
	const A_long        srcPitch,
	const A_long        dstPitch,
	const A_long        windowSize,
	const float         whiteValue
) noexcept
{
	PF_Pixel_VUYA_16u* pTmp1 = nullptr;
	PF_Pixel_VUYA_16u* pTmp2 = nullptr;

	A_long memBlockId = MemoryBufferAlloc (sizeX, sizeY, &pTmp1, &pTmp2);

	if (nullptr != pTmp1 && nullptr != pTmp2 && -1 != memBlockId)
	{
		constexpr int32_t convert_addendum{ 0 };
		constexpr eCOLOR_SPACE colorSpace{ BT709 };

		/* convert RGB image to YUV */
		imgRGB2YUV (pSrc, pTmp1, colorSpace, sizeX, sizeY, srcPitch, sizeX);
		/* call templated function for VUYA variant */
		NoiseClean_AlgoBilateralLuma (pTmp1, pTmp2, sizeX, sizeY, sizeX, sizeX, windowSize, whiteValue);
		/* convert processed YUV back to RGB */
		imgYUV2RGB (pTmp2, pDst, colorSpace, sizeX, sizeY, sizeX, dstPitch);

		/* Release temporary memory */
		MemoryBufferRelease (memBlockId);
		memBlockId = -1;
		pTmp1 = pTmp2 = nullptr;
	}

	return PF_Err_NONE;
}


PF_Err NoiseClean_AlgoBilateralLuma
(
	const PF_Pixel_ARGB_16u* __restrict pSrc,
	      PF_Pixel_ARGB_16u* __restrict pDst,
	const A_long        sizeX,
	const A_long        sizeY,
	const A_long        srcPitch,
	const A_long        dstPitch,
	const A_long        windowSize,
	const float         whiteValue
) noexcept
{
	PF_Pixel_VUYA_16u* pTmp1 = nullptr;
	PF_Pixel_VUYA_16u* pTmp2 = nullptr;

	A_long memBlockId = MemoryBufferAlloc (sizeX, sizeY, &pTmp1, &pTmp2);

	if (nullptr != pTmp1 && nullptr != pTmp2 && -1 != memBlockId)
	{
		constexpr int32_t convert_addendum{ 0 };
		constexpr eCOLOR_SPACE colorSpace{ BT709 };

		/* convert RGB image to YUV */
		imgRGB2YUV (pSrc, pTmp1, colorSpace, sizeX, sizeY, srcPitch, sizeX);
		/* call templated function for VUYA variant */
		NoiseClean_AlgoBilateralLuma (pTmp1, pTmp2, sizeX, sizeY, sizeX, sizeX, windowSize, whiteValue);
		/* convert processed YUV back to RGB */
		imgYUV2RGB (pTmp2, pDst, colorSpace, sizeX, sizeY, sizeX, dstPitch);

		/* Release temporary memory */
		MemoryBufferRelease (memBlockId);
		memBlockId = -1;
		pTmp1 = pTmp2 = nullptr;
	}

	return PF_Err_NONE;
}


PF_Err NoiseClean_AlgoBilateralLuma
(
	const PF_Pixel_BGRA_32f* __restrict pSrc,
	      PF_Pixel_BGRA_32f* __restrict pDst,
	const A_long        sizeX,
	const A_long        sizeY,
	const A_long        srcPitch,
	const A_long        dstPitch,
	const A_long        windowSize,
	const float         whiteValue
) noexcept
{
	PF_Pixel_VUYA_32f* pTmp1 = nullptr;
	PF_Pixel_VUYA_32f* pTmp2 = nullptr;

	A_long memBlockId = MemoryBufferAlloc (sizeX, sizeY, &pTmp1, &pTmp2);

	if (nullptr != pTmp1 && nullptr != pTmp2 && -1 != memBlockId)
	{
		constexpr int32_t convert_addendum{ 0 };
		constexpr eCOLOR_SPACE colorSpace { BT709 };

		/* convert RGB image to YUV */
		imgRGB2YUV (pSrc, pTmp1, colorSpace, sizeX, sizeY, srcPitch, sizeX);
		/* call templated function for VUYA variant */
		NoiseClean_AlgoBilateralLuma (pTmp1, pTmp2, sizeX, sizeY, sizeX, sizeX, windowSize, whiteValue);
		/* convert processed YUV back to RGB */
		imgYUV2RGB (pTmp2, pDst, colorSpace, sizeX, sizeY, sizeX, dstPitch);

		/* Release temporary memory */
		MemoryBufferRelease (memBlockId);
		memBlockId = -1;
		pTmp1 = pTmp2 = nullptr;
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
	const A_long bilateralWindowSize = CLAMP_VALUE(ODD_VALUE(static_cast<A_long>(params[eNOISE_CLEAN_BILATERAL_WINDOW_SLIDER]->u.sd.value)), cBilateralWindowMin, cBilateralWindowMax);

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
	const A_long bilateralWindowSize = CLAMP_VALUE(ODD_VALUE(static_cast<A_long>(params[eNOISE_CLEAN_BILATERAL_WINDOW_SLIDER]->u.sd.value)), cBilateralWindowMin, cBilateralWindowMax);

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
    const A_long bilateralWindowSize = CLAMP_VALUE(ODD_VALUE(static_cast<A_long>(params[eNOISE_CLEAN_BILATERAL_WINDOW_SLIDER]->u.sd.value)), cBilateralWindowMin, cBilateralWindowMax);

	auto const src_pitch = input->rowbytes  / static_cast<A_long>(PF_Pixel_ARGB_16u_size);
	auto const dst_pitch = output->rowbytes / static_cast<A_long>(PF_Pixel_ARGB_16u_size);
	auto const sizeY = output->height;
	auto const sizeX = output->width;

	return NoiseClean_AlgoBilateralLuma (localSrc, localDst, sizeX, sizeY, src_pitch, dst_pitch, bilateralWindowSize, static_cast<float>(u16_value_white));
}