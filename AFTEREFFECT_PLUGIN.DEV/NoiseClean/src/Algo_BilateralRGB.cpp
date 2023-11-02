#include "NoiseClean.hpp"
#include "PrSDKAESupport.h"
#include "CommonAuxPixFormat.hpp"
#include "CompileTimeUtils.hpp"
#include "ColorTransform.hpp"

inline A_long MemoryBufferAlloc
(
	const A_long&  sizeX,
	const A_long&  sizeY,
	fCIELabPix**   pBuf1,
	fCIELabPix**   pBuf2
) noexcept
{
	void* pAlgoMemory = nullptr;
	const A_long frameSize = sizeX * sizeY;
	constexpr A_long doubleBuffer = 2;
	const size_t requiredMemSize = ::CreateAlignment (frameSize * fCIELabPix_size * doubleBuffer, static_cast<size_t>(CACHE_LINE));
	const A_long blockId = ::GetMemoryBlock (static_cast<int32_t>(requiredMemSize), 0, &pAlgoMemory);

	if (nullptr != pAlgoMemory)
	{
#ifdef _DEBUG
		memset (pAlgoMemory, 0, requiredMemSize);
#endif
		*pBuf1 = static_cast<fCIELabPix*>(pAlgoMemory);
		*pBuf2 = *pBuf1 + frameSize;
	}
	else
		*pBuf1 = *pBuf2 = nullptr;

	return blockId;
}


inline void MemoryBufferRelease
(
	A_long blockId
) noexcept
{
	::FreeMemoryBlock (blockId);
	return;
}


template <typename T, std::enable_if_t<is_RGB_proc<T>::value>* = nullptr>
inline void ConvertToCIELab
(
	const T*    __restrict pSrc,
	fCIELabPix* __restrict pDst,
	const A_long  sizeX,
	const A_long  sizeY,
	const A_long  srcPitch,
	const float   fNorm
) noexcept
{
	constexpr float fReferences[3] = {
		cCOLOR_ILLUMINANT[observer_CIE_1931][color_ILLUMINANT_D65][0],
		cCOLOR_ILLUMINANT[observer_CIE_1931][color_ILLUMINANT_D65][1],
		cCOLOR_ILLUMINANT[observer_CIE_1931][color_ILLUMINANT_D65][2],
	};

	for (A_long j = 0; j < sizeY; j++)
	{
		const T* __restrict pSrcLine = pSrc + j * srcPitch;
		fCIELabPix* pDstLine = pDst + j * sizeX;

		for (A_long i = 0; i < sizeX; i++)
		{
			fRGB srcRgb;
			srcRgb.R = pSrcLine[i].R * fNorm;
			srcRgb.G = pSrcLine[i].G * fNorm;
			srcRgb.B = pSrcLine[i].B * fNorm;

			pDstLine[i] = RGB2CIELab (srcRgb, fReferences);
		}
	}

	return;
}

template <typename T, std::enable_if_t<is_YUV_proc<T>::value>* = nullptr>
inline void ConvertToCIELab
(
	const T*    __restrict pSrc,
	fCIELabPix* __restrict pDst,
	const A_long  sizeX,
	const A_long  sizeY,
	const A_long  srcPitch,
	const float   fNorm
) noexcept
{
	return;
}


template <typename T>
inline void ConvertCIELabToRGB
(
	const T*    __restrict pSrc,
	fCIELabPix* __restrict pTmp,
	      T*    __restrict pDst,
	const A_long  sizeX,
	const A_long  sizeY,
	const A_long  srcPitch,
	const A_long  dstPitch,
	const float   fNorm
) noexcept
{
	constexpr float fReferences[3] = {
		cCOLOR_ILLUMINANT[observer_CIE_1931][color_ILLUMINANT_D65][0],
		cCOLOR_ILLUMINANT[observer_CIE_1931][color_ILLUMINANT_D65][1],
		cCOLOR_ILLUMINANT[observer_CIE_1931][color_ILLUMINANT_D65][2],
	};

	for (A_long j = 0; j < sizeY; j++)
	{
		const T* __restrict pSrcLine    = pSrc + j * srcPitch;
		      T* __restrict pDstLine    = pDst + j * dstPitch;
		fCIELabPix* __restrict pTmpLine = pTmp + j * sizeX;

		for (A_long i = 0; i < sizeX; i++)
		{
			const fCIELabPix& pixelLab = pTmpLine[i];
			const fRGB rgb = CIELab2RGB (pixelLab, fReferences);

			pDstLine[i].B = rgb.B * fNorm;
			pDstLine[i].G = rgb.G * fNorm;
			pDstLine[i].R = rgb.R * fNorm;
			pDstLine[i].A = pSrcLine[i].A;
		}
	}

	return;
}


inline void BilateralFilter
(
	const fCIELabPix* __restrict pIn,
	      fCIELabPix* __restrict pOut,
	const A_long  sizeX,
	const A_long  sizeY,
	const A_long  windowSize
)
{
	return;
}


template <typename T, std::enable_if_t<is_YUV_proc<T>::value>* = nullptr>
PF_Err NoiseClean_AlgoBilateralColor
(
	const T* __restrict pSrc,
	      T* __restrict pDst,
	const A_long  sizeX,
	const A_long  sizeY,
	const A_long  srcPitch,
	const A_long  dstPitch,
	const A_long  windowSize,
	const float   fDiv
) noexcept
{
	PF_Err err = PF_Err_OUT_OF_MEMORY;

	/* allocate temporary memory storage */
	fCIELabPix* pTmpBuffer1 = nullptr;
	fCIELabPix* pTmpBuffer2 = nullptr;

	A_long blockId = MemoryBufferAlloc(sizeX, sizeY, &pTmpBuffer1, &pTmpBuffer2);

	if (nullptr != pTmpBuffer1 && nullptr != pTmpBuffer2 && -1 != blockId)
	{
		/* convert RGB to CIEL*a*b  */

		/* perform bilateral filter denoising */

		/* convert back from CIEL*a*b to RGB */

		/* release memory storage */
		MemoryBufferRelease (blockId);
		blockId = -1;
		pTmpBuffer1 = pTmpBuffer2 = nullptr;

		err = PF_Err_NONE;
	}

	return err;
}


template <typename T, std::enable_if_t<is_RGB_proc<T>::value>* = nullptr>
PF_Err NoiseClean_AlgoBilateralColor
(
	const T* __restrict pSrc,
	T* __restrict pDst,
	const A_long  sizeX,
	const A_long  sizeY,
	const A_long  srcPitch,
	const A_long  dstPitch,
	const A_long  windowSize,
	const float   fDiv
) noexcept
{
	fCIELabPix* pTmpBuffer1 = nullptr;
	fCIELabPix* pTmpBuffer2 = nullptr;
	PF_Err err = PF_Err_OUT_OF_MEMORY;

	const float fNorm = 1.0f / fDiv;

	/* allocate temporary memory storage */
	A_long blockId = MemoryBufferAlloc(sizeX, sizeY, &pTmpBuffer1, &pTmpBuffer2);

	if (nullptr != pTmpBuffer1 && nullptr != pTmpBuffer2 && -1 != blockId)
	{
		/* convert RGB to CIEL*a*b  */
		ConvertToCIELab (pSrc, pTmpBuffer1, sizeX, sizeY, srcPitch, fNorm);

		/* perform bilateral filter denoising */
		BilateralFilter (pTmpBuffer1, pTmpBuffer2, sizeX, sizeY, windowSize);

		/* convert back from CIEL*a*b to RGB */
		ConvertCIELabToRGB (pSrc, pTmpBuffer1, pDst, sizeX, sizeY, srcPitch, dstPitch, fDiv);

		/* release memory storage */
		MemoryBufferRelease (blockId);
		blockId = -1;
		pTmpBuffer1 = pTmpBuffer2 = nullptr;

		err = PF_Err_NONE;
	}

	return err;
}



PF_Err NoiseClean_AlgoBilateralRGB
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
	const int32_t bilateralWindowSize = 1 | CLAMP_VALUE(static_cast<int32_t>(params[eNOISE_CLEAN_BILATERAL_WINDOW_SLIDER]->u.sd.value), cBilateralWindowMin, cBilateralWindowMax);

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

				err = NoiseClean_AlgoBilateralColor (localSrc, localDst, sizeX, sizeY, linePitch, linePitch, bilateralWindowSize, static_cast<float>(u8_value_white));
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

				err = NoiseClean_AlgoBilateralColor (localSrc, localDst, sizeX, sizeY, linePitch, linePitch, bilateralWindowSize, static_cast<float>(u8_value_white));
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

				err = NoiseClean_AlgoBilateralColor (localSrc, localDst, sizeX, sizeY, linePitch, linePitch, bilateralWindowSize, 1.f);
			}
			break;

			case PrPixelFormat_BGRA_4444_16u:
			{
				const PF_Pixel_BGRA_16u* __restrict localSrc = reinterpret_cast<const PF_Pixel_BGRA_16u* __restrict>(pfLayer->data);
				      PF_Pixel_BGRA_16u* __restrict localDst = reinterpret_cast<      PF_Pixel_BGRA_16u* __restrict>(output->data);
				sizeY = pfLayer->extent_hint.bottom - pfLayer->extent_hint.top;
				sizeX = pfLayer->extent_hint.right  - pfLayer->extent_hint.left;
				linePitch = pfLayer->rowbytes / static_cast<A_long>(PF_Pixel_BGRA_16u_size);

				err = NoiseClean_AlgoBilateralColor (localSrc, localDst, sizeX, sizeY, linePitch, linePitch, bilateralWindowSize, static_cast<float>(u16_value_white));
			}
			break;

			case PrPixelFormat_BGRA_4444_32f:
			{
				const PF_Pixel_BGRA_32f* __restrict localSrc = reinterpret_cast<const PF_Pixel_BGRA_32f* __restrict>(pfLayer->data);
				      PF_Pixel_BGRA_32f* __restrict localDst = reinterpret_cast<      PF_Pixel_BGRA_32f* __restrict>(output->data);
				sizeY = pfLayer->extent_hint.bottom - pfLayer->extent_hint.top;
				sizeX = pfLayer->extent_hint.right  - pfLayer->extent_hint.left;
				linePitch = pfLayer->rowbytes / static_cast<A_long>(PF_Pixel_BGRA_32f_size);

				err = NoiseClean_AlgoBilateralColor (localSrc, localDst, sizeX, sizeY, linePitch, linePitch, bilateralWindowSize, 1.f);
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