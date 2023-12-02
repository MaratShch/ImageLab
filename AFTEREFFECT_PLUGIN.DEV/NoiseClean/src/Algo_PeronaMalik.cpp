#include "NoiseClean.hpp"
#include "PrSDKAESupport.h"
#include "NoiseCleanGFunction.hpp"
#include "FastAriphmetics.hpp"
#include "NoiseCleanAlgoMemory.hpp"


template <typename T, std::enable_if_t<is_YUV_proc<T>::value>* = nullptr>
inline PF_Err NoiseClean_AlgoAnisotropicDiffusion
(
	const T* __restrict pSrc,
	      T* __restrict pDst,
	const A_long&  sizeX,
	const A_long&  sizeY,
	const A_long&  srcPitch,
	const A_long&  dstPitch,
	const float&   noiseLevel,
	const float&   timeStep,
	const float&   maxVal
) noexcept
{
	A_long i, j;

	const A_long lastLine  = sizeY - 1;
	const A_long lastPixel = sizeX - 1;

	for (j = 0; j < sizeY; j++)
	{
		const A_long prevLine = FastCompute::Max(0,        j - 1);
		const A_long nextLine = FastCompute::Min(lastLine, j + 1);

		const T* __restrict pPrevLine = pSrc + srcPitch * prevLine;
		const T* __restrict pCurrLine = pSrc + srcPitch * j;
		const T* __restrict pNextLine = pSrc + srcPitch * nextLine;
		      T* __restrict pDstLine  = pDst + dstPitch * j;

		for (i = 0; i < sizeX; i++)
		{
			const A_long prevPixel = FastCompute::Max(0,         i - 1);
			const A_long nextPixel = FastCompute::Max(lastPixel, i + 1);

			const T& north   = pPrevLine[i];
			const T& south   = pNextLine[i];
			const T& west    = pCurrLine[prevPixel];
			const T& current = pCurrLine[i];
			const T& east    = pCurrLine[nextPixel];

			const float currentY  = static_cast<float>(current.Y);
			const float diffNorth = static_cast<float>(north.Y) - currentY;
			const float diffSouth = static_cast<float>(south.Y) - currentY;
			const float diffWest  = static_cast<float>(west.Y)  - currentY;
			const float diffEast  = static_cast<float>(east.Y)  - currentY;

			const float fSum = Gfunction (diffNorth, noiseLevel) * diffNorth +
				               Gfunction (diffSouth, noiseLevel) * diffSouth +
				               Gfunction (diffWest,  noiseLevel) * diffWest  +
				               Gfunction (diffEast,  noiseLevel) * diffEast;
			
			pDstLine[i].V = current.V;
			pDstLine[i].U = current.U;
			pDstLine[i].Y = CLAMP_VALUE(currentY + fSum * timeStep, 0.f, maxVal);
			pDstLine[i].A = current.A;
		}
	}

	return PF_Err_NONE;
}


template <typename T, std::enable_if_t<is_RGB_proc<T>::value>* = nullptr>
PF_Err NoiseClean_AlgoAnisotropicDiffusion
(
	const T* __restrict pSrc,
	      T* __restrict pDst,
	const A_long  sizeX,
	const A_long  sizeY,
	const A_long  srcPitch,
	const A_long  dstPitch,
	const float   noiseLevel,
	const float   timeStep,
	const float   maxVal
) noexcept
{
	return PF_Err_NONE;
}



template <typename T, std::enable_if_t<is_YUV_proc<T>::value>* = nullptr>
PF_Err NoiseClean_AlgoAnisotropicDiffusion
(
	const T* __restrict pSrc,
	T* __restrict pDst,
	const A_long  sizeX,
	const A_long  sizeY,
	const A_long  srcPitch,
	const A_long  dstPitch,
	const float   noiseLevel,
	const float   timeStep,
	const float   maxVal,
	const float   dispersion
) noexcept
{
	T* pTmp1 = nullptr;
	T* pTmp2 = nullptr;

	PF_Err err = PF_Err_OUT_OF_MEMORY;
	constexpr float minimalStep = 0.001f;

	A_long memBlockId = MemoryBufferAlloc (sizeX, sizeY, &pTmp1, &pTmp2);
	if (nullptr != pTmp1 && nullptr != pTmp2 && -1 != memBlockId)
	{
		float currentDispersion = 0.0f;
		float currentTimeStep = FastCompute::Min(timeStep, dispersion - currentDispersion);

		A_long ping = 0x0, pong = 0x1;
		A_long iterCnt = 0;

		do
		{
			if (0 == iterCnt)
			{

			}
			else if (currentDispersion + timeStep < dispersion)
			{
				ping ^= 0x1;
				pong ^= 0x1;
			}
			else
			{

			}

			iterCnt++;
			currentDispersion += currentTimeStep;
			currentTimeStep = FastCompute::Min (timeStep, dispersion - currentDispersion);

		} while (currentDispersion <= dispersion && currentTimeStep > minimalStep && err == PF_Err_NONE);

		MemoryBufferRelease(memBlockId);
		memBlockId = -1;
		pTmp1 = pTmp2 = nullptr;
		err = PF_Err_NONE;
	}

	return err;
}


PF_Err NoiseClean_AlgoPeronaMalik
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

	/* This plugin called frop PR - check video fomat */
	PrPixelFormat destinationPixelFormat = PrPixelFormat_Invalid;
	if (PF_Err_NONE == (AEFX_SuiteScoper<PF_PixelFormatSuite1>(in_data, kPFPixelFormatSuite, kPFPixelFormatSuiteVersion1, out_data)->GetPixelFormat(output, &destinationPixelFormat)))
	{
		const float fNoiseLevel = 0.3f;
		const float fTimeStep = 1.f;

		switch (destinationPixelFormat)
		{
			case PrPixelFormat_BGRA_4444_8u:
			{
				const PF_Pixel_BGRA_8u* __restrict localSrc = reinterpret_cast<const PF_Pixel_BGRA_8u* __restrict>(pfLayer->data);
					  PF_Pixel_BGRA_8u* __restrict localDst = reinterpret_cast<      PF_Pixel_BGRA_8u* __restrict>(output->data);
				sizeY = pfLayer->extent_hint.bottom - pfLayer->extent_hint.top;
				sizeX = pfLayer->extent_hint.right  - pfLayer->extent_hint.left;
				linePitch = pfLayer->rowbytes / static_cast<A_long>(PF_Pixel_BGRA_8u_size);

				err = NoiseClean_AlgoAnisotropicDiffusion (localSrc, localDst, sizeX, sizeY, linePitch, linePitch, fNoiseLevel, fTimeStep, static_cast<float>(u8_value_white));
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

				err = NoiseClean_AlgoAnisotropicDiffusion (localSrc, localDst, sizeX, sizeY, linePitch, linePitch, fNoiseLevel, fTimeStep, static_cast<float>(u8_value_white));
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

				err = NoiseClean_AlgoAnisotropicDiffusion(localSrc, localDst, sizeX, sizeY, linePitch, linePitch, fNoiseLevel, fTimeStep, f32_value_white);
			}
			break;

			case PrPixelFormat_BGRA_4444_16u:
			{
				const PF_Pixel_BGRA_16u* __restrict localSrc = reinterpret_cast<const PF_Pixel_BGRA_16u* __restrict>(pfLayer->data);
					  PF_Pixel_BGRA_16u* __restrict localDst = reinterpret_cast<      PF_Pixel_BGRA_16u* __restrict>(output->data);
				sizeY = pfLayer->extent_hint.bottom - pfLayer->extent_hint.top;
				sizeX = pfLayer->extent_hint.right   - pfLayer->extent_hint.left;
				linePitch = pfLayer->rowbytes / static_cast<A_long>(PF_Pixel_BGRA_16u_size);

				err = NoiseClean_AlgoAnisotropicDiffusion (localSrc, localDst, sizeX, sizeY, linePitch, linePitch, fNoiseLevel, fTimeStep, static_cast<float>(u16_value_white));
			}
			break;

			case PrPixelFormat_BGRA_4444_32f:
			{
				const PF_Pixel_BGRA_32f* __restrict localSrc = reinterpret_cast<const PF_Pixel_BGRA_32f* __restrict>(pfLayer->data);
					  PF_Pixel_BGRA_32f* __restrict localDst = reinterpret_cast<      PF_Pixel_BGRA_32f* __restrict>(output->data);
				sizeY = pfLayer->extent_hint.bottom - pfLayer->extent_hint.top;
				sizeX = pfLayer->extent_hint.right  - pfLayer->extent_hint.left;
				linePitch = pfLayer->rowbytes / static_cast<A_long>(PF_Pixel_BGRA_32f_size);

				err = NoiseClean_AlgoAnisotropicDiffusion (localSrc, localDst, sizeX, sizeY, linePitch, linePitch, fNoiseLevel, fTimeStep, f32_value_white);
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