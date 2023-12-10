#include "NoiseClean.hpp"
#include "PrSDKAESupport.h"
#include "NoiseCleanGFunction.hpp"
#include "FastAriphmetics.hpp"
#include "NoiseCleanAlgoMemory.hpp"
#include "ColorTransformMatrix.hpp"

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
	const float   maxVal
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
			const A_long nextPixel = FastCompute::Min(lastPixel, i + 1);

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
			
			const float finalY = CLAMP_VALUE(currentY + fSum * timeStep, 0.f, maxVal);
			pDstLine[i].V = current.V;
			pDstLine[i].U = current.U;
			pDstLine[i].Y = finalY;
			pDstLine[i].A = current.A;
		}
	}

	return PF_Err_NONE;
}


PF_Err NoiseClean_AlgoAnisotropicDiffusion
(
	const PF_Pixel_VUYA_32f* __restrict pSrc,
	      PF_Pixel_VUYA_32f* __restrict pDst,
	const A_long  sizeX,
	const A_long  sizeY,
	const A_long  srcPitch,
	const A_long  dstPitch,
	const float   noiseLevel,
	const float   timeStep,
	const float   maxVal
) noexcept
{
	A_long i, j;

	const A_long lastLine = sizeY - 1;
	const A_long lastPixel = sizeX - 1;

	constexpr float fScale = 255.f;
	constexpr float fNorm = 1.f / fScale;

	for (j = 0; j < sizeY; j++)
	{
		const A_long prevLine = FastCompute::Max(0, j - 1);
		const A_long nextLine = FastCompute::Min(lastLine, j + 1);

		const PF_Pixel_VUYA_32f* __restrict pPrevLine = pSrc + srcPitch * prevLine;
		const PF_Pixel_VUYA_32f* __restrict pCurrLine = pSrc + srcPitch * j;
		const PF_Pixel_VUYA_32f* __restrict pNextLine = pSrc + srcPitch * nextLine;
		      PF_Pixel_VUYA_32f* __restrict pDstLine  = pDst + dstPitch * j;

		for (i = 0; i < sizeX; i++)
		{
			const A_long prevPixel = FastCompute::Max(0, i - 1);
			const A_long nextPixel = FastCompute::Min(lastPixel, i + 1);

			const PF_Pixel_VUYA_32f& north   = pPrevLine[i];
			const PF_Pixel_VUYA_32f& south   = pNextLine[i];
			const PF_Pixel_VUYA_32f& west    = pCurrLine[prevPixel];
			const PF_Pixel_VUYA_32f& current = pCurrLine[i];
			const PF_Pixel_VUYA_32f& east    = pCurrLine[nextPixel];

			const float currentY  = fScale * current.Y;
			const float diffNorth = fScale * north.Y - currentY;
			const float diffSouth = fScale * south.Y - currentY;
			const float diffWest  = fScale * west.Y  - currentY;
			const float diffEast  = fScale * east.Y  - currentY;

			const float fSum = Gfunction(diffNorth, noiseLevel) * diffNorth +
				               Gfunction(diffSouth, noiseLevel) * diffSouth +
				               Gfunction(diffWest, noiseLevel)  * diffWest +
				               Gfunction(diffEast, noiseLevel)  * diffEast;

			const float finalY = CLAMP_VALUE(fNorm * (currentY + fSum * timeStep), 0.f, maxVal);
			pDstLine[i].V = fNorm * current.V;
			pDstLine[i].U = fNorm * current.U;
			pDstLine[i].Y = finalY;
			pDstLine[i].A = current.A;
		}
	}

	return PF_Err_NONE;
}

template <typename T, std::enable_if_t<is_YUV_proc<T>::value>* = nullptr>
PF_Err NoiseClean_AlgoAnisotropicDiffusion
(
	const T* __restrict pSrc,
	      T* __restrict pDst,
	A_long  sizeX,
	A_long  sizeY,
	A_long  srcPitch,
	A_long  dstPitch,
	float   noiseLevel,
	float   timeStep,
	float   maxVal,
	float   dispersion
) noexcept
{
	T* pTmp[2] {};
	constexpr float minimalStep = 0.001f;
	PF_Err err = PF_Err_OUT_OF_MEMORY;

	A_long memBlockId = MemoryBufferAlloc (sizeX, sizeY, &pTmp[0], &pTmp[1]);
	if (nullptr != pTmp[0] && nullptr != pTmp[1] && -1 != memBlockId)
	{
		T* srcBuffer = nullptr;
		T* dstBuffer = nullptr;

		float currentDispersion = 0.0f;
		float currentTimeStep = FastCompute::Min (timeStep, dispersion - currentDispersion);

		A_long ping = 0x0, pong = 0x1;
		A_long iterCnt = 0;
		A_long pitchSrc = 0, pitchDst = 0;

		do
		{
			if (0 == iterCnt)
			{
				srcBuffer = const_cast<T*>(pSrc);
				dstBuffer = pTmp[ping];
				pitchSrc  = srcPitch;
				pitchDst  = sizeX;
			}
			else if (currentDispersion + timeStep < dispersion)
			{
				srcBuffer = pTmp[ping];
				dstBuffer = pTmp[pong];
				pitchSrc  = pitchDst = sizeX;
				ping ^= 0x1;
				pong ^= 0x1;
			}
			else
			{
				srcBuffer = pTmp[ping];
				dstBuffer = pDst;
				pitchSrc  = sizeX;
				pitchDst  = dstPitch;
			}

			err = NoiseClean_AlgoAnisotropicDiffusion (srcBuffer, dstBuffer, sizeX, sizeY, pitchSrc, pitchDst, noiseLevel, timeStep, maxVal);

			iterCnt++;
			currentDispersion += currentTimeStep;
			currentTimeStep = FastCompute::Min (timeStep, dispersion - currentDispersion);

		} while (currentDispersion <= dispersion && currentTimeStep > minimalStep && err == PF_Err_NONE);

		MemoryBufferRelease(memBlockId);
		memBlockId = -1;
		pTmp[0] = pTmp[1] = nullptr;
		err = PF_Err_NONE;
	}

	return err;
}


/*
	convert from RGB domain from all supported pixel bit depth to YUVA_32f domain
*/
template <typename T, std::enable_if_t<is_RGB_proc<T>::value>* = nullptr>
inline void imgRGB2YUVConvert
(
	const T*           __restrict pSrc,
	PF_Pixel_VUYA_32f* __restrict pDst,
	A_long  sizeX,
	A_long  sizeY,
	A_long  srcPitch,
	A_long  dstPitch,
	float   maxVal,
	eCOLOR_SPACE transformSpace
) noexcept
{
	const float* __restrict colorMatrix = RGB2YUV[transformSpace];
	const float reciprocVal = 1.0f / maxVal;

	for (A_long j = 0; j < sizeY; j++)
	{
		const T*           __restrict pSrcLine = pSrc + j * srcPitch;
		PF_Pixel_VUYA_32f* __restrict pDstLine = pDst + j * dstPitch;

		for (A_long i = 0; i < sizeX; i++)
		{
			const float B = static_cast<float>(pSrcLine[i].B) * reciprocVal;
			const float G = static_cast<float>(pSrcLine[i].G) * reciprocVal;
			const float R = static_cast<float>(pSrcLine[i].R) * reciprocVal;

			pDstLine[i].A = static_cast<float>(pSrcLine[i].A);
			pDstLine[i].Y = R * colorMatrix[0] + G * colorMatrix[1] + B * colorMatrix[2];
			pDstLine[i].U = R * colorMatrix[3] + G * colorMatrix[4] + B * colorMatrix[5];
			pDstLine[i].V = R * colorMatrix[6] + G * colorMatrix[7] + B * colorMatrix[8];
		} /* for (A_long i = 0; i < sizeX; i++) */
	} /* for (A_long j = 0; j < sizeY; j++) */

	return;
}


/* 
	convert from YUVA_32f domain to RGB domain with all supported pixels bit depth
*/
template <typename T, std::enable_if_t<is_RGB_proc<T>::value>* = nullptr>
inline void imgYUV2RGBConvert
(
	const PF_Pixel_VUYA_32f* __restrict pSrc,
	T*                       __restrict pDst,
	A_long  sizeX,
	A_long  sizeY,
	A_long  srcPitch,
	A_long  dstPitch,
	float   maxVal,
	eCOLOR_SPACE transformSpace
) noexcept
{
	const float* __restrict colorMatrix = YUV2RGB[transformSpace];
	for (A_long j = 0; j < sizeY; j++)
	{
		const PF_Pixel_VUYA_32f* __restrict pSrcLine = pSrc + j * srcPitch;
		T*                       __restrict pDstLine = pDst + j * dstPitch;

		for (A_long i = 0; i < sizeX; i++)
		{
			const float R = pSrcLine[i].Y * colorMatrix[0] + pSrcLine[i].U * colorMatrix[1] + pSrcLine[i].V * colorMatrix[2];
			const float G = pSrcLine[i].Y * colorMatrix[3] + pSrcLine[i].U * colorMatrix[4] + pSrcLine[i].V * colorMatrix[5];
			const float B = pSrcLine[i].Y * colorMatrix[6] + pSrcLine[i].U * colorMatrix[7] + pSrcLine[i].V * colorMatrix[8];

			pDstLine[i].B = CLAMP_VALUE(B * maxVal, 0.f, maxVal);
			pDstLine[i].G = CLAMP_VALUE(G * maxVal, 0.f, maxVal);
			pDstLine[i].R = CLAMP_VALUE(R * maxVal, 0.f, maxVal);
			pDstLine[i].A = pSrcLine[i].A;
		} /* for (A_long i = 0; i < sizeX; i++) */
	} /* for (A_long j = 0; j < sizeY; j++) */

	return;
}


template <typename T, std::enable_if_t<is_RGB_proc<T>::value>* = nullptr>
PF_Err NoiseClean_AlgoAnisotropicDiffusion
(
	const T* __restrict pSrc,
	      T* __restrict pDst,
	A_long  sizeX,
	A_long  sizeY,
	A_long  srcPitch,
	A_long  dstPitch,
	float   noiseLevel,
	float   timeStep,
	float   maxVal,
	float   dispersion
) noexcept
{
	PF_Pixel_VUYA_32f* pTmp[2]{};
	constexpr float minimalStep = 0.001f;
	PF_Err err = PF_Err_OUT_OF_MEMORY;

	A_long memBlockId = MemoryBufferAlloc (sizeX, sizeY, &pTmp[0], &pTmp[1]);
	if (nullptr != pTmp[0] && nullptr != pTmp[1] && -1 != memBlockId)
	{
		A_long ping = 0x0, pong = 0x1;
		A_long pitchSrc = 0, pitchDst = 0;

		/* convert BGRA image to VUYA_32f */
		imgRGB2YUVConvert (pSrc, pTmp[ping], sizeX, sizeY, srcPitch, sizeX, maxVal, BT709);

		float currentDispersion = 0.0f;
		float currentTimeStep = FastCompute::Min(timeStep, dispersion - currentDispersion);

		do
		{
			err = NoiseClean_AlgoAnisotropicDiffusion (pTmp[ping], pTmp[pong], sizeX, sizeY, sizeX, sizeX, noiseLevel, timeStep, maxVal);

			currentDispersion += currentTimeStep;
			currentTimeStep = FastCompute::Min (timeStep, dispersion - currentDispersion);

			ping ^= 0x1;
			pong ^= 0x1;

		} while (currentDispersion <= dispersion && currentTimeStep > minimalStep && err == PF_Err_NONE);

		/* convert VUYA_32f back to BGRA */
		imgYUV2RGBConvert (pTmp[ping], pDst, sizeX, sizeY, sizeX, dstPitch, maxVal, BT709);

		MemoryBufferRelease (memBlockId);
		memBlockId = -1;
		pTmp[0] = pTmp[1] = nullptr;
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
	PrPixelFormat destinationPixelFormat = PrPixelFormat_Invalid;

	/* This plugin called frop PR - check video fomat */
	if (PF_Err_NONE == (AEFX_SuiteScoper<PF_PixelFormatSuite1>(in_data, kPFPixelFormatSuite, kPFPixelFormatSuiteVersion1, out_data)->GetPixelFormat(output, &destinationPixelFormat)))
	{
		/* read sliders positions */

		const float fNoiseLevel = 3.f;
		const float fTimeStep = 5.f / 10.f;
		const float fDispersion = 10.0f;

		const PF_LayerDef* pfLayer = reinterpret_cast<const PF_LayerDef*>(&params[eNOISE_CLEAN_INPUT]->u.ld);

		switch (destinationPixelFormat)
		{
			case PrPixelFormat_BGRA_4444_8u:
			{
				const PF_Pixel_BGRA_8u* __restrict localSrc = reinterpret_cast<const PF_Pixel_BGRA_8u* __restrict>(pfLayer->data);
					  PF_Pixel_BGRA_8u* __restrict localDst = reinterpret_cast<      PF_Pixel_BGRA_8u* __restrict>(output->data);
				sizeY = pfLayer->extent_hint.bottom - pfLayer->extent_hint.top;
				sizeX = pfLayer->extent_hint.right  - pfLayer->extent_hint.left;
				linePitch = pfLayer->rowbytes / static_cast<A_long>(PF_Pixel_BGRA_8u_size);

				err = NoiseClean_AlgoAnisotropicDiffusion (localSrc, localDst, sizeX, sizeY, linePitch, linePitch, fNoiseLevel, fTimeStep, static_cast<float>(u8_value_white), fDispersion);
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

				err = NoiseClean_AlgoAnisotropicDiffusion (localSrc, localDst, sizeX, sizeY, linePitch, linePitch, fNoiseLevel, fTimeStep, static_cast<float>(u8_value_white), fDispersion);
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

				err = NoiseClean_AlgoAnisotropicDiffusion(localSrc, localDst, sizeX, sizeY, linePitch, linePitch, fNoiseLevel, fTimeStep, f32_value_white, fDispersion);
			}
			break;

			case PrPixelFormat_BGRA_4444_16u:
			{
				const PF_Pixel_BGRA_16u* __restrict localSrc = reinterpret_cast<const PF_Pixel_BGRA_16u* __restrict>(pfLayer->data);
					  PF_Pixel_BGRA_16u* __restrict localDst = reinterpret_cast<      PF_Pixel_BGRA_16u* __restrict>(output->data);
				sizeY = pfLayer->extent_hint.bottom - pfLayer->extent_hint.top;
				sizeX = pfLayer->extent_hint.right   - pfLayer->extent_hint.left;
				linePitch = pfLayer->rowbytes / static_cast<A_long>(PF_Pixel_BGRA_16u_size);

				err = NoiseClean_AlgoAnisotropicDiffusion (localSrc, localDst, sizeX, sizeY, linePitch, linePitch, fNoiseLevel, fTimeStep, static_cast<float>(u16_value_white), fDispersion);
			}
			break;

			case PrPixelFormat_BGRA_4444_32f:
			{
				const PF_Pixel_BGRA_32f* __restrict localSrc = reinterpret_cast<const PF_Pixel_BGRA_32f* __restrict>(pfLayer->data);
					  PF_Pixel_BGRA_32f* __restrict localDst = reinterpret_cast<      PF_Pixel_BGRA_32f* __restrict>(output->data);
				sizeY = pfLayer->extent_hint.bottom - pfLayer->extent_hint.top;
				sizeX = pfLayer->extent_hint.right  - pfLayer->extent_hint.left;
				linePitch = pfLayer->rowbytes / static_cast<A_long>(PF_Pixel_BGRA_32f_size);

				err = NoiseClean_AlgoAnisotropicDiffusion (localSrc, localDst, sizeX, sizeY, linePitch, linePitch, fNoiseLevel, fTimeStep, f32_value_white, fDispersion);
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