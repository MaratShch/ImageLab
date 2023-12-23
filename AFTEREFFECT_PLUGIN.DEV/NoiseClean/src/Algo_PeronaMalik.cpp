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
			const T& west    = pCurrLine[prevPixel];
			const T& current = pCurrLine[i];
			const T& east    = pCurrLine[nextPixel];
			const T& south   = pNextLine[i];

			const float currentY  = static_cast<float>(current.Y);
			const float diffNorth = static_cast<float>(north.Y) - currentY;
			const float diffWest  = static_cast<float>(west.Y ) - currentY;
			const float diffEast  = static_cast<float>(east.Y ) - currentY;
			const float diffSouth = static_cast<float>(south.Y) - currentY;

			const float fSum = Gfunction (diffNorth, noiseLevel) * diffNorth +
				               Gfunction (diffWest,  noiseLevel) * diffWest  +
				               Gfunction (diffEast,  noiseLevel) * diffEast  +
				               Gfunction (diffSouth, noiseLevel) * diffSouth;
			
			const float finalY = CLAMP_VALUE(currentY + fSum * timeStep, 0.f, maxVal);
			pDstLine[i].V = current.V;
			pDstLine[i].U = current.U;
			pDstLine[i].Y = finalY;
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
	A_long i, j;

	const A_long lastLine  = sizeY - 1;
	const A_long lastPixel = sizeX - 1;

	for (j = 0; j < sizeY; j++)
	{
		const A_long prevLine = FastCompute::Max(0, j - 1);
		const A_long nextLine = FastCompute::Min(lastLine, j + 1);

		const T* __restrict pPrevLine = pSrc + srcPitch * prevLine;
		const T* __restrict pCurrLine = pSrc + srcPitch * j;
		const T* __restrict pNextLine = pSrc + srcPitch * nextLine;
		      T* __restrict pDstLine  = pDst + dstPitch * j;

		for (i = 0; i < sizeX; i++)
		{
			const A_long prevPixel = FastCompute::Max(0, i - 1);
			const A_long nextPixel = FastCompute::Min(lastPixel, i + 1);

			const T& north   = pPrevLine[i];
			const T& west    = pCurrLine[prevPixel];
			const T& current = pCurrLine[i];
			const T& east    = pCurrLine[nextPixel];
			const T& south   = pNextLine[i];

			const float currentR = static_cast<float>(current.R);
			const float currentG = static_cast<float>(current.G);
			const float currentB = static_cast<float>(current.B);

			const float diffNorthR = static_cast<float>(north.R) - currentR;
			const float diffNorthG = static_cast<float>(north.G) - currentG;
			const float diffNorthB = static_cast<float>(north.B) - currentB;

			const float diffWestR  = static_cast<float>(west.R) - currentR;
			const float diffWestG  = static_cast<float>(west.G) - currentG;
			const float diffWestB  = static_cast<float>(west.B) - currentB;

			const float diffEastR  = static_cast<float>(east.R) - currentR;
			const float diffEastG  = static_cast<float>(east.G) - currentG;
			const float diffEastB  = static_cast<float>(east.B) - currentB;

			const float diffSouthR = static_cast<float>(south.R) - currentR;
			const float diffSouthG = static_cast<float>(south.G) - currentG;
			const float diffSouthB = static_cast<float>(south.B) - currentB;

			const float fSumR = Gfunction (diffNorthR, noiseLevel) * diffNorthR +
				                Gfunction (diffWestR,  noiseLevel) * diffWestR  +
				                Gfunction (diffEastR,  noiseLevel) * diffEastR  +
				                Gfunction (diffSouthR, noiseLevel) * diffSouthR;

			const float fSumG = Gfunction (diffNorthG, noiseLevel) * diffNorthG +
				                Gfunction (diffWestG,  noiseLevel) * diffWestG  +
				                Gfunction (diffEastG,  noiseLevel) * diffEastG  +
				                Gfunction (diffSouthG, noiseLevel) * diffSouthG;

			const float fSumB = Gfunction (diffNorthB, noiseLevel) * diffNorthB +
								Gfunction (diffWestB,  noiseLevel) * diffWestB  +
								Gfunction (diffEastB,  noiseLevel) * diffEastB  +
								Gfunction (diffSouthB, noiseLevel) * diffSouthG;

			const float newR = currentR + fSumR * timeStep;
			const float newG = currentG + fSumG * timeStep;
			const float newB = currentB + fSumB * timeStep;

			pDstLine[i].R = CLAMP_VALUE(newR, 0.f, maxVal);
			pDstLine[i].G = CLAMP_VALUE(newG, 0.f, maxVal);
			pDstLine[i].B = CLAMP_VALUE(newB, 0.f, maxVal);
			pDstLine[i].A = current.A;
		}
	}

	return PF_Err_NONE;
}


template <typename T>
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
	float   dispersion,
	float   maxVal,
	float   minimalStep = 0.0010f
) noexcept
{
	T* pTmp[2] {};
#ifdef _DEBUG
	uint64_t dbgLoopCnt = 0u;
#endif
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

#ifdef _DEBUG
			dbgLoopCnt = iterCnt;
#endif

		} while (currentDispersion <= dispersion && currentTimeStep > minimalStep);

		MemoryBufferRelease(memBlockId);
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
		const PF_LayerDef* pfLayer = reinterpret_cast<const PF_LayerDef*>(&params[eNOISE_CLEAN_INPUT]->u.ld);
		constexpr float reciproc10 = 1.f / 10.f;

		/* read sliders positions */
		const float fDispersion = static_cast<const float>(params[eNOISE_CLEAN_ANYSOTROPIC_DISPERSION]->u.sd.value) * reciproc10;
		const float fTimeStep   = static_cast<const float>(params[eNOISE_CLEAN_ANYSOTROPIC_TIMESTEP  ]->u.sd.value) * reciproc10;
		const float fNoiseLevel = static_cast<const float>(params[eNOISE_CLEAN_ANYSOTROPIC_NOISELEVEL]->u.sd.value) * reciproc10;

		switch (destinationPixelFormat)
		{
			case PrPixelFormat_BGRA_4444_8u:
			{
				const PF_Pixel_BGRA_8u* __restrict localSrc = reinterpret_cast<const PF_Pixel_BGRA_8u* __restrict>(pfLayer->data);
					  PF_Pixel_BGRA_8u* __restrict localDst = reinterpret_cast<      PF_Pixel_BGRA_8u* __restrict>(output->data);
				sizeY = pfLayer->extent_hint.bottom - pfLayer->extent_hint.top;
				sizeX = pfLayer->extent_hint.right  - pfLayer->extent_hint.left;
				linePitch = pfLayer->rowbytes / static_cast<A_long>(PF_Pixel_BGRA_8u_size);

				err = NoiseClean_AlgoAnisotropicDiffusion (localSrc, localDst, sizeX, sizeY, linePitch, linePitch, fNoiseLevel, fTimeStep, fDispersion, static_cast<float>(u8_value_white));
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

				err = NoiseClean_AlgoAnisotropicDiffusion (localSrc, localDst, sizeX, sizeY, linePitch, linePitch, fNoiseLevel, fTimeStep, fDispersion, static_cast<float>(u8_value_white));
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

				err = NoiseClean_AlgoAnisotropicDiffusion (localSrc, localDst, sizeX, sizeY, linePitch, linePitch, fNoiseLevel, fTimeStep, fDispersion, f32_value_white);
			}
			break;

			case PrPixelFormat_BGRA_4444_16u:
			{
				const PF_Pixel_BGRA_16u* __restrict localSrc = reinterpret_cast<const PF_Pixel_BGRA_16u* __restrict>(pfLayer->data);
					  PF_Pixel_BGRA_16u* __restrict localDst = reinterpret_cast<      PF_Pixel_BGRA_16u* __restrict>(output->data);
				sizeY = pfLayer->extent_hint.bottom - pfLayer->extent_hint.top;
				sizeX = pfLayer->extent_hint.right  - pfLayer->extent_hint.left;
				linePitch = pfLayer->rowbytes / static_cast<A_long>(PF_Pixel_BGRA_16u_size);

				constexpr float minimalStep = 0.256f;

				err = NoiseClean_AlgoAnisotropicDiffusion (localSrc, localDst, sizeX, sizeY, linePitch, linePitch, fNoiseLevel, fTimeStep, fDispersion, static_cast<float>(u16_value_white), minimalStep);
			}
			break;

			case PrPixelFormat_BGRA_4444_32f:
			{
				const PF_Pixel_BGRA_32f* __restrict localSrc = reinterpret_cast<const PF_Pixel_BGRA_32f* __restrict>(pfLayer->data);
					  PF_Pixel_BGRA_32f* __restrict localDst = reinterpret_cast<      PF_Pixel_BGRA_32f* __restrict>(output->data);
				sizeY = pfLayer->extent_hint.bottom - pfLayer->extent_hint.top;
				sizeX = pfLayer->extent_hint.right  - pfLayer->extent_hint.left;
				linePitch = pfLayer->rowbytes / static_cast<A_long>(PF_Pixel_BGRA_32f_size);

				err = NoiseClean_AlgoAnisotropicDiffusion (localSrc, localDst, sizeX, sizeY, linePitch, linePitch, fNoiseLevel, fTimeStep, fDispersion, f32_value_white);
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


PF_Err NoiseClean_AlgoPeronaMalikAe8
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

	/* read sliders positions */
	const float fDispersion = static_cast<const float>(params[eNOISE_CLEAN_ANYSOTROPIC_DISPERSION]->u.sd.value) * reciproc10;
	const float fTimeStep   = static_cast<const float>(params[eNOISE_CLEAN_ANYSOTROPIC_TIMESTEP]->u.sd.value  ) * reciproc10;
	const float fNoiseLevel = static_cast<const float>(params[eNOISE_CLEAN_ANYSOTROPIC_NOISELEVEL]->u.sd.value) * reciproc10;

	return NoiseClean_AlgoAnisotropicDiffusion (localSrc, localDst, sizeX, sizeY, src_pitch, dst_pitch, fNoiseLevel, fTimeStep, fDispersion, static_cast<float>(u8_value_white));;
}


PF_Err NoiseClean_AlgoPeronaMalikAe16
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

	/* read sliders positions */
	const float fDispersion = static_cast<const float>(params[eNOISE_CLEAN_ANYSOTROPIC_DISPERSION]->u.sd.value) * reciproc10;
	const float fTimeStep   = static_cast<const float>(params[eNOISE_CLEAN_ANYSOTROPIC_TIMESTEP]->u.sd.value  ) * reciproc10;
	const float fNoiseLevel = static_cast<const float>(params[eNOISE_CLEAN_ANYSOTROPIC_NOISELEVEL]->u.sd.value) * reciproc10;

	return NoiseClean_AlgoAnisotropicDiffusion (localSrc, localDst, sizeX, sizeY, src_pitch, dst_pitch, fNoiseLevel, fTimeStep, fDispersion, static_cast<float>(u16_value_white));;
}