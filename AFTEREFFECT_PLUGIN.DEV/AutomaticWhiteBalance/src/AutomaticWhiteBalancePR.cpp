#include "AutomaticWhiteBalance.hpp"
#include "AlgMemoryHandler.hpp"
#include "ColorTransformMatrix.hpp"
#include "FastAriphmetics.hpp"

template <typename T>
inline void simple_image_copy_in_premier
(
	T* __restrict srcPix,
	T* __restrict dstPix,
	const A_long& width,
	const A_long& height,
	const A_long& linePitch
)
{
	__VECTOR_ALIGNED__
	for (A_long i = 0; i < height; i++)
	{
		memcpy(&dstPix[i*linePitch], &srcPix[i*linePitch], width * sizeof(T));
	}

	return;
}


static bool ProcessPrImage_BGRA_4444_8u
(
	PF_InData*   __restrict in_data,
	PF_OutData*  __restrict out_data,
	PF_ParamDef* __restrict params[],
	PF_LayerDef* __restrict output
) noexcept
{
	CACHE_ALIGN float U_avg[gMaxCnt]{};
	CACHE_ALIGN float V_avg[gMaxCnt]{};

	const PF_LayerDef* __restrict pfLayer = reinterpret_cast<PF_LayerDef* __restrict>(&params[AWB_INPUT]->u.ld);
	PF_Pixel_BGRA_8u*  __restrict localSrc = reinterpret_cast<PF_Pixel_BGRA_8u* __restrict>(pfLayer->data);
	PF_Pixel_BGRA_8u*  __restrict localDst = reinterpret_cast<PF_Pixel_BGRA_8u* __restrict>(output->data);

	PF_Pixel_BGRA_8u* __restrict srcInput  = nullptr;
	PF_Pixel_BGRA_8u* __restrict dstOutput = nullptr;

	const A_long height = pfLayer->extent_hint.bottom - pfLayer->extent_hint.top;
	const A_long width  = pfLayer->extent_hint.right  - pfLayer->extent_hint.left;
	const A_long line_pitch = pfLayer->rowbytes / static_cast<A_long>(PF_Pixel_BGRA_8u_size);

	/* test temporary buffers size and re-allocate if required new size */
	CAlgMemHandler* pMemHandler = ::getMemoryHandler();
	const size_t tmpMemSize = height * width * sizeof(PF_Pixel_BGRA_8u);

	int32_t totalGray = 0;
	int32_t srcIdx = 0;
	int32_t dstIdx = 1;

	float Y, U, V;
	float U_bar, V_bar, F, T = 0.30f;
	float newR, newG, newB;

	if (nullptr != pMemHandler && true == pMemHandler->MemInit(tmpMemSize))
	{
		const A_long iterCnt = 2;

		A_long i, j, k;

		const float* __restrict colorMatrixIn = RGB2YUV[BT601];
		const float* __restrict colorMatrixOut = YUV2RGB[BT601];

		const float* __restrict illuminate = GetIlluminate(DAYLIGHT);
		const float* __restrict colorAdaptation = GetColorAdaptation(CHROMATIC_CAT02);
		const float* __restrict colorAdaptationInv = GetColorAdaptationInv(CHROMATIC_CAT02);


		/* pass iterations in corresponding to slider position */
		for (k = 0; k < iterCnt; k++)
		{
			if (0 == k)
			{
				srcInput = localSrc;
				dstIdx++;
				dstIdx &= 0x1;
				dstOutput = reinterpret_cast<PF_Pixel_BGRA_8u* __restrict>(pMemHandler->GetMemory(dstIdx));
			} else if ((iterCnt-1) == k)
			{
				srcIdx = dstIdx;
				srcInput  = reinterpret_cast<PF_Pixel_BGRA_8u* __restrict>(pMemHandler->GetMemory(srcIdx));
				dstOutput = localDst;
			} /* if (k > 0) */
			else
			{
				srcIdx = dstIdx;
				dstIdx++;
				dstIdx &= 0x1;
				srcInput = reinterpret_cast<PF_Pixel_BGRA_8u* __restrict>(pMemHandler->GetMemory(srcIdx));
				dstOutput = reinterpret_cast<PF_Pixel_BGRA_8u* __restrict>(pMemHandler->GetMemory(dstIdx));
			}

			/* in first: collect statistics */
			U_bar = V_bar = 0.f;

			__VECTOR_ALIGNED__
			for (j = 0; j < height; j++)
			{
				const PF_Pixel_BGRA_8u* __restrict pSrcLine = localSrc + j * line_pitch;
				for (i = 0; i < width; i++)
				{
					/* convert RGB to YUV color space */
					Y = pSrcLine[i].R * colorMatrixIn[0] + pSrcLine[i].G * colorMatrixIn[1] + pSrcLine[i].B * colorMatrixIn[2];
					U = pSrcLine[i].R * colorMatrixIn[3] + pSrcLine[i].G * colorMatrixIn[4] + pSrcLine[i].B * colorMatrixIn[5];
					V = pSrcLine[i].R * colorMatrixIn[6] + pSrcLine[i].G * colorMatrixIn[7] + pSrcLine[i].B * colorMatrixIn[8];

					F = (abs(U) + abs(V)) / Y;
					if (F < T)
					{
						totalGray++;
						U_bar += U;
						V_bar += V;
					} /* if (F < T) */

				} /* for (i = 0; i < width; i++) */

			} /* for (j = 0; j < height; j++) */

			U_avg[iterCnt] = U_bar / totalGray;
			V_avg[iterCnt] = V_bar / totalGray;

			if (0 != iterCnt)
			{
				const float U_diff = U_avg[iterCnt] - U_avg[iterCnt - 1];
				const float V_diff = V_avg[iterCnt] - V_avg[iterCnt - 1];

				const float normVal = asqrt(U_diff * U_diff + V_diff * V_diff);

				if (normVal < algAWBepsilon)
				{
					// U and V no longer improving, so just copy source to destination and break the loop
					simple_image_copy_in_premier (srcInput, localDst, height, width, line_pitch);

					/* release temporary memory buffers on exit from function */
					if (0 == k)
						pMemHandler->ReleaseMemory(dstIdx);
					else if ((iterCnt - 1) == k)
						pMemHandler->ReleaseMemory(srcIdx);
					else
					{
						pMemHandler->ReleaseMemory(srcIdx);
						pMemHandler->ReleaseMemory(dstIdx);
					}

					return true; // U and V no longer improving
				}
			}

			const float restored_R = 100.0f * colorMatrixOut[0] + U_avg[iterCnt] * colorMatrixOut[1] + V_avg[iterCnt] * colorMatrixOut[2];
			const float restored_G = 100.0f * colorMatrixOut[3] + U_avg[iterCnt] * colorMatrixOut[4] + V_avg[iterCnt] * colorMatrixOut[5];
			const float restored_B = 100.0f * colorMatrixOut[6] + U_avg[iterCnt] * colorMatrixOut[7] + V_avg[iterCnt] * colorMatrixOut[8];

			// calculate xy chromaticity
			const float xX = restored_R * sRGBtoXYZ[0] + restored_G * sRGBtoXYZ[1] + restored_B * sRGBtoXYZ[2];
			const float xY = restored_R * sRGBtoXYZ[3] + restored_G * sRGBtoXYZ[4] + restored_B * sRGBtoXYZ[5];
			const float xZ = restored_R * sRGBtoXYZ[6] + restored_G * sRGBtoXYZ[7] + restored_B * sRGBtoXYZ[8];

			const float xXYZSum  = xX + xY + xZ;
			const float xyEst[2] = { xX / xXYZSum, xY / xXYZSum };
			const float xyEstDiv = 100.0f / xyEst[1];

			// Converts xyY chromaticity to CIE XYZ.
			const float xyzEst[3] = { xyEstDiv * xyEst[0], 
				                      100.0f, 
				                      xyEstDiv * (1.0f - xyEst[0] - xyEst[1]) };

			const float gainTarget[3] =
			{
				illuminate[0] * colorAdaptation[0] + illuminate[1] * colorAdaptation[1] + illuminate[2] * colorAdaptation[2],
				illuminate[0] * colorAdaptation[3] + illuminate[1] * colorAdaptation[4] + illuminate[2] * colorAdaptation[5],
				illuminate[0] * colorAdaptation[6] + illuminate[1] * colorAdaptation[7] + illuminate[2] * colorAdaptation[8]
			};

			const float gainEstimated[3] =
			{
				xyzEst[0] * colorAdaptation[0] + xyzEst[1] * colorAdaptation[1] + xyzEst[2] * colorAdaptation[2],
				xyzEst[0] * colorAdaptation[3] + xyzEst[1] * colorAdaptation[4] + xyzEst[2] * colorAdaptation[5],
				xyzEst[0] * colorAdaptation[6] + xyzEst[1] * colorAdaptation[7] + xyzEst[2] * colorAdaptation[8]
			};

			const float finalGain[3] =
			{
				gainTarget[0] / gainEstimated[0],
				gainTarget[1] / gainEstimated[1],
				gainTarget[2] / gainEstimated[2]
			};

			const float diagGain[9] =
			{
				finalGain[0], 0.0f, 0.0f,
				0.0f, finalGain[1], 0.0f,
				0.0f, 0.0f, finalGain[2]
			};

			const float mulGain[9] = 
			{
				diagGain[0] * colorAdaptation[0] + diagGain[1] * colorAdaptation[3] + diagGain[2] * colorAdaptation[6],
				diagGain[0] * colorAdaptation[1] + diagGain[1] * colorAdaptation[4] + diagGain[2] * colorAdaptation[7],
				diagGain[0] * colorAdaptation[2] + diagGain[1] * colorAdaptation[5] + diagGain[2] * colorAdaptation[8],

				diagGain[3] * colorAdaptation[0] + diagGain[4] * colorAdaptation[3] + diagGain[5] * colorAdaptation[6],
				diagGain[3] * colorAdaptation[1] + diagGain[4] * colorAdaptation[4] + diagGain[5] * colorAdaptation[7],
				diagGain[3] * colorAdaptation[2] + diagGain[4] * colorAdaptation[5] + diagGain[5] * colorAdaptation[8],

				diagGain[6] * colorAdaptation[0] + diagGain[7] * colorAdaptation[3] + diagGain[8] * colorAdaptation[6],
				diagGain[6] * colorAdaptation[1] + diagGain[7] * colorAdaptation[4] + diagGain[8] * colorAdaptation[7],
				diagGain[6] * colorAdaptation[2] + diagGain[7] * colorAdaptation[5] + diagGain[8] * colorAdaptation[8]
			};

			const float outMatrix[9] = 
			{
				colorAdaptationInv[0] * mulGain[0] + colorAdaptationInv[1] * mulGain[3] + colorAdaptationInv[2] * mulGain[6],
				colorAdaptationInv[0] * mulGain[1] + colorAdaptationInv[1] * mulGain[4] + colorAdaptationInv[2] * mulGain[7],
				colorAdaptationInv[0] * mulGain[2] + colorAdaptationInv[1] * mulGain[5] + colorAdaptationInv[2] * mulGain[8],

				colorAdaptationInv[3] * mulGain[0] + colorAdaptationInv[4] * mulGain[3] + colorAdaptationInv[5] * mulGain[6],
				colorAdaptationInv[3] * mulGain[1] + colorAdaptationInv[4] * mulGain[4] + colorAdaptationInv[5] * mulGain[7],
				colorAdaptationInv[3] * mulGain[2] + colorAdaptationInv[4] * mulGain[5] + colorAdaptationInv[5] * mulGain[8],

				colorAdaptationInv[6] * mulGain[0] + colorAdaptationInv[7] * mulGain[3] + colorAdaptationInv[8] * mulGain[6],
				colorAdaptationInv[6] * mulGain[1] + colorAdaptationInv[7] * mulGain[4] + colorAdaptationInv[8] * mulGain[7],
				colorAdaptationInv[6] * mulGain[2] + colorAdaptationInv[7] * mulGain[5] + colorAdaptationInv[8] * mulGain[8],
			};

			const float mult[9] = 
			{
				XYZtosRGB[0] * outMatrix[0] + XYZtosRGB[1] * outMatrix[3] + XYZtosRGB[2] * outMatrix[6],
				XYZtosRGB[0] * outMatrix[1] + XYZtosRGB[1] * outMatrix[4] + XYZtosRGB[2] * outMatrix[7],
				XYZtosRGB[0] * outMatrix[2] + XYZtosRGB[1] * outMatrix[5] + XYZtosRGB[2] * outMatrix[8],

				XYZtosRGB[3] * outMatrix[0] + XYZtosRGB[4] * outMatrix[3] + XYZtosRGB[5] * outMatrix[6],
				XYZtosRGB[3] * outMatrix[1] + XYZtosRGB[4] * outMatrix[4] + XYZtosRGB[5] * outMatrix[7],
				XYZtosRGB[3] * outMatrix[2] + XYZtosRGB[4] * outMatrix[5] + XYZtosRGB[5] * outMatrix[8],

				XYZtosRGB[6] * outMatrix[0] + XYZtosRGB[7] * outMatrix[3] + XYZtosRGB[8] * outMatrix[6],
				XYZtosRGB[6] * outMatrix[1] + XYZtosRGB[7] * outMatrix[4] + XYZtosRGB[8] * outMatrix[7],
				XYZtosRGB[6] * outMatrix[2] + XYZtosRGB[7] * outMatrix[5] + XYZtosRGB[8] * outMatrix[8]
			};

			CACHE_ALIGN const float correctionMatrix[3] = 
			{
				mult[0] * sRGBtoXYZ[0] + mult[1] * sRGBtoXYZ[3] + mult[2] * sRGBtoXYZ[6],
				mult[3] * sRGBtoXYZ[1] + mult[4] * sRGBtoXYZ[4] + mult[5] * sRGBtoXYZ[7],
				mult[6] * sRGBtoXYZ[2] + mult[7] * sRGBtoXYZ[5] + mult[8] * sRGBtoXYZ[8]
			};


			newR = newG = newB = 0.0f;

			/* in second: perform balance based on computed coefficients */
			for (j = 0; j < height; j++)
			{
				const PF_Pixel_BGRA_8u* __restrict pSrcLine = srcInput  + j * (srcInput  == localSrc ? line_pitch : width);
				const PF_Pixel_BGRA_8u* __restrict pOrigSrc = localSrc  + j * line_pitch;
				      PF_Pixel_BGRA_8u* __restrict pDstLine = dstOutput + j * (dstOutput == localDst ? line_pitch : width);

				__VECTOR_ALIGNED__
				for (i = 0; i < width; i++)
				{
					newB = pSrcLine[i].B * correctionMatrix[2];
					newG = pSrcLine[i].G * correctionMatrix[1];
					newR = pSrcLine[i].R * correctionMatrix[0];

					pDstLine[i].B = static_cast<A_u_char>(CLAMP_VALUE(newB, 0.f, 255.f));
					pDstLine[i].G = static_cast<A_u_char>(CLAMP_VALUE(newG, 0.f, 255.f));
					pDstLine[i].R = static_cast<A_u_char>(CLAMP_VALUE(newR, 0.f, 255.f));
					pDstLine[i].A = pOrigSrc[i].A; /* copy ALPHA channel from source */

				} /* for (i = 0; i < width; i++) */

			} /* for (j = 0; j < height; j++) */


			/* release temporary memory buffers on exit from function */
			if (0 == k)
				pMemHandler->ReleaseMemory(dstIdx);
			else if ((iterCnt - 1) == k)
				pMemHandler->ReleaseMemory(srcIdx);
			else
			{
				pMemHandler->ReleaseMemory(srcIdx);
				pMemHandler->ReleaseMemory(dstIdx);
			}

		} /* for (k = 0; k < iterCnt; k++) */


	} /* if (true == getMemoryHandler()->MemInit(tmpMemSize)) */

	return true;
}


PF_Err ProcessImgInPR
(
	PF_InData*   __restrict in_data,
	PF_OutData*  __restrict out_data,
	PF_ParamDef* __restrict params[],
	PF_LayerDef* __restrict output,
	const PrPixelFormat&    pixelFormat
) noexcept
{
	bool bValue = true;

	/* acquire controls parameters */

	switch (pixelFormat)
	{
		case PrPixelFormat_BGRA_4444_8u:
			bValue = ProcessPrImage_BGRA_4444_8u(in_data, out_data, params, output);
		break;

		default:
			bValue = false;
		break;
	}


	return (true == bValue ? PF_Err_NONE : PF_Err_INTERNAL_STRUCT_DAMAGED);
}