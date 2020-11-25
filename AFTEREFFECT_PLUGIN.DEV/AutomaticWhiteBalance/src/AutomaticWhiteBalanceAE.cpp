#include "AutomaticWhiteBalance.hpp"
#include "AlgCommonFunctins.hpp"


static bool ProcessImgInAE_8bits
(
	PF_InData*   __restrict in_data,
	PF_OutData*  __restrict out_data,
	PF_ParamDef* __restrict params[],
	PF_LayerDef* __restrict output
) noexcept
{
	CACHE_ALIGN float U_avg[gMaxCnt]{};
	CACHE_ALIGN float V_avg[gMaxCnt]{};

	const PF_EffectWorld* __restrict input = reinterpret_cast<const PF_EffectWorld* __restrict>(&params[AWB_INPUT]->u.ld);
	PF_Pixel_ARGB_8u*  __restrict localSrc = reinterpret_cast<PF_Pixel_ARGB_8u* __restrict>(input->data);
	PF_Pixel_ARGB_8u*  __restrict localDst = reinterpret_cast<PF_Pixel_ARGB_8u* __restrict>(output->data);

	PF_Pixel_ARGB_8u* __restrict srcInput = nullptr;
	PF_Pixel_ARGB_8u* __restrict dstOutput = nullptr;

	const A_long height = output->height;
	const A_long width = output->width;
	const A_long src_line_pitch = input->rowbytes / sizeof(PF_Pixel8);
	const A_long dst_line_pitch = output->rowbytes / sizeof(PF_Pixel8);

	int32_t srcIdx = 0;
	int32_t dstIdx = 1;

	A_long srcPitch = 0;
	A_long dstPitch = 0;

	float T = 0.30f;
	float uAvg, vAvg;

	const int32_t sliderIterCnt = 2;
	const int32_t iterCnt = MIN(sliderIterCnt, gMaxCnt);

	/* test temporary buffers size and re-allocate if required new size */
	CAlgMemHandler* pMemHandler = ::getMemoryHandler();
	const size_t tmpMemSize = height * width * sizeof(PF_Pixel_ARGB_8u);

	if (nullptr != pMemHandler && true == pMemHandler->MemInit(tmpMemSize))
	{

		/* pass iterations in corresponding to slider position */
		for (A_long k = 0; k < iterCnt; k++)
		{
			if (0 == k)
			{
				srcInput = localSrc;
				dstIdx++;
				dstIdx &= 0x1;
				dstOutput = reinterpret_cast<PF_Pixel_ARGB_8u* __restrict>(pMemHandler->GetMemory(dstIdx));
				srcPitch = src_line_pitch;
				dstPitch = width;
			}
			else if ((iterCnt - 1) == k)
			{
				srcIdx = dstIdx;
				srcInput = reinterpret_cast<PF_Pixel_ARGB_8u* __restrict>(pMemHandler->GetMemory(srcIdx));
				dstOutput = localDst;
				srcPitch = width;
				dstPitch = dst_line_pitch;
			} /* if (k > 0) */
			else
			{
				srcIdx = dstIdx;
				dstIdx++;
				dstIdx &= 0x1;
				srcInput  = reinterpret_cast<PF_Pixel_ARGB_8u* __restrict>(pMemHandler->GetMemory(srcIdx));
				dstOutput = reinterpret_cast<PF_Pixel_ARGB_8u* __restrict>(pMemHandler->GetMemory(dstIdx));
				srcPitch = dstPitch = width;
			}

			/* collect statistics about image and compute averages values for U and for V components */
			collect_rgb_statistics(srcInput, width, height, srcPitch, T, BT601, &uAvg, &vAvg);
			U_avg[k] = uAvg;
			V_avg[k] = vAvg;

			if (k > 0)
			{
				const float U_diff = U_avg[k] - U_avg[k - 1];
				const float V_diff = V_avg[k] - V_avg[k - 1];

				const float normVal = asqrt(U_diff * U_diff + V_diff * V_diff);

				if (normVal < algAWBepsilon)
				{
					// U and V no longer improving, so just copy source to destination and break the loop
					simple_image_copy (srcInput, localDst, height, width, srcPitch, dst_line_pitch);

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
			} /* if (k > 0) */

			  /* compute correction matrix */
			CACHE_ALIGN float correctionMatrix[3]{};
			compute_correction_matrix(uAvg, vAvg, BT601, DAYLIGHT, CHROMATIC_CAT02, correctionMatrix);

			/* in second: perform image color correction */
			image_rgb_correction(srcInput, dstOutput, width, height,
				srcPitch, dstPitch, correctionMatrix);

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


static bool ProcessImgInAE_16bits
(
	PF_InData*   __restrict in_data,
	PF_OutData*  __restrict out_data,
	PF_ParamDef* __restrict params[],
	PF_LayerDef* __restrict output
) noexcept
{
	return true;
}


PF_Err ProcessImgInAE
(
	PF_InData*   __restrict in_data,
	PF_OutData*  __restrict out_data,
	PF_ParamDef* __restrict params[],
	PF_LayerDef* __restrict output
) noexcept
{
	return (true == (PF_WORLD_IS_DEEP(output) ?
		ProcessImgInAE_16bits(in_data, out_data, params, output) :
		ProcessImgInAE_8bits (in_data, out_data, params, output) ) ? PF_Err_NONE : PF_Err_INVALID_INDEX);
}