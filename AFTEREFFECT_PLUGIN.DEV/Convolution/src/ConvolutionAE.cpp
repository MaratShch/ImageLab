#include "Convolution.hpp"
#include "Kernels.hpp"


static bool ProcessImgInAE_8bits
(
	PF_InData*   __restrict in_data,
	PF_OutData*  __restrict out_data,
	PF_ParamDef* __restrict params[],
	PF_LayerDef* __restrict output,
	const uint32_t& choosed_kernel
) noexcept
{
	const PF_EffectWorld* __restrict input = reinterpret_cast<const PF_EffectWorld* __restrict>(&params[CONVOLUTION_INPUT]->u.ld);
    PF_Pixel8* __restrict localSrc = reinterpret_cast<PF_Pixel8* __restrict>(input->data);
	PF_Pixel8* __restrict localDst = reinterpret_cast<PF_Pixel8* __restrict>(output->data);
	IAbsrtactKernel<int32_t>* __restrict iKernel = nullptr;

	A_long j = 0, i = 0, l = 0, m = 0, k = 0;
	A_long accumR, accumG, accumB, k_idx = 0, j_idx = 0, i_idx = 0, src_idx = 0, dst_idx = 0;

	bool bReturn = true;

	if (nullptr != (iKernel = GetKernel<int32_t>(choosed_kernel)))
	{
		if (true == iKernel->LoadKernel())
		{
			const uint32_t kernSize = iKernel->GetSize();
			const int32_t* __restrict kernArray = iKernel->GetArray();
			const int32_t elements = static_cast<int32_t>(kernSize * kernSize);
			const int32_t halfKernelSize = static_cast<int32_t>(kernSize) >> 1;

			const float kernSum = iKernel->Normalizing();

			const A_long src_pitch = input->rowbytes / sizeof(PF_Pixel8);
			const A_long dst_pitch = output->rowbytes / sizeof(PF_Pixel8);

			const A_long height = output->height;
			const A_long width = output->width;

			for (j = 0; j < height; j++)
			{
				for (i = 0; i < width; i++)
				{
					k_idx = elements - 1;
					accumR = accumG = accumB = 0;

					for (l = -halfKernelSize; l <= halfKernelSize; l++)
					{
						j_idx = j + l;
						if (j_idx < 0 || j_idx >= height)
						{
							k_idx--;
							continue;
						}

						for (m = -halfKernelSize; m <= halfKernelSize; m++)
						{
							i_idx = i + m;
							if (i_idx < 0 || i_idx >= width)
							{
								k_idx--;
								continue;
							}

							src_idx = j_idx * src_pitch + i_idx;

							accumR += (static_cast<int32_t>(localSrc[src_idx].red)   * kernArray[k_idx]);
							accumG += (static_cast<int32_t>(localSrc[src_idx].green) * kernArray[k_idx]);
							accumB += (static_cast<int32_t>(localSrc[src_idx].blue)  * kernArray[k_idx]);

							k_idx--;

						} /* for (m = -halfKernelSize; m < halfKernelSize; m++) */

					} /* for (l = -halfKernelSize; l < halfKernelSize; l++) */

					const float outR = static_cast<float>(accumR) / kernSum;
					const float outG = static_cast<float>(accumG) / kernSum;
					const float outB = static_cast<float>(accumB) / kernSum;

					dst_idx = j * dst_pitch + i;
					localDst[dst_idx].red = static_cast<A_u_char>(CLAMP_VALUE(outR, 0.f, 255.f));
					localDst[dst_idx].green = static_cast<A_u_char>(CLAMP_VALUE(outG, 0.f, 255.f));
					localDst[dst_idx].blue = static_cast<A_u_char>(CLAMP_VALUE(outB, 0.f, 255.f));
					localDst[dst_idx].alpha = localSrc[src_idx].alpha;
				}
			}

		} /* if (true == iKernel->LoadKernel()) */
		else
			bReturn = false;

	} /* if (nullptr != (iKernel = GetKernel<int32_t>(choosed_kernel))) */
	else
		bReturn = false;

	/* something going wrong - let's simply copy input buffer to output */
	if (false == bReturn)
	{
		AEFX_SuiteScoper<PF_WorldTransformSuite1> worldTransformSite =
			AEFX_SuiteScoper<PF_WorldTransformSuite1>(
				in_data,
				kPFWorldTransformSuite,
				kPFWorldTransformSuiteVersion1,
				out_data);

		bReturn =
			(PF_Err_NONE == worldTransformSite->copy(in_data->effect_ref, &params[CONVOLUTION_INPUT]->u.ld, output, nullptr, nullptr)) ? true : false;
	}

	return bReturn;
}



static bool ProcessImgInAE_16bits
(
	PF_InData*   __restrict in_data,
	PF_OutData*  __restrict out_data,
	PF_ParamDef* __restrict params[],
	PF_LayerDef* __restrict output,
	const uint32_t& choosed_kernel
) noexcept
{
	const PF_EffectWorld* __restrict input = reinterpret_cast<const PF_EffectWorld* __restrict>(&params[CONVOLUTION_INPUT]->u.ld);
	PF_Pixel16* __restrict localSrc = reinterpret_cast<PF_Pixel16* __restrict>(input->data);
	PF_Pixel16* __restrict localDst = reinterpret_cast<PF_Pixel16* __restrict>(output->data);
	IAbsrtactKernel<int32_t>* __restrict iKernel = nullptr;

	A_long j = 0, i = 0, l = 0, m = 0, k = 0;
	A_long accumR, accumG, accumB, k_idx = 0, j_idx = 0, i_idx = 0, src_idx = 0, dst_idx = 0;

	bool bReturn = true;

	if (nullptr != (iKernel = GetKernel<int32_t>(choosed_kernel)))
	{
		if (true == iKernel->LoadKernel())
		{
			const uint32_t kernSize = iKernel->GetSize();
			const int32_t* __restrict kernArray = iKernel->GetArray();
			const int32_t elements = static_cast<int32_t>(kernSize * kernSize);
			const int32_t halfKernelSize = static_cast<int32_t>(kernSize) >> 1;

			const float kernSum = iKernel->Normalizing();

			const A_long src_pitch = input->rowbytes / sizeof(PF_Pixel16);
			const A_long dst_pitch = output->rowbytes / sizeof(PF_Pixel16);

			const A_long height = output->height;
			const A_long width = output->width;

			for (j = 0; j < height; j++)
			{
				for (i = 0; i < width; i++)
				{
					k_idx = elements - 1;
					accumR = accumG = accumB = 0;

					for (l = -halfKernelSize; l <= halfKernelSize; l++)
					{
						j_idx = j + l;
						if (j_idx < 0 || j_idx >= height)
						{
							k_idx--;
							continue;
						}

						for (m = -halfKernelSize; m <= halfKernelSize; m++)
						{
							i_idx = i + m;
							if (i_idx < 0 || i_idx >= width)
							{
								k_idx--;
								continue;
							}

							src_idx = j_idx * src_pitch + i_idx;

							accumR += (static_cast<int32_t>(localSrc[src_idx].red)   * kernArray[k_idx]);
							accumG += (static_cast<int32_t>(localSrc[src_idx].green) * kernArray[k_idx]);
							accumB += (static_cast<int32_t>(localSrc[src_idx].blue)  * kernArray[k_idx]);

							k_idx--;

						} /* for (m = -halfKernelSize; m < halfKernelSize; m++) */

					} /* for (l = -halfKernelSize; l < halfKernelSize; l++) */

					const float outR = static_cast<float>(accumR) / kernSum;
					const float outG = static_cast<float>(accumG) / kernSum;
					const float outB = static_cast<float>(accumB) / kernSum;

					dst_idx = j * dst_pitch + i;
					localDst[dst_idx].red = static_cast<A_u_short>(CLAMP_VALUE(outR, 0.f, 32767.f));
					localDst[dst_idx].green = static_cast<A_u_short>(CLAMP_VALUE(outG, 0.f, 32767.f));
					localDst[dst_idx].blue = static_cast<A_u_short>(CLAMP_VALUE(outB, 0.f, 32767.f));
					localDst[dst_idx].alpha = localSrc[src_idx].alpha;
				}
			}

		} /* if (true == iKernel->LoadKernel()) */
		else
			bReturn = false;

	} /* if (nullptr != (iKernel = GetKernel<int32_t>(choosed_kernel))) */
	else
		bReturn = false;

	/* something going wrong - let's simply copy input buffer to output */
	if (false == bReturn)
	{
		AEFX_SuiteScoper<PF_WorldTransformSuite1> worldTransformSite =
			AEFX_SuiteScoper<PF_WorldTransformSuite1>(
				in_data,
				kPFWorldTransformSuite,
				kPFWorldTransformSuiteVersion1,
				out_data);

		bReturn =
			(PF_Err_NONE == worldTransformSite->copy_hq (in_data->effect_ref, &params[CONVOLUTION_INPUT]->u.ld, output, nullptr, nullptr)) ? true : false;
	}

	return bReturn;
}


bool ProcessImgInAE
(
	PF_InData*   __restrict in_data,
	PF_OutData*  __restrict out_data,
	PF_ParamDef* __restrict params[],
	PF_LayerDef* __restrict output,
	const uint32_t& choosed_kernel
) noexcept
{
	return ( PF_WORLD_IS_DEEP(output) ?
		ProcessImgInAE_16bits(in_data, out_data, params, output, choosed_kernel) :
        ProcessImgInAE_8bits(in_data, out_data, params, output, choosed_kernel) );
}