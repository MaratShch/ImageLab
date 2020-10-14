#include "Convolution.hpp"
#include "Kernels.hpp"

static void
ProcessPrImage_BGRA_4444_8u
(
	PF_InData*   __restrict in_data,
	PF_OutData*  __restrict out_data,
	PF_ParamDef* __restrict params[],
	PF_LayerDef* __restrict output
)
{
	const PF_LayerDef* __restrict pfLayer  = reinterpret_cast<PF_LayerDef* __restrict>(&params[0]->u.ld);
	PF_Pixel_BGRA_8u*  __restrict localSrc = reinterpret_cast<PF_Pixel_BGRA_8u* __restrict>(pfLayer->data);
	PF_Pixel_BGRA_8u*  __restrict localDst = reinterpret_cast<PF_Pixel_BGRA_8u* __restrict>(output->data);

	const PF_ParamValue convKernelType{ params[KERNEL_CHECKBOX]->u.pd.value };
	const uint32_t choosedKernel = static_cast<uint32_t>(convKernelType);
	IAbsrtactKernel<int32_t>* iKernel = GetKernel<int32_t>(choosedKernel - 1);

	A_long i = 0, j = 0;
	A_long l = 0, m = 0, k = 0;

	A_long accumR, accumG, accumB, k_idx = 0, j_idx = 0, i_idx = 0, idx = 0, dst_idx = 0;

	if (nullptr != iKernel)
	{
		if (true == iKernel->LoadKernel())
		{
			const uint32_t kernSize  = iKernel->GetSize();
			const int32_t* kernArray = iKernel->GetArray();
			const int32_t elements   = static_cast<int32_t>(kernSize * kernSize);
			const int32_t  halfKernelSize = static_cast<int32_t>(kernSize) >> 1;

			float kernSum = 0.f;
			for (k = 0; k < elements; k++)
			{
				kernSum += static_cast<float>(kernArray[k]);
			}

			const A_long height = output->extent_hint.bottom - output->extent_hint.top;
			const A_long width = in_data->width;
			const A_long line_pitch = pfLayer->rowbytes / static_cast<A_long>(PF_Pixel_BGRA_8u_size);

			for (j = 0; j < height; j++)
			{
				const A_long line_idx = j * line_pitch;

				for (i = 0; i < in_data->width; i++)
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

							idx = j_idx * line_pitch + i_idx;

							accumR += localSrc[idx].R * kernArray[k_idx];
							accumG += localSrc[idx].G * kernArray[k_idx];
							accumB += localSrc[idx].B * kernArray[k_idx];
							
							k_idx--;

						} /* for (m = -halfKernelSize; m < halfKernelSize; m++) */

					} /* for (l = -halfKernelSize; l < halfKernelSize; l++) */

					const float outR = static_cast<float>(accumR) / kernSum;
					const float outG = static_cast<float>(accumG) / kernSum;
					const float outB = static_cast<float>(accumB) / kernSum;

					dst_idx = line_idx + i;
					localDst[dst_idx].R = static_cast<A_u_char>(CLAMP_VALUE(outR, 0.f, 255.f));
					localDst[dst_idx].G = static_cast<A_u_char>(CLAMP_VALUE(outG, 0.f, 255.f));
					localDst[dst_idx].B = static_cast<A_u_char>(CLAMP_VALUE(outB, 0.f, 255.f));

				} /* for (i = 0; i < in_data->width; i++) */

			} /* for (j = 0; j < height; j++) */
		
		} /* if (true == iKernel->LoadKernel()) */

	} /* if (nullptr != iKernel) */

	return;
}


bool ProcessImgInPR
(
	PF_InData*   __restrict in_data,
	PF_OutData*  __restrict out_data,
	PF_ParamDef* __restrict params[],
	PF_LayerDef* __restrict output,
	const PrPixelFormat& destinationPixelFormat
)
{
		switch (destinationPixelFormat)
		{
			case PrPixelFormat_BGRA_4444_8u:
				ProcessPrImage_BGRA_4444_8u(in_data, out_data, params, output);
			break;

			default:
			break;
		} /* switch (destinationPixelFormat) */

	return true;
}