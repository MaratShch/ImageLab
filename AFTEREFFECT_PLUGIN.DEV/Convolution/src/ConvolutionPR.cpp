#include "Convolution.hpp"
#include "Kernels.hpp"


static void
ProcessPrImage_BGRA_4444_8u
(
	PF_InData*   __restrict in_data,
	PF_OutData*  __restrict out_data,
	PF_ParamDef* __restrict params[],
	PF_LayerDef* __restrict output,
	const uint32_t& choosedKernel
) noexcept
{
	IAbsrtactKernel<int32_t>* iKernel = nullptr;
	const PF_LayerDef* __restrict pfLayer  = reinterpret_cast<PF_LayerDef* __restrict>(&params[0]->u.ld);
	PF_Pixel_BGRA_8u*  __restrict localSrc = reinterpret_cast<PF_Pixel_BGRA_8u* __restrict>(pfLayer->data);
	PF_Pixel_BGRA_8u*  __restrict localDst = reinterpret_cast<PF_Pixel_BGRA_8u* __restrict>(output->data);

	A_long i = 0, j = 0;
	A_long l = 0, m = 0, k = 0;
	A_long accumR, accumG, accumB, k_idx = 0, j_idx = 0, i_idx = 0, src_idx = 0, dst_idx = 0;

	if (nullptr != (iKernel = GetKernel<int32_t>(choosedKernel)))
	{
		if (true == iKernel->LoadKernel())
		{
			const uint32_t kernSize  = iKernel->GetSize();
			const int32_t* __restrict kernArray = iKernel->GetArray();
			const int32_t elements   = static_cast<int32_t>(kernSize * kernSize);
			const int32_t halfKernelSize = static_cast<int32_t>(kernSize) >> 1;

			const float kernSum = iKernel->Normalizing();

			const A_long height = pfLayer->extent_hint.bottom - pfLayer->extent_hint.top;
			const A_long width  = pfLayer->extent_hint.right  - pfLayer->extent_hint.left;
			const A_long line_pitch = pfLayer->rowbytes / static_cast<A_long>(PF_Pixel_BGRA_8u_size);

			for (j = 0; j < height; j++)
			{
				const A_long line_idx = j * line_pitch;

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

							src_idx = j_idx * line_pitch + i_idx;

							accumR += (static_cast<int32_t>(localSrc[src_idx].R) * kernArray[k_idx]);
							accumG += (static_cast<int32_t>(localSrc[src_idx].G) * kernArray[k_idx]);
							accumB += (static_cast<int32_t>(localSrc[src_idx].B) * kernArray[k_idx]);
							
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
					localDst[dst_idx].A = localSrc[dst_idx].A;

				} /* for (i = 0; i < in_data->width; i++) */

			} /* for (j = 0; j < height; j++) */
		
		} /* if (true == iKernel->LoadKernel()) */

	} /* if (nullptr != (iKernel = GetKernel<int32_t>(choosedKernel))) */

	iKernel = nullptr; /* for DBG purpose */

	return;
}


static void
ProcessPrImage_BGRA_4444_16u
(
	PF_InData*   __restrict in_data,
	PF_OutData*  __restrict out_data,
	PF_ParamDef* __restrict params[],
	PF_LayerDef* __restrict output,
	const uint32_t& choosedKernel
) noexcept
{
	IAbsrtactKernel<int32_t>* iKernel = nullptr;
	const PF_LayerDef* __restrict pfLayer = reinterpret_cast<PF_LayerDef* __restrict>(&params[0]->u.ld);
	PF_Pixel_BGRA_16u*  __restrict localSrc = reinterpret_cast<PF_Pixel_BGRA_16u* __restrict>(pfLayer->data);
	PF_Pixel_BGRA_16u*  __restrict localDst = reinterpret_cast<PF_Pixel_BGRA_16u* __restrict>(output->data);

	A_long i = 0, j = 0;
	A_long l = 0, m = 0, k = 0;
	A_long accumR, accumG, accumB, k_idx = 0, j_idx = 0, i_idx = 0, src_idx = 0, dst_idx = 0;

	if (nullptr != (iKernel = GetKernel<int32_t>(choosedKernel)))
	{
		if (true == iKernel->LoadKernel())
		{
			const uint32_t kernSize = iKernel->GetSize();
			const int32_t* __restrict kernArray = iKernel->GetArray();
			const int32_t elements = static_cast<int32_t>(kernSize * kernSize);
			const int32_t halfKernelSize = static_cast<int32_t>(kernSize) >> 1;

			const float kernSum = iKernel->Normalizing();

			const A_long height = pfLayer->extent_hint.bottom - pfLayer->extent_hint.top;
			const A_long width  = pfLayer->extent_hint.right - pfLayer->extent_hint.left;
			const A_long line_pitch = pfLayer->rowbytes / static_cast<A_long>(PF_Pixel_BGRA_16u_size);

			for (j = 0; j < height; j++)
			{
				const A_long line_idx = j * line_pitch;

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

							src_idx = j_idx * line_pitch + i_idx;

							accumR += (static_cast<int32_t>(localSrc[src_idx].R) * kernArray[k_idx]);
							accumG += (static_cast<int32_t>(localSrc[src_idx].G) * kernArray[k_idx]);
							accumB += (static_cast<int32_t>(localSrc[src_idx].B) * kernArray[k_idx]);

							k_idx--;

						} /* for (m = -halfKernelSize; m < halfKernelSize; m++) */

					} /* for (l = -halfKernelSize; l < halfKernelSize; l++) */

					const float outR = static_cast<float>(accumR) / kernSum;
					const float outG = static_cast<float>(accumG) / kernSum;
					const float outB = static_cast<float>(accumB) / kernSum;

					dst_idx = line_idx + i;
					localDst[dst_idx].R = static_cast<A_u_short>(CLAMP_VALUE(outR, 0.f, 32767.0f));
					localDst[dst_idx].G = static_cast<A_u_short>(CLAMP_VALUE(outG, 0.f, 32767.0f));
					localDst[dst_idx].B = static_cast<A_u_short>(CLAMP_VALUE(outB, 0.f, 32767.0f));
					localDst[dst_idx].A = localSrc[dst_idx].A;

				} /* for (i = 0; i < in_data->width; i++) */

			} /* for (j = 0; j < height; j++) */

		} /* if (true == iKernel->LoadKernel()) */

	} /* if (nullptr != (iKernel = GetKernel<int32_t>(choosedKernel))) */

	iKernel = nullptr; /* for DBG purpose */

	return;
}


static void
ProcessPrImage_BGRA_4444_32f
(
	PF_InData*   __restrict in_data,
	PF_OutData*  __restrict out_data,
	PF_ParamDef* __restrict params[],
	PF_LayerDef* __restrict output,
	const uint32_t& choosedKernel
) noexcept
{
	IAbsrtactKernel<float>* iKernel = nullptr;
	const PF_LayerDef* __restrict pfLayer = reinterpret_cast<PF_LayerDef* __restrict>(&params[0]->u.ld);
	PF_Pixel_BGRA_32f*  __restrict localSrc = reinterpret_cast<PF_Pixel_BGRA_32f* __restrict>(pfLayer->data);
	PF_Pixel_BGRA_32f*  __restrict localDst = reinterpret_cast<PF_Pixel_BGRA_32f* __restrict>(output->data);

	A_long i = 0, j = 0;
	A_long l = 0, m = 0, k = 0;
	A_long k_idx = 0, j_idx = 0, i_idx = 0, src_idx = 0, dst_idx = 0;
	float accumR, accumG, accumB;

	if (nullptr != (iKernel = GetKernel<float>(choosedKernel)))
	{
		if (true == iKernel->LoadKernel())
		{
			const uint32_t kernSize = iKernel->GetSize();
			const float* __restrict kernArray = iKernel->GetArray();
			const int32_t elements = static_cast<int32_t>(kernSize * kernSize);
			const int32_t halfKernelSize = static_cast<int32_t>(kernSize) >> 1;

			const float kernSum = iKernel->Normalizing();

			const A_long height = pfLayer->extent_hint.bottom - pfLayer->extent_hint.top;
			const A_long width = pfLayer->extent_hint.right - pfLayer->extent_hint.left;
			const A_long line_pitch = pfLayer->rowbytes / static_cast<A_long>(PF_Pixel_BGRA_32f_size);

			for (j = 0; j < height; j++)
			{
				const A_long line_idx = j * line_pitch;

				for (i = 0; i < width; i++)
				{
					k_idx = elements - 1;
					accumR = accumG = accumB = 0.f;

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

							src_idx = j_idx * line_pitch + i_idx;

							accumR += localSrc[src_idx].R * kernArray[k_idx];
							accumG += localSrc[src_idx].G * kernArray[k_idx];
							accumB += localSrc[src_idx].B * kernArray[k_idx];

							k_idx--;

						} /* for (m = -halfKernelSize; m < halfKernelSize; m++) */

					} /* for (l = -halfKernelSize; l < halfKernelSize; l++) */

					const float outR = static_cast<float>(accumR) / kernSum;
					const float outG = static_cast<float>(accumG) / kernSum;
					const float outB = static_cast<float>(accumB) / kernSum;

					dst_idx = line_idx + i;
					localDst[dst_idx].R = CLAMP_VALUE(outR, f32_value_black, f32_value_white);
					localDst[dst_idx].G = CLAMP_VALUE(outG, f32_value_black, f32_value_white);
					localDst[dst_idx].B = CLAMP_VALUE(outB, f32_value_black, f32_value_white);
					localDst[dst_idx].A = localSrc[dst_idx].A;

				} /* for (i = 0; i < in_data->width; i++) */

			} /* for (j = 0; j < height; j++) */

		} /* if (true == iKernel->LoadKernel()) */

	} /* if (nullptr != (iKernel = GetKernel<int32_t>(choosedKernel))) */

	iKernel = nullptr; /* for DBG purpose */

	return;
}



bool ProcessImgInPR
(
	PF_InData*   __restrict in_data,
	PF_OutData*  __restrict out_data,
	PF_ParamDef* __restrict params[],
	PF_LayerDef* __restrict output,
	const PrPixelFormat& destinationPixelFormat,
	const uint32_t& choosedKernel
) noexcept
{
	bool bValue = true;

		switch (destinationPixelFormat)
		{
			case PrPixelFormat_BGRA_4444_8u:
				ProcessPrImage_BGRA_4444_8u (in_data, out_data, params, output, choosedKernel);
			break;

			case PrPixelFormat_BGRA_4444_16u:
				ProcessPrImage_BGRA_4444_16u (in_data, out_data, params, output, choosedKernel);
			break;

			case PrPixelFormat_VUYA_4444_8u:
			break;

			case PrPixelFormat_VUYA_4444_8u_709:
			break;

			case PrPixelFormat_BGRA_4444_32f:
				ProcessPrImage_BGRA_4444_32f (in_data, out_data, params, output, choosedKernel);
			break;

			case PrPixelFormat_VUYA_4444_32f:
			break;

			case PrPixelFormat_VUYA_4444_32f_709:
			break;

			case PrPixelFormat_RGB_444_10u:
			break;

			default:
				bValue = false;
			break;
		} /* switch (destinationPixelFormat) */

	return bValue;
}

