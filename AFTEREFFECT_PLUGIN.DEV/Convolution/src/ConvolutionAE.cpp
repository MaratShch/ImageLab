#include "Convolution.hpp"
#include "Kernels.hpp"

// AE NON COMPLETED! A LOT OF ISSUES!!!!
typedef struct ConvAEParams
{
	IAbsrtactKernel<int32_t>* iKernel;
	IAbsrtactKernel<float>*   fKernel;
	A_long line_pitch;
	A_long height;
	A_long width;
}ConvAEParams;

static uint32_t dbgCnt;


static PF_Err
FilterImage8(
	void*      __restrict refcon,
	A_long		xL,
	A_long		yL,
	PF_Pixel8* __restrict inP,
	PF_Pixel8* __restrict outP) noexcept
{
	ConvAEParams* AeParams = reinterpret_cast<ConvAEParams*>(refcon);
	IAbsrtactKernel<int32_t>* __restrict iKernel = AeParams->iKernel;
	const A_long& line_pitch{ AeParams->line_pitch / static_cast<A_long>(PF_Pixel_ARGB_8u_size) };
	const A_long& height{ AeParams->height };
	const A_long& width { AeParams->width };

	dbgCnt++;

#if 0
	if (nullptr != iKernel)
	{
		const uint32_t kernSize = iKernel->GetSize();
		const int32_t* __restrict kernArray = iKernel->GetArray();
		const int32_t elements = static_cast<int32_t>(kernSize * kernSize);
		const int32_t halfKernelSize = static_cast<int32_t>(kernSize) >> 1;
		A_long i, j;
		A_long accumR = 0, accumG = 0, accumB = 0;
		A_long k_idx = elements - 1, j_idx = 0, i_idx = 0;

		for (j = -halfKernelSize; j <= halfKernelSize; j++)
		{
			j_idx = yL + j;
			if (j_idx < 0 || j_idx >= height)
			{
				k_idx--;
				continue;
			}

			for (i = -halfKernelSize; i <= halfKernelSize; i++)
			{
				i_idx = xL + i;
				if (i_idx < 0 || i_idx >= width)
				{
					k_idx--;
					continue;
				}


				outP->alpha = inP->alpha;
				outP->red = inP->green;
				outP->green = inP->blue;
				outP->blue = inP->red;

					
			} /* for (i = -halfKernelSize; i <= halfKernelSize; i++) */

		} /* for (j = -halfKernelSize; j <= halfKernelSize; j++) */

	} /* if (nullptr != iKernel) */
#endif

	return PF_Err_NONE;
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
#if 0

	ConvAEParams aeParams{
		GetKernel<int32_t>(choosed_kernel),
		nullptr,
		(reinterpret_cast<PF_LayerDef*>(&params[0]->u.ld))->rowbytes,
		linesL,
		pixelsL
	};

	dbgCnt = 0u;

	AEFX_SuiteScoper<PF_Iterate8Suite1> iterate8Suite =
		AEFX_SuiteScoper<PF_Iterate8Suite1>(
			in_data,
			kPFIterate8Suite,
			kPFIterate8SuiteVersion1,
			out_data);

	const PF_Err err = 
		iterate8Suite->iterate(
		in_data,
		0,									// progress base
		linesL,								// progress final
		&params[0]->u.ld,					// src 
		NULL,								// area - null for all pixels
		reinterpret_cast<void*>(&aeParams),	// refcon - your custom data pointer
		FilterImage8,						// pixel function pointer
		output);							// dest

	return (PF_Err_NONE == err ? true : false);
#else

	IAbsrtactKernel<int32_t>* iKernel = nullptr;
	const PF_LayerDef* __restrict pfLayer = reinterpret_cast<const PF_LayerDef* __restrict>(&params[0]->u.ld);
	PF_Pixel_BGRA_8u*  __restrict localSrc = reinterpret_cast<PF_Pixel_BGRA_8u* __restrict>(pfLayer->data);
	PF_Pixel_BGRA_8u*  __restrict localDst = reinterpret_cast<PF_Pixel_BGRA_8u* __restrict>(output->data);

	const A_long height = pfLayer->height;
	const A_long width  = pfLayer->width;
	const A_long line_pitch = pfLayer->rowbytes / static_cast<A_long>(PF_Pixel_BGRA_8u_size);

	A_long i = 0, j = 0;
	A_long l = 0, m = 0, k = 0;

	A_long accumR, accumG, accumB, k_idx = 0, j_idx = 0, i_idx = 0, src_idx = 0, dst_idx = 0;
	A_long line_idx = 0;


	if (nullptr != (iKernel = GetKernel<int32_t>(choosed_kernel)))
	{
		if (true == iKernel->LoadKernel())
		{
			const uint32_t kernSize = iKernel->GetSize();
			const int32_t* __restrict kernArray = iKernel->GetArray();
			const int32_t elements = static_cast<int32_t>(kernSize * kernSize);
			const int32_t halfKernelSize = static_cast<int32_t>(kernSize) >> 1;

			const float kernSum = iKernel->Normalizing();


			for (j = 0; j < height; j++)
			{
				line_idx = j * line_pitch;

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
					localDst[dst_idx].R = static_cast<A_u_char>(CLAMP_VALUE(outR, 0.f, 255.f));
					localDst[dst_idx].G = static_cast<A_u_char>(CLAMP_VALUE(outG, 0.f, 255.f));
					localDst[dst_idx].B = static_cast<A_u_char>(CLAMP_VALUE(outB, 0.f, 255.f));
					localDst[dst_idx].A = localSrc[dst_idx].A;

				} /* for (i = 0; i < width; i++) */

			} /* for (j = 0; j < height; j++) */

		} /* if (true == iKernel->LoadKernel()) */

	} /* if (nullptr != (iKernel = GetKernel<int32_t>(choosed_kernel))) */

	iKernel = nullptr; /* for DBG purpose */

	return true;
#endif
}