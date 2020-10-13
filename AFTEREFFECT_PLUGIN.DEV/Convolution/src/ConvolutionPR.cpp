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
	A_long i = 0, j = 0;

	const PF_ParamValue convKernelType{ params[KERNEL_CHECKBOX]->u.pd.value };
	const uint32_t choosedKernel = static_cast<uint32_t>(convKernelType);
	IAbsrtactKernel<int32_t>* iKernel = GetKernel<int32_t>(choosedKernel - 1);

	if (nullptr != iKernel)
	{
		if (true == iKernel->LoadKernel())
		{
			const int32_t* kernArray = iKernel->GetArray();
			const uint32_t kernSize  = iKernel->GetSize();

			const PF_LayerDef* __restrict pfLayer = reinterpret_cast<PF_LayerDef* __restrict>(&params[0]->u.ld);
			PF_Pixel_BGRA_8u* __restrict localSrc = reinterpret_cast<PF_Pixel_BGRA_8u* __restrict>(pfLayer->data);
			PF_Pixel_BGRA_8u* __restrict localDst = reinterpret_cast<PF_Pixel_BGRA_8u* __restrict>(output->data);

			const A_long height = output->extent_hint.bottom - output->extent_hint.top;
			const A_long width = in_data->width;
			const A_long nextLine = (pfLayer->rowbytes - in_data->width * sizeof(PF_Pixel_BGRA_8u)) >> 2;

			for (j = 0; j < height; j++)
			{
				for (i = 0; i < in_data->width; i++)
				{
					localDst->A = localSrc->A;
					localDst->B = localSrc->R;
					localDst->G = localSrc->G;
					localDst->R = localSrc->B;

					localSrc++;
					localDst++;
				} /* for (i = 0; i < in_data->width; i++) */

				localSrc += nextLine;
				localDst += nextLine;

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