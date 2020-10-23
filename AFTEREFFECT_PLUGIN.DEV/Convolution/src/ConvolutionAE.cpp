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
	IAbsrtactKernel<int32_t>* iKernel = nullptr;
	const PF_LayerDef*      __restrict pfLayer = reinterpret_cast<PF_LayerDef* __restrict>(&params[0]->u.ld);
	const PF_Pixel_ARGB_8u* __restrict localSrc = reinterpret_cast<const PF_Pixel_ARGB_8u* __restrict>(pfLayer->data);
	PF_Pixel_ARGB_8u*       __restrict localDst = reinterpret_cast<PF_Pixel_ARGB_8u* __restrict>(output->data);

	if (nullptr != (iKernel = GetKernel<int32_t>(choosed_kernel)))
	{
		if (true == iKernel->LoadKernel())
		{
			const uint32_t kernSize = iKernel->GetSize();
			const int32_t* __restrict kernArray = iKernel->GetArray();
			const int32_t elements = static_cast<int32_t>(kernSize * kernSize);
			const int32_t halfKernelSize = static_cast<int32_t>(kernSize) >> 1;

			const float kernSum = iKernel->Normalizing();

			const A_long height = pfLayer->extent_hint.bottom - pfLayer->extent_hint.top;
			const A_long width = pfLayer->extent_hint.right - pfLayer->extent_hint.left;
			const A_long line_pitch = pfLayer->rowbytes / static_cast<A_long>(PF_Pixel_ARGB_8u_size);


		} /* if (true == iKernel->LoadKernel()) */
	} /* if (nullptr != (iKernel = GetKernel<int32_t>(choosed_kernel))) */

	return true;
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
	IAbsrtactKernel<int32_t>* iKernel = nullptr;
	const PF_LayerDef*       __restrict pfLayer = reinterpret_cast<PF_LayerDef* __restrict>(&params[0]->u.ld);
	const PF_Pixel_ARGB_16u* __restrict localSrc = reinterpret_cast<const PF_Pixel_ARGB_16u* __restrict>(pfLayer->data);
	PF_Pixel_ARGB_16u*       __restrict localDst = reinterpret_cast<PF_Pixel_ARGB_16u* __restrict>(output->data);

	if (nullptr != (iKernel = GetKernel<int32_t>(choosed_kernel)))
	{
		if (true == iKernel->LoadKernel())
		{
			const uint32_t kernSize = iKernel->GetSize();
			const int32_t* __restrict kernArray = iKernel->GetArray();
			const int32_t elements = static_cast<int32_t>(kernSize * kernSize);
			const int32_t halfKernelSize = static_cast<int32_t>(kernSize) >> 1;

			const float kernSum = iKernel->Normalizing();

			const A_long height = pfLayer->extent_hint.bottom - pfLayer->extent_hint.top;
			const A_long width = pfLayer->extent_hint.right - pfLayer->extent_hint.left;
			const A_long line_pitch = pfLayer->rowbytes / static_cast<A_long>(PF_Pixel_ARGB_16u_size);


		} /* if (true == iKernel->LoadKernel()) */
	} /* if (nullptr != (iKernel = GetKernel<int32_t>(choosed_kernel))) */

	return true;
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
		ProcessImgInAE_8bits (in_data, out_data, params, output, choosed_kernel) );
}