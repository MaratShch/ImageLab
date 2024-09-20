#include "AverageFilter.hpp"
#include "AverageFilterAlgo.hpp"
#include "AverageFilterEnum.hpp"


PF_Err AverageFilter_InAE_8bits
(
	PF_InData*   in_data,
	PF_OutData*  out_data,
	PF_ParamDef* params[],
	PF_LayerDef* output
) noexcept
{
	const PF_EffectWorld*   __restrict input = reinterpret_cast<const PF_EffectWorld* __restrict>(&params[eAEVRAGE_FILTER_INPUT]->u.ld);
	const PF_Pixel_ARGB_8u* __restrict localSrc = reinterpret_cast<const PF_Pixel_ARGB_8u* __restrict>(input->data);
	      PF_Pixel_ARGB_8u* __restrict localDst = reinterpret_cast<      PF_Pixel_ARGB_8u* __restrict>(output->data);

	/* check "Large Window Size" from checkbox */
	const A_long windowSize = ((0u != params[eAEVRAGE_FILTER_LARGE_WINDOW]->u.bd.value) ? 5 : 3);

	auto const src_pitch = input->rowbytes  / static_cast<A_long>(PF_Pixel_ARGB_8u_size);
	auto const dst_pitch = output->rowbytes / static_cast<A_long>(PF_Pixel_ARGB_8u_size);
	auto const sizeY = output->height;
	auto const sizeX = output->width;

	AverageFilterAlgo (localSrc, localDst, sizeX, sizeY, src_pitch, dst_pitch, windowSize);

	return PF_Err_NONE;
}


PF_Err AverageFilter_InAE_16bits
(
	PF_InData*   in_data,
	PF_OutData*  out_data,
	PF_ParamDef* params[],
	PF_LayerDef* output
) noexcept
{
	const PF_EffectWorld*    __restrict input = reinterpret_cast<const PF_EffectWorld* __restrict>(&params[eAEVRAGE_FILTER_INPUT]->u.ld);
	const PF_Pixel_ARGB_16u* __restrict localSrc = reinterpret_cast<const PF_Pixel_ARGB_16u* __restrict>(input->data);
	      PF_Pixel_ARGB_16u* __restrict localDst = reinterpret_cast<      PF_Pixel_ARGB_16u* __restrict>(output->data);

	/* check "Large Window Size" from checkbox */
	const A_long windowSize = ((0u != params[eAEVRAGE_FILTER_LARGE_WINDOW]->u.bd.value) ? 5 : 3);

	auto const src_pitch = input->rowbytes  / static_cast<A_long>(PF_Pixel_ARGB_16u_size);
	auto const dst_pitch = output->rowbytes / static_cast<A_long>(PF_Pixel_ARGB_16u_size);
	auto const sizeY = output->height;
	auto const sizeX = output->width;

	AverageFilterAlgo (localSrc, localDst, sizeX, sizeY, src_pitch, dst_pitch, windowSize);

	return PF_Err_NONE;
}


PF_Err
ProcessImgInAE
(
	PF_InData*		in_data,
	PF_OutData*		out_data,
	PF_ParamDef*	params[],
	PF_LayerDef*	output
) noexcept
{
	return (PF_WORLD_IS_DEEP(output) ?
		AverageFilter_InAE_16bits(in_data, out_data, params, output) :
		AverageFilter_InAE_8bits (in_data, out_data, params, output));
}