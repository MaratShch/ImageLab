#include "BlackAndWhiteProc.hpp"
#include "PrSDKAESupport.h"


static PF_Err ProcessImgInAE_8bits
(
	PF_InData*    in_data,
	PF_OutData*   out_data,
	PF_ParamDef*  params[],
	PF_LayerDef*  output
) noexcept
{
	const PF_EffectWorld*   input    = reinterpret_cast<const PF_EffectWorld*>(&params[IMAGE_BW_FILTER_INPUT]->u.ld);
	const PF_Pixel_ARGB_8u* __restrict localSrc = reinterpret_cast<const PF_Pixel_ARGB_8u* __restrict>(input->data);
	      PF_Pixel_ARGB_8u* __restrict localDst = reinterpret_cast<      PF_Pixel_ARGB_8u* __restrict>(output->data);

	const A_long algoAdvanced = params[IMAGE_BW_ADVANCED_ALGO]->u.bd.value;
	
	auto const src_pitch = input->rowbytes  / static_cast<A_long>(PF_Pixel_ARGB_8u_size);
	auto const dst_pitch = output->rowbytes / static_cast<A_long>(PF_Pixel_ARGB_8u_size);

	auto const& height = output->height;
	auto const& width  = output->width;

	if (0 != algoAdvanced)
		ProcessImageAdvanced (localSrc, localDst, width, height, src_pitch, dst_pitch);
	else
		ProcessImage (localSrc, localDst, width, height, src_pitch, dst_pitch, 0);

	return PF_Err_NONE;
}


static PF_Err ProcessImgInAE_16bits
(
	PF_InData*    in_data,
	PF_OutData*   out_data,
	PF_ParamDef*  params[],
	PF_LayerDef*  output
) noexcept
{
	const PF_EffectWorld*   input = reinterpret_cast<const PF_EffectWorld*>(&params[IMAGE_BW_FILTER_INPUT]->u.ld);
	const PF_Pixel_ARGB_16u* __restrict localSrc = reinterpret_cast<const PF_Pixel_ARGB_16u* __restrict>(input->data);
	      PF_Pixel_ARGB_16u* __restrict localDst = reinterpret_cast<      PF_Pixel_ARGB_16u* __restrict>(output->data);

    const A_long algoAdvanced = params[IMAGE_BW_ADVANCED_ALGO]->u.bd.value;

	auto const src_pitch = input->rowbytes  / static_cast<A_long>(PF_Pixel_ARGB_16u_size);
	auto const dst_pitch = output->rowbytes / static_cast<A_long>(PF_Pixel_ARGB_16u_size);

	auto const& height = output->height;
	auto const& width  = output->width;

	if (0 != algoAdvanced)
		ProcessImageAdvanced (localSrc, localDst, width, height, src_pitch, dst_pitch);
	else
		ProcessImage (localSrc, localDst, width, height, src_pitch, dst_pitch, 0);

	return PF_Err_NONE;
}


PF_Err ProcessImgInAE
(
	PF_InData*    in_data,
	PF_OutData*   out_data,
	PF_ParamDef*  params[],
	PF_LayerDef*  output
) noexcept
{
	return (PF_WORLD_IS_DEEP(output) ?
		ProcessImgInAE_16bits(in_data, out_data, params, output) :
		ProcessImgInAE_8bits (in_data, out_data, params, output));
}

