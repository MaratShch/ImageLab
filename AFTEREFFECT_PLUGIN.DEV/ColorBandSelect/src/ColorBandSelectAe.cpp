#include "ColorBandSelect.hpp"
#include "ColorBandSelectEnums.hpp"
#include "ColorBandSelectProc.hpp"

PF_Err ColorBandSelectInAE_8bits
(
	PF_InData*   in_data,
	PF_OutData*  out_data,
	PF_ParamDef* params[],
	PF_LayerDef* output
) noexcept
{
	PF_EffectWorld*   __restrict input    = reinterpret_cast<PF_EffectWorld* __restrict>(&params[COLOR_BAND_FILTER_INPUT]->u.ld);
	const PF_Pixel_ARGB_8u* __restrict localSrc = reinterpret_cast<const PF_Pixel_ARGB_8u* __restrict>(input->data);
	PF_Pixel_ARGB_8u*       __restrict localDst = reinterpret_cast<PF_Pixel_ARGB_8u* __restrict>(output->data);

	auto const src_pitch = input->rowbytes  / static_cast<A_long>(PF_Pixel_ARGB_8u_size);
	auto const dst_pitch = output->rowbytes / static_cast<A_long>(PF_Pixel_ARGB_8u_size);

	auto const height = output->height;
	auto const width  = output->width;

	const A_long ChannelR = params[COLOR_BAND_CHANNEL_RED]->u.bd.value;
	const A_long ChannelG = params[COLOR_BAND_CHANNEL_GREEN]->u.bd.value;
	const A_long ChannelB = params[COLOR_BAND_CHANNEL_BLUE]->u.bd.value;

	PF_Err err = PF_Err_NONE;

	if (0 != ChannelR && 0 != ChannelG && 0 != ChannelB)
	{
		auto const& worldTransformSuite{ AEFX_SuiteScoper<PF_WorldTransformSuite1>(in_data, kPFWorldTransformSuite, kPFWorldTransformSuiteVersion1, out_data) };
		err = worldTransformSuite->copy (in_data->effect_ref, input, output, NULL, NULL);
	}
	else
		ImgCopyByChannelMask (localSrc, localDst, src_pitch, dst_pitch, width, height, ChannelR, ChannelG, ChannelB);

	return err;
}


PF_Err ColorBandSelectInAE_16bits
(
	PF_InData*   in_data,
	PF_OutData*  out_data,
	PF_ParamDef* params[],
	PF_LayerDef* output
) noexcept
{
	PF_EffectWorld*    __restrict input    = reinterpret_cast<      PF_EffectWorld* __restrict>(&params[COLOR_BAND_FILTER_INPUT]->u.ld);
	const PF_Pixel_ARGB_16u* __restrict localSrc = reinterpret_cast<const PF_Pixel_ARGB_16u* __restrict>(input->data);
	PF_Pixel_ARGB_16u*       __restrict localDst = reinterpret_cast<PF_Pixel_ARGB_16u* __restrict>(output->data);

	auto const src_pitch = input->rowbytes  / static_cast<A_long>(PF_Pixel_ARGB_16u_size);
	auto const dst_pitch = output->rowbytes / static_cast<A_long>(PF_Pixel_ARGB_16u_size);

	auto const height = output->height;
	auto const width  = output->width;

	const A_long ChannelR = params[COLOR_BAND_CHANNEL_RED  ]->u.bd.value;
	const A_long ChannelG = params[COLOR_BAND_CHANNEL_GREEN]->u.bd.value;
	const A_long ChannelB = params[COLOR_BAND_CHANNEL_BLUE ]->u.bd.value;

	PF_Err err = PF_Err_NONE;

	if (0 != ChannelR && 0 != ChannelG && 0 != ChannelB)
	{
		auto const& worldTransformSuite{ AEFX_SuiteScoper<PF_WorldTransformSuite1>(in_data, kPFWorldTransformSuite, kPFWorldTransformSuiteVersion1, out_data) };
		err = worldTransformSuite->copy_hq (in_data->effect_ref, input, output, NULL, NULL);
	}
	else
		ImgCopyByChannelMask (localSrc, localDst, src_pitch, dst_pitch, width, height, ChannelR, ChannelG, ChannelB);

	return err;
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
		ColorBandSelectInAE_16bits(in_data, out_data, params, output) :
		ColorBandSelectInAE_8bits (in_data, out_data, params, output));
}