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


PF_Err ColorBandSelectInAE_32bits
(
    PF_InData*   in_data,
    PF_OutData*  out_data,
    PF_ParamDef* params[],
    PF_LayerDef* output
) noexcept
{
    PF_EffectWorld*    __restrict input = reinterpret_cast<      PF_EffectWorld* __restrict>(&params[COLOR_BAND_FILTER_INPUT]->u.ld);
    const PF_Pixel_ARGB_32f* __restrict localSrc = reinterpret_cast<const PF_Pixel_ARGB_32f* __restrict>(input->data);
    PF_Pixel_ARGB_32f*       __restrict localDst = reinterpret_cast<PF_Pixel_ARGB_32f* __restrict>(output->data);

    auto const src_pitch = input->rowbytes  / static_cast<A_long>(PF_Pixel_ARGB_32f_size);
    auto const dst_pitch = output->rowbytes / static_cast<A_long>(PF_Pixel_ARGB_32f_size);

    auto const height = output->height;
    auto const width = output->width;

    const A_long ChannelR = params[COLOR_BAND_CHANNEL_RED]->u.bd.value;
    const A_long ChannelG = params[COLOR_BAND_CHANNEL_GREEN]->u.bd.value;
    const A_long ChannelB = params[COLOR_BAND_CHANNEL_BLUE]->u.bd.value;

    PF_Err err = PF_Err_NONE;

    if (0 != ChannelR && 0 != ChannelG && 0 != ChannelB)
    {
        auto const& worldTransformSuite{ AEFX_SuiteScoper<PF_WorldTransformSuite1>(in_data, kPFWorldTransformSuite, kPFWorldTransformSuiteVersion1, out_data) };
        err = worldTransformSuite->copy_hq(in_data->effect_ref, input, output, NULL, NULL);
    }
    else
        ImgCopyByChannelMask (localSrc, localDst, src_pitch, dst_pitch, width, height, ChannelR, ChannelG, ChannelB);

    return err;
}


inline PF_Err ColorBandSelectInAE_DeepWorld
(
    PF_InData*   in_data,
    PF_OutData*  out_data,
    PF_ParamDef* params[],
    PF_LayerDef* output
) noexcept
{
    PF_Err	err = PF_Err_NONE;
    PF_PixelFormat format = PF_PixelFormat_INVALID;
    AEFX_SuiteScoper<PF_WorldSuite2> wsP = AEFX_SuiteScoper<PF_WorldSuite2>(in_data, kPFWorldSuite, kPFWorldSuiteVersion2, out_data);
    if (PF_Err_NONE == wsP->PF_GetPixelFormat(reinterpret_cast<PF_EffectWorld* __restrict>(&params[COLOR_BAND_FILTER_INPUT]->u.ld), &format))
    {
        err = (format == PF_PixelFormat_ARGB128 ?
            ColorBandSelectInAE_32bits(in_data, out_data, params, output) : ColorBandSelectInAE_16bits(in_data, out_data, params, output));
    }
    else
        err = PF_Err_UNRECOGNIZED_PARAM_TYPE;

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
        ColorBandSelectInAE_DeepWorld (in_data, out_data, params, output) :
		ColorBandSelectInAE_8bits (in_data, out_data, params, output));
}