#include "BlackAndWhiteProc.hpp"
#include "PrSDKAESupport.h"


PF_Err ProcessImgInAE_8bits
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


PF_Err ProcessImgInAE_16bits
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


PF_Err ProcessImgInAE_32bits
(
    PF_InData*    in_data,
    PF_OutData*   out_data,
    PF_ParamDef*  params[],
    PF_LayerDef*  output
) noexcept
{
    const PF_EffectWorld*   input = reinterpret_cast<const PF_EffectWorld*>(&params[IMAGE_BW_FILTER_INPUT]->u.ld);
    const PF_Pixel_ARGB_32f* __restrict localSrc = reinterpret_cast<const PF_Pixel_ARGB_32f* __restrict>(input->data);
          PF_Pixel_ARGB_32f* __restrict localDst = reinterpret_cast<      PF_Pixel_ARGB_32f* __restrict>(output->data);

    const A_long algoAdvanced = params[IMAGE_BW_ADVANCED_ALGO]->u.bd.value;

    auto const src_pitch = input->rowbytes  / static_cast<A_long>(PF_Pixel_ARGB_32f_size);
    auto const dst_pitch = output->rowbytes / static_cast<A_long>(PF_Pixel_ARGB_32f_size);

    auto const& height = output->height;
    auto const& width = output->width;

    if (0 != algoAdvanced)
        ProcessImageAdvanced(localSrc, localDst, width, height, src_pitch, dst_pitch);
    else
        ProcessImage(localSrc, localDst, width, height, src_pitch, dst_pitch, 0);

    return PF_Err_NONE;
}


inline PF_Err ProcessImgInAE_DeepWorld
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
    if (PF_Err_NONE == wsP->PF_GetPixelFormat(reinterpret_cast<PF_EffectWorld* __restrict>(&params[IMAGE_BW_FILTER_INPUT]->u.ld), &format))
    {
        err = (format == PF_PixelFormat_ARGB128 ?
            ProcessImgInAE_32bits(in_data, out_data, params, output) : ProcessImgInAE_16bits(in_data, out_data, params, output));
    }
    else
        err = PF_Err_UNRECOGNIZED_PARAM_TYPE;

    return err;
}


PF_Err ProcessImgInAE
(
	PF_InData*    in_data,
	PF_OutData*   out_data,
	PF_ParamDef*  params[],
	PF_LayerDef*  output
) noexcept
{
#if !defined __INTEL_COMPILER 
    _MM_SET_FLUSH_ZERO_MODE(_MM_FLUSH_ZERO_ON);
    _MM_SET_DENORMALS_ZERO_MODE(_MM_DENORMALS_ZERO_ON);
#endif

	return (PF_WORLD_IS_DEEP(output) ?
        ProcessImgInAE_DeepWorld(in_data, out_data, params, output) :
		ProcessImgInAE_8bits (in_data, out_data, params, output));
}

