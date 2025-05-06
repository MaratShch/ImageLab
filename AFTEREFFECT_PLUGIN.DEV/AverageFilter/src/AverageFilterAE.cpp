#include "AverageFilter.hpp"
#include "AverageAFilterAlgo.hpp"
#include "AverageGFilterAlgo.hpp"
#include "AverageFilterEnum.hpp"


PF_Err AverageFilter_InAE_8bits
(
	PF_InData*   in_data,
	PF_OutData*  out_data,
	PF_ParamDef* params[],
	PF_LayerDef* output
) noexcept
{
	const PF_EffectWorld*   __restrict input    = reinterpret_cast<const PF_EffectWorld*   __restrict>(&params[eAEVRAGE_FILTER_INPUT]->u.ld);
	const PF_Pixel_ARGB_8u* __restrict localSrc = reinterpret_cast<const PF_Pixel_ARGB_8u* __restrict>(input->data);
	      PF_Pixel_ARGB_8u* __restrict localDst = reinterpret_cast<      PF_Pixel_ARGB_8u* __restrict>(output->data);

    // check average typ//
    const A_long isGeometric = params[eAVERAGE_FILTER_GEOMETRIC_AVERAGE]->u.bd.value;

	// check "Window Size" from popup
    eAVERAGE_FILTER_WINDOW_SIZE const windowSizeEnum{ static_cast<const eAVERAGE_FILTER_WINDOW_SIZE>(params[eAEVRAGE_FILTER_WINDOW_SIZE]->u.pd.value - 1) };
	const A_long windowSize = WindowSizeEnum2Value(windowSizeEnum);
	if (windowSize <= 0) // normally this comparison should be always false/
		return PF_Err_INVALID_INDEX;

	auto const src_pitch = input->rowbytes  / static_cast<A_long>(PF_Pixel_ARGB_8u_size);
	auto const dst_pitch = output->rowbytes / static_cast<A_long>(PF_Pixel_ARGB_8u_size);
	auto const sizeY = output->height;
	auto const sizeX = output->width;

	const A_long err = ((0 == isGeometric) ?
		AverageFilterAlgo(localSrc, localDst, sizeX, sizeY, src_pitch, dst_pitch, windowSize) :
		GeomethricAverageFilterAlgo(localSrc, localDst, sizeX, sizeY, src_pitch, dst_pitch, windowSize));

	return err;
}


PF_Err AverageFilter_InAE_16bits
(
	PF_InData*   in_data,
	PF_OutData*  out_data,
	PF_ParamDef* params[],
	PF_LayerDef* output
) noexcept
{
	const PF_EffectWorld*    __restrict input    = reinterpret_cast<const PF_EffectWorld*    __restrict>(&params[eAEVRAGE_FILTER_INPUT]->u.ld);
	const PF_Pixel_ARGB_16u* __restrict localSrc = reinterpret_cast<const PF_Pixel_ARGB_16u* __restrict>(input->data);
	      PF_Pixel_ARGB_16u* __restrict localDst = reinterpret_cast<      PF_Pixel_ARGB_16u* __restrict>(output->data);

    const A_long isGeometric = params[eAVERAGE_FILTER_GEOMETRIC_AVERAGE]->u.bd.value;

	// check "Window Size" from popup
	eAVERAGE_FILTER_WINDOW_SIZE const windowSizeEnum{ static_cast<const eAVERAGE_FILTER_WINDOW_SIZE>(params[eAEVRAGE_FILTER_WINDOW_SIZE]->u.pd.value - 1) };
	const A_long windowSize = WindowSizeEnum2Value(windowSizeEnum);
	if (windowSize <= 0) // normally this comparison should be always false
	    return PF_Err_INVALID_INDEX;

	auto const src_pitch = input->rowbytes  / static_cast<A_long>(PF_Pixel_ARGB_16u_size);
	auto const dst_pitch = output->rowbytes / static_cast<A_long>(PF_Pixel_ARGB_16u_size);
	auto const sizeY = output->height;
	auto const sizeX = output->width;

	const A_long err = ((0 == isGeometric) ?
		AverageFilterAlgo(localSrc, localDst, sizeX, sizeY, src_pitch, dst_pitch, windowSize) :
		GeomethricAverageFilterAlgo(localSrc, localDst, sizeX, sizeY, src_pitch, dst_pitch, windowSize));

	return err;
}


PF_Err AverageFilter_InAE_32bits
(
    PF_InData*   in_data,
    PF_OutData*  out_data,
    PF_ParamDef* params[],
    PF_LayerDef* output
) noexcept
{
    const PF_EffectWorld*    __restrict input    = reinterpret_cast<const PF_EffectWorld*    __restrict>(&params[eAEVRAGE_FILTER_INPUT]->u.ld);
    const PF_Pixel_ARGB_32f* __restrict localSrc = reinterpret_cast<const PF_Pixel_ARGB_32f* __restrict>(input->data);
          PF_Pixel_ARGB_32f* __restrict localDst = reinterpret_cast<      PF_Pixel_ARGB_32f* __restrict>(output->data);

    const A_long isGeometric = params[eAVERAGE_FILTER_GEOMETRIC_AVERAGE]->u.bd.value;

    // check "Window Size" from popup
    eAVERAGE_FILTER_WINDOW_SIZE const windowSizeEnum{ static_cast<const eAVERAGE_FILTER_WINDOW_SIZE>(params[eAEVRAGE_FILTER_WINDOW_SIZE]->u.pd.value - 1) };
    const A_long windowSize = WindowSizeEnum2Value(windowSizeEnum);
    if (windowSize <= 0) // normally this comparison should be always false
        return PF_Err_INVALID_INDEX;

    auto const src_pitch = input->rowbytes  / static_cast<A_long>(PF_Pixel_ARGB_32f_size);
    auto const dst_pitch = output->rowbytes / static_cast<A_long>(PF_Pixel_ARGB_32f_size);
    auto const sizeY = output->height;
    auto const sizeX = output->width;

    const A_long err = ((0 == isGeometric) ?
        AverageFilterAlgo(localSrc, localDst, sizeX, sizeY, src_pitch, dst_pitch, windowSize) :
        GeomethricAverageFilterAlgo(localSrc, localDst, sizeX, sizeY, src_pitch, dst_pitch, windowSize));

    return err;
}


inline PF_Err AverageFilter_InAE_DeepWorld
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
    if (PF_Err_NONE == wsP->PF_GetPixelFormat(reinterpret_cast<PF_EffectWorld* __restrict>(&params[eAEVRAGE_FILTER_INPUT]->u.ld), &format))
    {
        err = (format == PF_PixelFormat_ARGB128 ?
            AverageFilter_InAE_32bits(in_data, out_data, params, output) : AverageFilter_InAE_16bits(in_data, out_data, params, output));
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
        AverageFilter_InAE_DeepWorld(in_data, out_data, params, output) :
		AverageFilter_InAE_8bits (in_data, out_data, params, output));
}