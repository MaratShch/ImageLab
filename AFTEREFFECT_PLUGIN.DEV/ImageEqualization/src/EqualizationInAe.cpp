#include "ImageEqualization.hpp"


PF_Err
ImageEqualizationInAE_8bits
(
	PF_InData*		in_data,
	PF_OutData*		out_data,
	PF_ParamDef*	params[],
	PF_LayerDef*	output
) noexcept
{
	return PF_Err_NONE;
}

PF_Err
ImageEqualizationInAE_16bits
(
	PF_InData*		in_data,
	PF_OutData*		out_data,
	PF_ParamDef*	params[],
	PF_LayerDef*	output
) noexcept
{
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
		ImageEqualizationInAE_16bits(in_data, out_data, params, output) :
		ImageEqualizationInAE_8bits (in_data, out_data, params, output));
}