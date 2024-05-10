#include "PerceptualColorCorrection.hpp"

PF_Err
RenderInAfterEffect_8bits
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
RenderInAfterEffect_16bits
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
RenderInAfterEffect
(
	PF_InData*		in_data,
	PF_OutData*		out_data,
	PF_ParamDef*	params[],
	PF_LayerDef*	output
)
{
	return (PF_WORLD_IS_DEEP(output) ?
		RenderInAfterEffect_16bits(in_data, out_data, params, output) :
		RenderInAfterEffect_8bits (in_data, out_data, params, output));
}