#include "ColorSubstitution.hpp"


PF_Err ColorSubstiutution_InAE_8bits
(
	PF_InData*   in_data,
	PF_OutData*  out_data,
	PF_ParamDef* params[],
	PF_LayerDef* output
) noexcept
{
	return PF_Err_NONE;
}

PF_Err ColorSubstiutution_InAE_16bits
(
	PF_InData*   in_data,
	PF_OutData*  out_data,
	PF_ParamDef* params[],
	PF_LayerDef* output
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
        ColorSubstiutution_InAE_16bits(in_data, out_data, params, output) :
        ColorSubstiutution_InAE_8bits (in_data, out_data, params, output));
}