#include "ColorTemperature.hpp"


PF_Err ColorTemperature_InAE_8bits
(
	PF_InData*   in_data,
	PF_OutData*  out_data,
	PF_ParamDef* params[],
	PF_LayerDef* output
) noexcept
{
	return PF_Err_NONE;
}

PF_Err ColorTemperature_InAE_16bits
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
		ColorTemperature_InAE_16bits(in_data, out_data, params, output) :
		ColorTemperature_InAE_8bits (in_data, out_data, params, output));
}