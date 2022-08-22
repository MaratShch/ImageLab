#include "NoiseClean.hpp"


inline PF_Err NoiseCleanInAE_8bits
(
	PF_InData*   in_data,
	PF_OutData*  out_data,
	PF_ParamDef* params[],
	PF_LayerDef* output
) noexcept
{
	return NoiseCleanAe_ARGB_4444_8u (in_data, out_data, params, output);
}


inline PF_Err NoiseCleanInAE_16bits
(
	PF_InData*   in_data,
	PF_OutData*  out_data,
	PF_ParamDef* params[],
	PF_LayerDef* output
) noexcept
{
	return NoiseCleanAe_ARGB_4444_16u (in_data, out_data, params, output);
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
		NoiseCleanInAE_16bits(in_data, out_data, params, output) :
		NoiseCleanInAE_8bits(in_data, out_data, params, output));
}