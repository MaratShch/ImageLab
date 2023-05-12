#include "ColorCorrectionCIELab.hpp"
#include "PrSDKAESupport.h"


inline PF_Err ColorCorrectionCieLABInAe_8bits
(
	PF_InData*   in_data,
	PF_OutData*  out_data,
	PF_ParamDef* params[],
	PF_LayerDef* output
) noexcept
{
	PF_Err err = PF_Err_NONE;
	return err;
}


inline PF_Err ColorCorrectionCieLABInAe_16bits
(
	PF_InData*   in_data,
	PF_OutData*  out_data,
	PF_ParamDef* params[],
	PF_LayerDef* output
) noexcept
{
	PF_Err err = PF_Err_NONE;
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
		ColorCorrectionCieLABInAe_16bits(in_data, out_data, params, output) :
		ColorCorrectionCieLABInAe_8bits(in_data, out_data, params, output));
}