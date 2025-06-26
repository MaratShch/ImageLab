#include "PerceptualColorCorrection.hpp"
#include "PerceptualColorCorrectionEnum.hpp"


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
RenderInAfterEffect_32bits
(
    PF_InData*		in_data,
    PF_OutData*		out_data,
    PF_ParamDef*	params[],
    PF_LayerDef*	output
) noexcept
{
    return PF_Err_NONE;
}


inline PF_Err RenderInAfterEffect_DeepWorld
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
    if (PF_Err_NONE == wsP->PF_GetPixelFormat(reinterpret_cast<PF_EffectWorld* __restrict>(&params[PERCOLOR_CORRECTION_INPUT]->u.ld), &format))
    {
        err = (format == PF_PixelFormat_ARGB128 ?
            RenderInAfterEffect_32bits(in_data, out_data, params, output) : RenderInAfterEffect_16bits(in_data, out_data, params, output));
    }
    else
        err = PF_Err_UNRECOGNIZED_PARAM_TYPE;

    return err;
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
        RenderInAfterEffect_DeepWorld (in_data, out_data, params, output) :
		RenderInAfterEffect_8bits (in_data, out_data, params, output));
}