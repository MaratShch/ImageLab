#include "AuthomaticWhiteBalance.hpp"
#include "AlgCommonFunctions.hpp"
#include "AlgCorrectionMatrix.hpp"

PF_Err ProcessImgInAE_8bits
(
	PF_InData*    in_data,
	PF_OutData*   out_data,
	PF_ParamDef*  params[],
	PF_LayerDef*  output
) 
{
    return PF_Err_NONE;
}


PF_Err ProcessImgInAE_16bits
(
	PF_InData*    in_data,
	PF_OutData*   out_data,
	PF_ParamDef*  params[],
	PF_LayerDef*  output
) 
{
    return PF_Err_NONE;
}


PF_Err ProcessImgInAE_32bits
(
    PF_InData*    in_data,
    PF_OutData*   out_data,
    PF_ParamDef*  params[],
    PF_LayerDef*  output
) 
{
    return PF_Err_NONE;
}


inline PF_Err ProcessImgInAE_DeepWord
(
    PF_InData*   in_data,
    PF_OutData*  out_data,
    PF_ParamDef* params[],
    PF_LayerDef* output
) 
{
    PF_Err	err = PF_Err_NONE;
    PF_PixelFormat format = PF_PixelFormat_INVALID;
    AEFX_SuiteScoper<PF_WorldSuite2> wsP = AEFX_SuiteScoper<PF_WorldSuite2>(in_data, kPFWorldSuite, kPFWorldSuiteVersion2, out_data);
    if (PF_Err_NONE == wsP->PF_GetPixelFormat(reinterpret_cast<PF_EffectWorld* __restrict>(&params[AWB_INPUT]->u.ld), &format))
        err = (format == PF_PixelFormat_ARGB128 ?
            ProcessImgInAE_32bits(in_data, out_data, params, output) : ProcessImgInAE_16bits(in_data, out_data, params, output));
    else
        PF_Err_UNRECOGNIZED_PARAM_TYPE;

    return err;
}


PF_Err ProcessImgInAE
(
	PF_InData*    in_data,
	PF_OutData*   out_data,
	PF_ParamDef*  params[],
	PF_LayerDef*  output
) 
{
	return (true == (PF_WORLD_IS_DEEP(output) ?
        ProcessImgInAE_DeepWord (in_data, out_data, params, output) :
		ProcessImgInAE_8bits (in_data, out_data, params, output) ) ? PF_Err_NONE : PF_Err_INVALID_INDEX);
}