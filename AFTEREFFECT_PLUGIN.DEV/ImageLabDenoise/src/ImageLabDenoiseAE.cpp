#include "ImageLabDenoise.hpp"
#include "ImageLabDenoiseEnum.hpp"


PF_Err ImageDenoise_InAE_8bits
(
	PF_InData*   in_data,
	PF_OutData*  out_data,
	PF_ParamDef* params[],
	PF_LayerDef* output
) 
{
	return PF_Err_NONE;
}

PF_Err  ImageDenoise_InAE_16bits
(
	PF_InData*   in_data,
	PF_OutData*  out_data,
	PF_ParamDef* params[],
	PF_LayerDef* output
) 
{
	return PF_Err_NONE;
}

PF_Err ImageDenoise_InAE_32bits
(
    PF_InData*   in_data,
    PF_OutData*  out_data,
    PF_ParamDef* params[],
    PF_LayerDef* output
) 
{
    return PF_Err_NONE;
}


inline PF_Err ImageDenoise_InAE_DeepWorld
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
    if (PF_Err_NONE == wsP->PF_GetPixelFormat(reinterpret_cast<PF_EffectWorld* __restrict>(&params[UnderlyingType(DenoiseControl::eIMAGE_LAB_DENOISE_INPUT)]->u.ld), &format))
    {
        err = (format == PF_PixelFormat_ARGB128 ?
            ImageDenoise_InAE_32bits (in_data, out_data, params, output) : ImageDenoise_InAE_16bits (in_data, out_data, params, output));
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
) 
{
	return (PF_WORLD_IS_DEEP(output) ?
        ImageDenoise_InAE_DeepWorld (in_data, out_data, params, output) :
        ImageDenoise_InAE_8bits (in_data, out_data, params, output));
}