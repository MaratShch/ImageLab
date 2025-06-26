#include "ImageGeometry.hpp"
#include "ImageGeometryEnum.hpp"


PF_Err ImageGeometry_InAE_8bits
(
	PF_InData*   in_data,
	PF_OutData*  out_data,
	PF_ParamDef* params[],
	PF_LayerDef* output
) noexcept
{
	return PF_Err_NONE;
}

PF_Err ImageGeometry_InAE_16bits
(
	PF_InData*   in_data,
	PF_OutData*  out_data,
	PF_ParamDef* params[],
	PF_LayerDef* output
) noexcept
{
	return PF_Err_NONE;
}

PF_Err ImageGeometry_InAE_32bits
(
    PF_InData*   in_data,
    PF_OutData*  out_data,
    PF_ParamDef* params[],
    PF_LayerDef* output
) noexcept
{
    return PF_Err_NONE;
}


inline PF_Err ImageSharpening_InAE_DeepWorld
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
    if (PF_Err_NONE == wsP->PF_GetPixelFormat(reinterpret_cast<PF_EffectWorld* __restrict>(&params[eIMAGE_GEOMETRY_INPUT]->u.ld), &format))
    {
        err = (format == PF_PixelFormat_ARGB128 ?
            ImageGeometry_InAE_32bits(in_data, out_data, params, output) : ImageGeometry_InAE_16bits(in_data, out_data, params, output));
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
		ImageGeometry_InAE_16bits(in_data, out_data, params, output) :
		ImageGeometry_InAE_8bits (in_data, out_data, params, output));
}