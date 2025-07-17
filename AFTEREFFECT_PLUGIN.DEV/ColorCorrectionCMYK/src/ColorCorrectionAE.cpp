#include "ColorCorrectionCMYK.hpp"
#include "ColorCorrectionEnums.hpp"

PF_Err ProcessImgInAE_8bits
(
	PF_InData*   in_data,
	PF_OutData*  out_data,
	PF_ParamDef* params[],
	PF_LayerDef* output
) noexcept
{
	PF_Err err = PF_Err_NONE;

	/* scan control setting */
	auto const cType   = params[COLOR_CORRECT_SPACE_POPUP]->u.pd.value;
	auto const Coarse1 = params[COLOR_CORRECT_SLIDER1]->u.sd.value;
	auto const Fine1   = params[COLOR_CORRECT_SLIDER2]->u.fs_d.value;
	auto const Coarse2 = params[COLOR_CORRECT_SLIDER3]->u.sd.value;
	auto const Fine2   = params[COLOR_CORRECT_SLIDER4]->u.fs_d.value;
	auto const Coarse3 = params[COLOR_CORRECT_SLIDER5]->u.sd.value;
	auto const Fine3   = params[COLOR_CORRECT_SLIDER6]->u.fs_d.value;
	auto const Coarse4 = params[COLOR_CORRECT_SLIDER7]->u.sd.value;
	auto const Fine4   = params[COLOR_CORRECT_SLIDER8]->u.fs_d.value;

	eCOLOR_SPACE_TYPE const& colorSpaceType = static_cast<eCOLOR_SPACE_TYPE const>(cType - 1);
	float const cVal = static_cast<float>(static_cast<double>(Coarse1) + Fine1);
	float const mVal = static_cast<float>(static_cast<double>(Coarse2) + Fine2);
	float const yVal = static_cast<float>(static_cast<double>(Coarse3) + Fine3);
	float const kVal = static_cast<float>(static_cast<double>(Coarse4) + Fine4);

	switch (colorSpaceType)
	{
		case COLOR_SPACE_CMYK:
			aeProcessImage_ARGB_4444_8u_CMYK(in_data, out_data, params, output, cVal, mVal, yVal, kVal);
		break;
		case COLOR_SPACE_RGB:
			aeProcessImage_ARGB_4444_8u_RGB(in_data, out_data, params, output, cVal, mVal, yVal);
		break;
		default:
			err = PF_Err_INVALID_INDEX;
		break;
	}

	return err;
}


PF_Err ProcessImgInAE_16bits
(
	PF_InData*   in_data,
	PF_OutData*  out_data,
	PF_ParamDef* params[],
	PF_LayerDef* output
) noexcept
{
	PF_Err err = PF_Err_NONE;

	/* scan control setting */
	auto const cType   = params[COLOR_CORRECT_SPACE_POPUP]->u.pd.value;
	auto const Coarse1 = params[COLOR_CORRECT_SLIDER1]->u.sd.value;
	auto const Fine1   = params[COLOR_CORRECT_SLIDER2]->u.fs_d.value;
	auto const Coarse2 = params[COLOR_CORRECT_SLIDER3]->u.sd.value;
	auto const Fine2   = params[COLOR_CORRECT_SLIDER4]->u.fs_d.value;
	auto const Coarse3 = params[COLOR_CORRECT_SLIDER5]->u.sd.value;
	auto const Fine3   = params[COLOR_CORRECT_SLIDER6]->u.fs_d.value;
	auto const Coarse4 = params[COLOR_CORRECT_SLIDER7]->u.sd.value;
	auto const Fine4   = params[COLOR_CORRECT_SLIDER8]->u.fs_d.value;

	eCOLOR_SPACE_TYPE const& colorSpaceType = static_cast<eCOLOR_SPACE_TYPE const>(cType - 1);
	float const cVal = static_cast<float>(static_cast<double>(Coarse1) + Fine1);
	float const mVal = static_cast<float>(static_cast<double>(Coarse2) + Fine2);
	float const yVal = static_cast<float>(static_cast<double>(Coarse3) + Fine3);
	float const kVal = static_cast<float>(static_cast<double>(Coarse4) + Fine4);

	switch (colorSpaceType)
	{
		case COLOR_SPACE_CMYK:
			aeProcessImage_ARGB_4444_16u_CMYK(in_data, out_data, params, output, cVal, mVal, yVal, kVal);
		break;
		case COLOR_SPACE_RGB:
			aeProcessImage_ARGB_4444_16u_RGB(in_data, out_data, params, output, cVal, mVal, yVal);
		break;
		default:
			err = PF_Err_INVALID_INDEX;
		break;
	}

	return err;
}


PF_Err ProcessImgInAE_32bits
(
    PF_InData*   in_data,
    PF_OutData*  out_data,
    PF_ParamDef* params[],
    PF_LayerDef* output
) noexcept
{
    PF_Err err = PF_Err_NONE;

    /* scan control setting */
    auto const cType = params[COLOR_CORRECT_SPACE_POPUP]->u.pd.value;
    auto const Coarse1 = params[COLOR_CORRECT_SLIDER1]->u.sd.value;
    auto const Fine1 = params[COLOR_CORRECT_SLIDER2]->u.fs_d.value;
    auto const Coarse2 = params[COLOR_CORRECT_SLIDER3]->u.sd.value;
    auto const Fine2 = params[COLOR_CORRECT_SLIDER4]->u.fs_d.value;
    auto const Coarse3 = params[COLOR_CORRECT_SLIDER5]->u.sd.value;
    auto const Fine3 = params[COLOR_CORRECT_SLIDER6]->u.fs_d.value;
    auto const Coarse4 = params[COLOR_CORRECT_SLIDER7]->u.sd.value;
    auto const Fine4 = params[COLOR_CORRECT_SLIDER8]->u.fs_d.value;

    eCOLOR_SPACE_TYPE const& colorSpaceType = static_cast<eCOLOR_SPACE_TYPE const>(cType - 1);
    float const cVal = static_cast<float>(static_cast<double>(Coarse1) + Fine1);
    float const mVal = static_cast<float>(static_cast<double>(Coarse2) + Fine2);
    float const yVal = static_cast<float>(static_cast<double>(Coarse3) + Fine3);
    float const kVal = static_cast<float>(static_cast<double>(Coarse4) + Fine4);

    switch (colorSpaceType)
    {
        case COLOR_SPACE_CMYK:
            aeProcessImage_ARGB_4444_32f_CMYK(in_data, out_data, params, output, cVal, mVal, yVal, kVal);
        break;
        case COLOR_SPACE_RGB:
            aeProcessImage_ARGB_4444_32f_RGB(in_data, out_data, params, output, cVal, mVal, yVal);
        break;
        default:
            err = PF_Err_INVALID_INDEX;
        break;
    }

    return err;
}


inline PF_Err ProcessImgInAE_DeepWord
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
    if (PF_Err_NONE == wsP->PF_GetPixelFormat(reinterpret_cast<PF_EffectWorld* __restrict>(&params[COLOR_CORRECT_INPUT]->u.ld), &format))
        err = (format == PF_PixelFormat_ARGB128 ?
            ProcessImgInAE_32bits(in_data, out_data, params, output) : ProcessImgInAE_16bits(in_data, out_data, params, output));
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
        ProcessImgInAE_DeepWord (in_data, out_data, params, output) :
		ProcessImgInAE_8bits (in_data, out_data, params, output));
}