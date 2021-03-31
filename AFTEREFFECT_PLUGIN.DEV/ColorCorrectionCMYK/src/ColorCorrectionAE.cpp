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
	auto const& cType   = params[COLOR_CORRECT_SPACE_POPUP]->u.pd.value;
	auto const& Coarse1 = params[COLOR_CORRECT_SLIDER1]->u.sd.value;
	auto const& Fine1   = params[COLOR_CORRECT_SLIDER2]->u.fs_d.value;
	auto const& Coarse2 = params[COLOR_CORRECT_SLIDER3]->u.sd.value;
	auto const& Fine2   = params[COLOR_CORRECT_SLIDER4]->u.fs_d.value;
	auto const& Coarse3 = params[COLOR_CORRECT_SLIDER5]->u.sd.value;
	auto const& Fine3   = params[COLOR_CORRECT_SLIDER6]->u.fs_d.value;
	auto const& Coarse4 = params[COLOR_CORRECT_SLIDER7]->u.sd.value;
	auto const& Fine4   = params[COLOR_CORRECT_SLIDER7]->u.fs_d.value;

	eCOLOR_SPACE_TYPE const& colorSpaceType = static_cast<eCOLOR_SPACE_TYPE const>(cType - 1);

	switch (cType)
	{
		case COLOR_SPACE_CMYK:
		break;
		case COLOR_SPACE_RGB:
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
	auto const& cType = params[COLOR_CORRECT_SPACE_POPUP]->u.pd.value;
	auto const& Coarse1 = params[COLOR_CORRECT_SLIDER1]->u.sd.value;
	auto const& Fine1 = params[COLOR_CORRECT_SLIDER2]->u.fs_d.value;
	auto const& Coarse2 = params[COLOR_CORRECT_SLIDER3]->u.sd.value;
	auto const& Fine2 = params[COLOR_CORRECT_SLIDER4]->u.fs_d.value;
	auto const& Coarse3 = params[COLOR_CORRECT_SLIDER5]->u.sd.value;
	auto const& Fine3 = params[COLOR_CORRECT_SLIDER6]->u.fs_d.value;
	auto const& Coarse4 = params[COLOR_CORRECT_SLIDER7]->u.sd.value;
	auto const& Fine4 = params[COLOR_CORRECT_SLIDER7]->u.fs_d.value;

	eCOLOR_SPACE_TYPE const& colorSpaceType = static_cast<eCOLOR_SPACE_TYPE const>(cType - 1);

	switch (cType)
	{
		case COLOR_SPACE_CMYK:
		break;
		case COLOR_SPACE_RGB:
		break;
		default:
		err = PF_Err_INVALID_INDEX;
		break;
	}

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
		ProcessImgInAE_16bits(in_data, out_data, params, output) :
		ProcessImgInAE_8bits(in_data, out_data, params, output));
}