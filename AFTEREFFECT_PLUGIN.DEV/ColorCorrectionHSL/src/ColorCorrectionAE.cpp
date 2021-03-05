#include "ColorCorrectionHSL.hpp"

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
	auto const& lwbType = params[COLOR_CORRECT_SPACE_POPUP]->u.pd.value;
	auto const& hueCoarse = params[COLOR_CORRECT_HUE_COARSE_LEVEL]->u.ad.value;
	auto const& hueFine = params[COLOR_HUE_FINE_LEVEL_SLIDER]->u.fs_d.value;
	auto const& satCoarse = params[COLOR_SATURATION_COARSE_LEVEL_SLIDER]->u.sd.value;
	auto const& satFine = params[COLOR_SATURATION_FINE_LEVEL_SLIDER]->u.fs_d.value;
	auto const& lwbCoarse = params[COLOR_LWIP_COARSE_LEVEL_SLIDER]->u.sd.value;
	auto const& lwbFine = params[COLOR_LWIP_FINE_LEVEL_SLIDER]->u.fs_d.value;

	float const& totalHue = normalize_hue_wheel(static_cast<float>(hueCoarse) / 65536.f + hueFine);
	float const& totalSat = static_cast<float>(satCoarse) + satFine;
	float const& totalLwb = static_cast<float>(lwbCoarse) + lwbFine;

	eCOLOR_SPACE_TYPE const& colorSpaceType = static_cast<eCOLOR_SPACE_TYPE const>(lwbType - 1);

	switch (colorSpaceType)
	{
		case COLOR_SPACE_HSL:
			err = prProcessImage_ARGB_4444_8u_HSL(in_data, out_data, params, output, totalHue, totalSat, totalLwb);
		break;
		case COLOR_SPACE_HSV:
			err = prProcessImage_ARGB_4444_8u_HSV(in_data, out_data, params, output, totalHue, totalSat, totalLwb);
		break;
		case COLOR_SPACE_HSI:
			err = prProcessImage_ARGB_4444_8u_HSI(in_data, out_data, params, output, totalHue, totalSat, totalLwb);
		break;
		case COLOR_SPACE_HSP:
			err = prProcessImage_ARGB_4444_8u_HSP(in_data, out_data, params, output, totalHue, totalSat, totalLwb);
		break;
		case COLOR_SPACE_HSLuv:
			err = prProcessImage_ARGB_4444_8u_HSLuv(in_data, out_data, params, output, totalHue, totalSat, totalLwb);
		break;
		case COLOR_SPACE_HPLuv:
			err = prProcessImage_ARGB_4444_8u_HPLuv(in_data, out_data, params, output, totalHue, totalSat, totalLwb);
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
	auto const& lwbType = params[COLOR_CORRECT_SPACE_POPUP]->u.pd.value;
	auto const& hueCoarse = params[COLOR_CORRECT_HUE_COARSE_LEVEL]->u.ad.value;
	auto const& hueFine = params[COLOR_HUE_FINE_LEVEL_SLIDER]->u.fs_d.value;
	auto const& satCoarse = params[COLOR_SATURATION_COARSE_LEVEL_SLIDER]->u.sd.value;
	auto const& satFine = params[COLOR_SATURATION_FINE_LEVEL_SLIDER]->u.fs_d.value;
	auto const& lwbCoarse = params[COLOR_LWIP_COARSE_LEVEL_SLIDER]->u.sd.value;
	auto const& lwbFine = params[COLOR_LWIP_FINE_LEVEL_SLIDER]->u.fs_d.value;

	float const& totalHue = normalize_hue_wheel(static_cast<float>(hueCoarse) / 65536.f + hueFine);
	float const& totalSat = static_cast<float>(satCoarse) + satFine;
	float const& totalLwb = static_cast<float>(lwbCoarse) + lwbFine;

	eCOLOR_SPACE_TYPE const& colorSpaceType = static_cast<eCOLOR_SPACE_TYPE const>(lwbType - 1);

	switch (colorSpaceType)
	{
		case COLOR_SPACE_HSL:
			err = prProcessImage_ARGB_4444_16u_HSL(in_data, out_data, params, output, totalHue, totalSat, totalLwb);
		break;
		case COLOR_SPACE_HSV:
			err = prProcessImage_ARGB_4444_16u_HSV(in_data, out_data, params, output, totalHue, totalSat, totalLwb);
		break;
		case COLOR_SPACE_HSI:
			err = prProcessImage_ARGB_4444_16u_HSI(in_data, out_data, params, output, totalHue, totalSat, totalLwb);
		break;
		case COLOR_SPACE_HSP:
			err = prProcessImage_ARGB_4444_16u_HSP(in_data, out_data, params, output, totalHue, totalSat, totalLwb);
		break;
		case COLOR_SPACE_HSLuv:
			err = prProcessImage_ARGB_4444_16u_HSLuv(in_data, out_data, params, output, totalHue, totalSat, totalLwb);
		break;
		case COLOR_SPACE_HPLuv:
			err = prProcessImage_ARGB_4444_16u_HPLuv(in_data, out_data, params, output, totalHue, totalSat, totalLwb);
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