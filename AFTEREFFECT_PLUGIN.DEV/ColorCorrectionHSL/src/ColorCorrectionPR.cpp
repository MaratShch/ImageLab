#include "ColorCorrectionHSL.hpp"
#include "PrSDKAESupport.h"


PF_Err
ProcessImgInPR
(
	PF_InData*		in_data,
	PF_OutData*		out_data,
	PF_ParamDef*	params[],
	PF_LayerDef*	output
) noexcept
{
	PF_Err err = PF_Err_NONE;
	PF_Err errFormat = PF_Err_INVALID_INDEX;

	/* scan control setting */
	auto const& lwbType   = params[COLOR_CORRECT_SPACE_POPUP]->u.pd.value;
	auto const& hueCoarse = params[COLOR_CORRECT_HUE_COARSE_LEVEL]->u.ad.value;
	auto const& hueFine   = params[COLOR_HUE_FINE_LEVEL_SLIDER]->u.fs_d.value;
	auto const& satCoarse = params[COLOR_SATURATION_COARSE_LEVEL_SLIDER]->u.sd.value;
	auto const& satFine   = params[COLOR_SATURATION_FINE_LEVEL_SLIDER]->u.fs_d.value;
	auto const& lwbCoarse = params[COLOR_LWIP_COARSE_LEVEL_SLIDER]->u.sd.value;
	auto const& lwbFine   = params[COLOR_LWIP_FINE_LEVEL_SLIDER]->u.fs_d.value;

	float const& totalHue = normalize_hue_wheel(static_cast<float>(hueCoarse) / 65536.f + hueFine);
	float const& totalSat = static_cast<float>(satCoarse) + satFine;
	float const& totalLwb = static_cast<float>(lwbCoarse) + lwbFine;

	eCOLOR_SPACE_TYPE const& colorSpaceType = static_cast<eCOLOR_SPACE_TYPE const>(lwbType - 1);

	/* This plugin called frop PR - check video fomat */
	AEFX_SuiteScoper<PF_PixelFormatSuite1> pixelFormatSuite =
		AEFX_SuiteScoper<PF_PixelFormatSuite1>(
			in_data,
			kPFPixelFormatSuite,
			kPFPixelFormatSuiteVersion1,
			out_data);

	PrPixelFormat destinationPixelFormat = PrPixelFormat_Invalid;
	if (PF_Err_NONE == (errFormat = pixelFormatSuite->GetPixelFormat(output, &destinationPixelFormat)))
	{
		switch (destinationPixelFormat)
		{
			case PrPixelFormat_BGRA_4444_8u:
			{
				switch (colorSpaceType)
				{
					case COLOR_SPACE_HSL:
						err = prProcessImage_BGRA_4444_8u_HSL(in_data, out_data, params, output, totalHue, totalSat, totalLwb);
					break;
					case COLOR_SPACE_HSV:
						err = prProcessImage_BGRA_4444_8u_HSV(in_data, out_data, params, output, totalHue, totalSat, totalLwb);
					break;
					case COLOR_SPACE_HSI:
						err = prProcessImage_BGRA_4444_8u_HSI(in_data, out_data, params, output, totalHue, totalSat, totalLwb);
					break;
					case COLOR_SPACE_HSP:
						err = prProcessImage_BGRA_4444_8u_HSP(in_data, out_data, params, output, totalHue, totalSat, totalLwb);
					break;
					case COLOR_SPACE_HSLuma:
					break;
				}
			}
			break;

			case PrPixelFormat_BGRA_4444_16u:
			{
				switch (colorSpaceType)
				{
					case COLOR_SPACE_HSL:
						err = prProcessImage_BGRA_4444_16u_HSL(in_data, out_data, params, output, totalHue, totalSat, totalLwb);
					break;
					case COLOR_SPACE_HSV:
						err = prProcessImage_BGRA_4444_16u_HSV(in_data, out_data, params, output, totalHue, totalSat, totalLwb);
					break;
					case COLOR_SPACE_HSI:
					break;
					case COLOR_SPACE_HSP:
					break;
					case COLOR_SPACE_HSLuma:
					break;
				}
			}
			break;

			case PrPixelFormat_BGRA_4444_32f:
			{
				switch (colorSpaceType)
				{
					case COLOR_SPACE_HSL:
						err = prProcessImage_BGRA_4444_32f_HSL(in_data, out_data, params, output, totalHue, totalSat, totalLwb);
					break;
					case COLOR_SPACE_HSV:
						err = prProcessImage_BGRA_4444_32f_HSV(in_data, out_data, params, output, totalHue, totalSat, totalLwb);
					break;
					case COLOR_SPACE_HSI:
					break;
					case COLOR_SPACE_HSP:
					break;
					case COLOR_SPACE_HSLuma:
					break;
				}
			}
			break;

			case PrPixelFormat_RGB_444_10u:
			{
				switch (colorSpaceType)
				{
					case COLOR_SPACE_HSL:
						err = prProcessImage_RGB_444_10u_HSL(in_data, out_data, params, output, totalHue, totalSat, totalLwb);
					break;
					case COLOR_SPACE_HSV:
						err = prProcessImage_RGB_444_10u_HSV(in_data, out_data, params, output, totalHue, totalSat, totalLwb);
					break;
					case COLOR_SPACE_HSI:
					break;
					case COLOR_SPACE_HSP:
					break;
					case COLOR_SPACE_HSLuma:
					break;
				}
			}
			break;

			case PrPixelFormat_VUYA_4444_8u_709:
			case PrPixelFormat_VUYA_4444_8u:
			{
				auto const& isBT709 = PrPixelFormat_VUYA_4444_8u_709 == destinationPixelFormat;
				switch (colorSpaceType)
				{
					case COLOR_SPACE_HSL:
						err = prProcessImage_VUYA_4444_8u_HSL(in_data, out_data, params, output, totalHue, totalSat, totalLwb, isBT709);
					break;
					case COLOR_SPACE_HSV:
						err = prProcessImage_VUYA_4444_8u_HSV(in_data, out_data, params, output, totalHue, totalSat, totalLwb, isBT709);
					break;
					case COLOR_SPACE_HSI:
					break;
					case COLOR_SPACE_HSP:
					break;
					case COLOR_SPACE_HSLuma:
					break;
				}
			}
			break;

			case PrPixelFormat_VUYA_4444_32f_709:
			case PrPixelFormat_VUYA_4444_32f:
			{
				auto const& isBT709 = PrPixelFormat_VUYA_4444_32f_709 == destinationPixelFormat;
				switch (colorSpaceType)
				{
					case COLOR_SPACE_HSL:
						err = prProcessImage_VUYA_4444_32f_HSL(in_data, out_data, params, output, totalHue, totalSat, totalLwb, isBT709);
					break;
					case COLOR_SPACE_HSV:
						err = prProcessImage_VUYA_4444_32f_HSV(in_data, out_data, params, output, totalHue, totalSat, totalLwb, isBT709);
					break;
					case COLOR_SPACE_HSI:
					break;
					case COLOR_SPACE_HSP:
					break;
					case COLOR_SPACE_HSLuma:
					break;
				}
			}
			break;

			default:
				err = PF_Err_INVALID_INDEX;
			break;
		}
	} /* if (PF_Err_NONE == (errFormat = pixelFormatSuite->GetPixelFormat(output, &destinationPixelFormat))) */
	else
	{
		/* error in determine pixel format */
		err = PF_Err_INVALID_INDEX;
	}

	return err;
}