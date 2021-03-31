#include "ColorCorrectionCMYK.hpp"
#include "ColorCorrectionEnums.hpp"
#include "PrSDKAESupport.h"
#include "RGB2CMYK.hpp"

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
	auto const& cType   = params[COLOR_CORRECT_SPACE_POPUP]->u.pd.value;
	auto const& Coarse1 = params[COLOR_CORRECT_SLIDER1]->u.sd.value;
	auto const& Fine1   = params[COLOR_CORRECT_SLIDER2]->u.fs_d.value;
	auto const& Coarse2 = params[COLOR_CORRECT_SLIDER3]->u.sd.value;
	auto const& Fine2   = params[COLOR_CORRECT_SLIDER4]->u.fs_d.value;
	auto const& Coarse3 = params[COLOR_CORRECT_SLIDER5]->u.sd.value;
	auto const& Fine3   = params[COLOR_CORRECT_SLIDER6]->u.fs_d.value;
	auto const& Coarse4 = params[COLOR_CORRECT_SLIDER7]->u.sd.value;
	auto const& Fine4   = params[COLOR_CORRECT_SLIDER8]->u.fs_d.value;

	eCOLOR_SPACE_TYPE const& colorSpaceType = static_cast<eCOLOR_SPACE_TYPE const>(cType - 1);
	float const& cVal = static_cast<float>(static_cast<double>(Coarse1) + Fine1);
	float const& mVal = static_cast<float>(static_cast<double>(Coarse2) + Fine2);
	float const& yVal = static_cast<float>(static_cast<double>(Coarse3) + Fine3);
	float const& kVal = static_cast<float>(static_cast<double>(Coarse4) + Fine4);

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
					case COLOR_SPACE_CMYK:
						prProcessImage_BGRA_4444_8u_CMYK(in_data, out_data, params, output, cVal, mVal, yVal, kVal);
					break;
					case COLOR_SPACE_RGB:
						prProcessImage_BGRA_4444_8u_RGB(in_data, out_data, params, output, cVal, mVal, yVal);
					break;
					default:
						err = PF_Err_INVALID_INDEX;
					break;
				}
			}
			break;

			case PrPixelFormat_BGRA_4444_16u:
			{
				switch (colorSpaceType)
				{
					case COLOR_SPACE_CMYK:
						prProcessImage_BGRA_4444_16u_CMYK(in_data, out_data, params, output, cVal, mVal, yVal, kVal);
					break;
					case COLOR_SPACE_RGB:
						prProcessImage_BGRA_4444_16u_RGB(in_data, out_data, params, output, cVal, mVal, yVal);
					break;
					default:
						err = PF_Err_INVALID_INDEX;
					break;
				}
			}
			break;

			case PrPixelFormat_BGRA_4444_32f:
			{
				switch (colorSpaceType)
				{
					case COLOR_SPACE_CMYK:
						prProcessImage_BGRA_4444_32f_CMYK(in_data, out_data, params, output, cVal, mVal, yVal, kVal);
					break;
					case COLOR_SPACE_RGB:
						prProcessImage_BGRA_4444_32f_RGB(in_data, out_data, params, output, cVal, mVal, yVal);
					break;
					default:
						err = PF_Err_INVALID_INDEX;
					break;
				}
			}
			break;

			case PrPixelFormat_RGB_444_10u:
			{
				switch (colorSpaceType)
				{
					case COLOR_SPACE_CMYK:
					break;
					case COLOR_SPACE_RGB:
					break;
					default:
					err = PF_Err_INVALID_INDEX;
					break;
				}
			}
			break;

			case PrPixelFormat_VUYA_4444_8u_709:
			case PrPixelFormat_VUYA_4444_8u:
			{
				auto const& isBT709 = (PrPixelFormat_VUYA_4444_8u_709 == destinationPixelFormat);
				switch (colorSpaceType)
				{
					case COLOR_SPACE_CMYK:
					break;
					case COLOR_SPACE_RGB:
					break;
					default:
					err = PF_Err_INVALID_INDEX;
					break;
				}
			}
			break;

			case PrPixelFormat_VUYA_4444_32f_709:
			case PrPixelFormat_VUYA_4444_32f:
			{
				auto const& isBT709 = (PrPixelFormat_VUYA_4444_32f_709 == destinationPixelFormat);
				switch (colorSpaceType)
				{
					case COLOR_SPACE_CMYK:
					break;
					case COLOR_SPACE_RGB:
					break;
					default:
					err = PF_Err_INVALID_INDEX;
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
		err = PF_Err_UNRECOGNIZED_PARAM_TYPE;
	}

	return err;
}