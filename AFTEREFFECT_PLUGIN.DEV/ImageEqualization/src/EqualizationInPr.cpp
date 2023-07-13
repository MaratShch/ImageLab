#include "ImageEqualization.hpp"
#include "PrSDKAESupport.h"


PF_Err PR_ImageEq_Manual
(
	PF_InData*    in_data,
	PF_OutData*   out_data,
	PF_ParamDef*  params[],
	PF_LayerDef*  output
) noexcept
{
	PF_Err err = PF_Err_NONE;
	PrPixelFormat destinationPixelFormat = PrPixelFormat_Invalid;

	/* This plugin called from PR - check video fomat */
	if (PF_Err_NONE == AEFX_SuiteScoper<PF_PixelFormatSuite1>(in_data, kPFPixelFormatSuite, kPFPixelFormatSuiteVersion1, out_data)->GetPixelFormat(output, &destinationPixelFormat))
	{
		switch (destinationPixelFormat)
		{
			case PrPixelFormat_BGRA_4444_8u:
				err = PR_ImageEq_Manual_BGRA_4444_8u(in_data, out_data, params, output);
			break;

			case PrPixelFormat_BGRA_4444_16u:
				err = PR_ImageEq_Manual_BGRA_4444_16u(in_data, out_data, params, output);
			break;

			case PrPixelFormat_BGRA_4444_32f:
				err = PR_ImageEq_Manual_BGRA_4444_32f(in_data, out_data, params, output);
			break;

			case PrPixelFormat_VUYA_4444_8u:
				err = PR_ImageEq_Manual_VUYA_4444_8u(in_data, out_data, params, output);
			break;

			case PrPixelFormat_VUYA_4444_8u_709:
				err = PR_ImageEq_Manual_VUYA_4444_8u_709(in_data, out_data, params, output);
			break;

			case PrPixelFormat_VUYA_4444_32f:
				err = PR_ImageEq_Manual_VUYA_4444_32f(in_data, out_data, params, output);
			break;

			case PrPixelFormat_VUYA_4444_32f_709:
				err = PR_ImageEq_Manual_VUYA_4444_32f_709(in_data, out_data, params, output);
			break;

			default:
			break;
		} /* switch (destinationPixelFormat) */

	} /* if (PF_Err_NONE == (errFormat = pixelFormatSuite->GetPixelFormat(output, &destinationPixelFormat))) */
	else
	{
		/* error in determine pixel format */
		err = PF_Err_UNRECOGNIZED_PARAM_TYPE;
	}

	return err;
}


PF_Err PR_ImageEq_Linear
(
	PF_InData*    in_data,
	PF_OutData*   out_data,
	PF_ParamDef*  params[],
	PF_LayerDef*  output
) noexcept
{
	PF_Err err = PF_Err_NONE;
	PrPixelFormat destinationPixelFormat = PrPixelFormat_Invalid;

	/* This plugin called from PR - check video fomat */
	if (PF_Err_NONE == AEFX_SuiteScoper<PF_PixelFormatSuite1>(in_data, kPFPixelFormatSuite, kPFPixelFormatSuiteVersion1, out_data)->GetPixelFormat(output, &destinationPixelFormat))
	{
		switch (destinationPixelFormat)
		{
			case PrPixelFormat_BGRA_4444_8u:
				err = PR_ImageEq_Linear_BGRA_4444_8u(in_data, out_data, params, output);
			break;

			case PrPixelFormat_BGRA_4444_16u:
				err = PR_ImageEq_Linear_BGRA_4444_16u (in_data, out_data, params, output);
			break;

			case PrPixelFormat_BGRA_4444_32f:
				err = PR_ImageEq_Linear_BGRA_4444_32f (in_data, out_data, params, output);
			break;

			case PrPixelFormat_VUYA_4444_8u:
				err = PR_ImageEq_Linear_VUYA_4444_8u (in_data, out_data, params, output);
			break;

			case PrPixelFormat_VUYA_4444_8u_709:
				err = PR_ImageEq_Linear_VUYA_4444_8u_709 (in_data, out_data, params, output);
			break;

			case PrPixelFormat_VUYA_4444_32f:
				err = PR_ImageEq_Linear_VUYA_4444_32f (in_data, out_data, params, output);
			break;

			case PrPixelFormat_VUYA_4444_32f_709:
				err = PR_ImageEq_Linear_VUYA_4444_32f_709 (in_data, out_data, params, output);
			break;

			default:
			break;
		} /* switch (destinationPixelFormat) */

	} /* if (PF_Err_NONE == (errFormat = pixelFormatSuite->GetPixelFormat(output, &destinationPixelFormat))) */
	else
	{
		/* error in determine pixel format */
		err = PF_Err_UNRECOGNIZED_PARAM_TYPE;
	}

	return err;
}


PF_Err PR_ImageEq_Bright
(
	PF_InData*    in_data,
	PF_OutData*   out_data,
	PF_ParamDef*  params[],
	PF_LayerDef*  output
) noexcept
{
	PF_Err err = PF_Err_NONE;
	PrPixelFormat destinationPixelFormat = PrPixelFormat_Invalid;

	/* This plugin called from PR - check video fomat */
	if (PF_Err_NONE == AEFX_SuiteScoper<PF_PixelFormatSuite1>(in_data, kPFPixelFormatSuite, kPFPixelFormatSuiteVersion1, out_data)->GetPixelFormat(output, &destinationPixelFormat))
	{
		switch (destinationPixelFormat)
		{
			case PrPixelFormat_BGRA_4444_8u:
				err = PR_ImageEq_Bright_BGRA_4444_8u (in_data, out_data, params, output);
			break;

			case PrPixelFormat_BGRA_4444_16u:
				err = PR_ImageEq_Bright_BGRA_4444_16u (in_data, out_data, params, output);
			break;

			case PrPixelFormat_BGRA_4444_32f:
				err = PR_ImageEq_Bright_BGRA_4444_32f (in_data, out_data, params, output);
			break;

			case PrPixelFormat_VUYA_4444_8u:
				err = PR_ImageEq_Bright_VUYA_4444_8u (in_data, out_data, params, output);
			break;

			case PrPixelFormat_VUYA_4444_8u_709:
				err = PR_ImageEq_Bright_VUYA_4444_8u_709 (in_data, out_data, params, output);
			break;

			case PrPixelFormat_VUYA_4444_32f:
				err = PR_ImageEq_Bright_VUYA_4444_32f (in_data, out_data, params, output);
			break;

			case PrPixelFormat_VUYA_4444_32f_709:
				err = PR_ImageEq_Bright_VUYA_4444_32f_709 (in_data, out_data, params, output);
			break;

			default:
				err = PF_Err_UNRECOGNIZED_PARAM_TYPE;
			break;
		} /* switch (destinationPixelFormat) */

	} /* if (PF_Err_NONE == (errFormat = pixelFormatSuite->GetPixelFormat(output, &destinationPixelFormat))) */
	else
	{
		/* error in determine pixel format */
		err = PF_Err_UNRECOGNIZED_PARAM_TYPE;
	}

	return err;
}


PF_Err PR_ImageEq_Dark
(
	PF_InData*    in_data,
	PF_OutData*   out_data,
	PF_ParamDef*  params[],
	PF_LayerDef*  output
) noexcept
{
	PF_Err err = PF_Err_NONE;
	PrPixelFormat destinationPixelFormat = PrPixelFormat_Invalid;

	/* This plugin called from PR - check video fomat */
	if (PF_Err_NONE == AEFX_SuiteScoper<PF_PixelFormatSuite1>(in_data, kPFPixelFormatSuite, kPFPixelFormatSuiteVersion1, out_data)->GetPixelFormat(output, &destinationPixelFormat))
	{
		switch (destinationPixelFormat)
		{
			case PrPixelFormat_BGRA_4444_8u:
				err = PR_ImageEq_Dark_BGRA_4444_8u(in_data, out_data, params, output);
			break;

			case PrPixelFormat_BGRA_4444_16u:
				err = PR_ImageEq_Dark_BGRA_4444_16u(in_data, out_data, params, output);
			break;

			case PrPixelFormat_BGRA_4444_32f:
				err = PR_ImageEq_Dark_BGRA_4444_32f(in_data, out_data, params, output);
			break;

			case PrPixelFormat_VUYA_4444_8u:
				err = PR_ImageEq_Dark_VUYA_4444_8u(in_data, out_data, params, output);
			break;

			case PrPixelFormat_VUYA_4444_8u_709:
				err = PR_ImageEq_Dark_VUYA_4444_8u_709(in_data, out_data, params, output);
			break;

			case PrPixelFormat_VUYA_4444_32f:
				err = PR_ImageEq_Dark_VUYA_4444_32f(in_data, out_data, params, output);
			break;

			case PrPixelFormat_VUYA_4444_32f_709:
				err = PR_ImageEq_Dark_VUYA_4444_32f_709(in_data, out_data, params, output);
			break;

			default:
				err = PF_Err_UNRECOGNIZED_PARAM_TYPE;
			break;
		} /* switch (destinationPixelFormat) */

	} /* if (PF_Err_NONE == (errFormat = pixelFormatSuite->GetPixelFormat(output, &destinationPixelFormat))) */
	else
	{
		/* error in determine pixel format */
		err = PF_Err_UNRECOGNIZED_PARAM_TYPE;
	}

	return err;
}


PF_Err PR_ImageEq_Exponential
(
	PF_InData*    in_data,
	PF_OutData*   out_data,
	PF_ParamDef*  params[],
	PF_LayerDef*  output
) noexcept
{
	PF_Err err = PF_Err_NONE;
	PrPixelFormat destinationPixelFormat = PrPixelFormat_Invalid;

	/* This plugin called from PR - check video fomat */
	if (PF_Err_NONE == AEFX_SuiteScoper<PF_PixelFormatSuite1>(in_data, kPFPixelFormatSuite, kPFPixelFormatSuiteVersion1, out_data)->GetPixelFormat(output, &destinationPixelFormat))
	{
		switch (destinationPixelFormat)
		{
			case PrPixelFormat_BGRA_4444_8u:
				err = PR_ImageEq_Exponential_BGRA_4444_8u(in_data, out_data, params, output);
			break;

			case PrPixelFormat_BGRA_4444_16u:
				err = PR_ImageEq_Exponential_BGRA_4444_16u(in_data, out_data, params, output);
			break;

			case PrPixelFormat_BGRA_4444_32f:
				err = PR_ImageEq_Exponential_BGRA_4444_32f(in_data, out_data, params, output);
			break;

			case PrPixelFormat_VUYA_4444_8u:
				err = PR_ImageEq_Exponential_VUYA_4444_8u(in_data, out_data, params, output);
			break;

			case PrPixelFormat_VUYA_4444_8u_709:
				err = PR_ImageEq_Exponential_VUYA_4444_8u_709(in_data, out_data, params, output);
			break;

			case PrPixelFormat_VUYA_4444_32f:
				err = PR_ImageEq_Exponential_VUYA_4444_32f(in_data, out_data, params, output);
			break;

			case PrPixelFormat_VUYA_4444_32f_709:
				err = PR_ImageEq_Exponential_VUYA_4444_32f_709(in_data, out_data, params, output);
			break;

			default:
				err = PF_Err_UNRECOGNIZED_PARAM_TYPE;
			break;
		} /* switch (destinationPixelFormat) */

	} /* if (PF_Err_NONE == (errFormat = pixelFormatSuite->GetPixelFormat(output, &destinationPixelFormat))) */
	else
	{
		/* error in determine pixel format */
		err = PF_Err_UNRECOGNIZED_PARAM_TYPE;
	}

	return err;
}


PF_Err PR_ImageEq_Sigmoid
(
	PF_InData*    in_data,
	PF_OutData*   out_data,
	PF_ParamDef*  params[],
	PF_LayerDef*  output
) noexcept
{
	PF_Err err = PF_Err_NONE;
	PrPixelFormat destinationPixelFormat = PrPixelFormat_Invalid;

	/* This plugin called from PR - check video fomat */
	if (PF_Err_NONE == AEFX_SuiteScoper<PF_PixelFormatSuite1>(in_data, kPFPixelFormatSuite, kPFPixelFormatSuiteVersion1, out_data)->GetPixelFormat(output, &destinationPixelFormat))
	{
		switch (destinationPixelFormat)
		{
			case PrPixelFormat_BGRA_4444_8u:
				err = PR_ImageEq_Sigmoid_BGRA_4444_8u (in_data, out_data, params, output);
			break;

			case PrPixelFormat_BGRA_4444_16u:
				err = PR_ImageEq_Sigmoid_BGRA_4444_16u (in_data, out_data, params, output);
			break;

			case PrPixelFormat_BGRA_4444_32f:
				err = PR_ImageEq_Sigmoid_BGRA_4444_32f (in_data, out_data, params, output);
			break;

			case PrPixelFormat_VUYA_4444_8u:
			case PrPixelFormat_VUYA_4444_8u_709:
				err = PR_ImageEq_Sigmoid_VUYA_4444_8u_709 (in_data, out_data, params, output);
			break;

			case PrPixelFormat_VUYA_4444_32f:
			case PrPixelFormat_VUYA_4444_32f_709:
				err = PR_ImageEq_Sigmoid_VUYA_4444_32f_709 (in_data, out_data, params, output);
			break;

			default:
			break;
		} /* switch (destinationPixelFormat) */

	} /* if (PF_Err_NONE == (errFormat = pixelFormatSuite->GetPixelFormat(output, &destinationPixelFormat))) */
	else
	{
		/* error in determine pixel format */
		err = PF_Err_UNRECOGNIZED_PARAM_TYPE;
	}

	return err;
}


PF_Err PR_ImageEq_Advanced
(
	PF_InData*    in_data,
	PF_OutData*   out_data,
	PF_ParamDef*  params[],
	PF_LayerDef*  output
) noexcept
{
	PF_Err err = PF_Err_NONE;
	PrPixelFormat destinationPixelFormat = PrPixelFormat_Invalid;

	/* This plugin called from PR - check video fomat */
	if (PF_Err_NONE == AEFX_SuiteScoper<PF_PixelFormatSuite1>(in_data, kPFPixelFormatSuite, kPFPixelFormatSuiteVersion1, out_data)->GetPixelFormat(output, &destinationPixelFormat))
	{
		switch (destinationPixelFormat)
		{
			case PrPixelFormat_BGRA_4444_8u:
				err = PR_ImageEq_Advanced_BGRA_4444_8u(in_data, out_data, params, output);
			break;

			case PrPixelFormat_BGRA_4444_16u:
				err = PR_ImageEq_Advanced_BGRA_4444_16u(in_data, out_data, params, output);
			break;

			case PrPixelFormat_BGRA_4444_32f:
				err = PR_ImageEq_Advanced_BGRA_4444_32f(in_data, out_data, params, output);
			break;

			case PrPixelFormat_VUYA_4444_8u:
			case PrPixelFormat_VUYA_4444_8u_709:
				err = PR_ImageEq_Advanced_VUYA_4444_8u_709(in_data, out_data, params, output);
			break;

			case PrPixelFormat_VUYA_4444_32f:
			case PrPixelFormat_VUYA_4444_32f_709:
				err = PR_ImageEq_Advanced_VUYA_4444_32f_709(in_data, out_data, params, output);
			break;

			default:
			break;
		} /* switch (destinationPixelFormat) */

	} /* if (PF_Err_NONE == (errFormat = pixelFormatSuite->GetPixelFormat(output, &destinationPixelFormat))) */
	else
	{
		/* error in determine pixel format */
		err = PF_Err_UNRECOGNIZED_PARAM_TYPE;
	}

	return err;
}


PF_Err ProcessImgInPR
(
	PF_InData*    in_data,
	PF_OutData*   out_data,
	PF_ParamDef*  params[],
	PF_LayerDef*  output
) noexcept
{
	PF_Err err{ PF_Err_NONE };
	const ImageEqPopupAlgo eqType{ static_cast<const ImageEqPopupAlgo>(params[IMAGE_EQUALIZATION_POPUP_PRESET]->u.pd.value - 1) };

	switch (eqType)
	{
		case IMAGE_EQ_LINEAR:
			PR_ImageEq_Linear (in_data, out_data, params, output);
		break;

		case IMAGE_EQ_BRIGHT:
			PR_ImageEq_Bright (in_data, out_data, params, output);
		break;

		case IMAGE_EQ_DARK:
			PR_ImageEq_Dark (in_data, out_data, params, output);
		break;

		case IMAGE_EQ_EXPONENTIAL:
			PR_ImageEq_Exponential (in_data, out_data, params, output);
		break;

		case IMAGE_EQ_SIGMOID:
			PR_ImageEq_Sigmoid (in_data, out_data, params, output);
		break;

		case IMAGE_EQ_ADVANCED:
			PR_ImageEq_Advanced (in_data, out_data, params, output);
		break;

		case IMAGE_EQ_MANUAL:
			PR_ImageEq_Manual (in_data, out_data, params, output);
		break;

		case IMAGE_EQ_NONE:
		default:
			err = PF_COPY(&params[IMAGE_EQUALIZATION_FILTER_INPUT]->u.ld, output, NULL, NULL);
		break;
	}

	return err;
}
