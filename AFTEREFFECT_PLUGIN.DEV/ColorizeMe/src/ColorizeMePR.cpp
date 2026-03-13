#include "ColorizeMe.hpp"
#include "CommonDebugUtils.hpp"


bool ProcessPrImage_BGRA_4444_8u
(
	PF_InData*    in_data,
	PF_OutData*   out_data,
	PF_ParamDef*  params[],
	PF_LayerDef*  output
) noexcept
{
	return true;
}


bool ProcessPrImage_BGRA_4444_16u
(
	PF_InData*    in_data,
	PF_OutData*   out_data,
	PF_ParamDef*  params[],
	PF_LayerDef*  output
) noexcept
{
	return true;
}

bool ProcessPrImage_BGRA_4444_32f
(
	PF_InData*    in_data,
	PF_OutData*   out_data,
	PF_ParamDef*  params[],
	PF_LayerDef*  output
) noexcept
{
	return true;
}


bool ProcessPrImage_VUYA_4444_8u
(
	PF_InData*    in_data,
	PF_OutData*   out_data,
	PF_ParamDef*  params[],
	PF_LayerDef*  output,
	const bool isBT709
) noexcept
{
	return true;
}


bool ProcessPrImage_VUYA_4444_32f
(
	PF_InData*    in_data,
	PF_OutData*   out_data,
	PF_ParamDef*  params[],
	PF_LayerDef*  output,
	const bool isBT709
) noexcept
{
	return true;
}


bool ProcessPrImage_RGB_444_10u
(
	PF_InData*    in_data,
	PF_OutData*   out_data,
	PF_ParamDef*  params[],
	PF_LayerDef*  output
) noexcept
{
	return true;
}



PF_Err ProcessImgInPR
(
	PF_InData*    in_data,
	PF_OutData*   out_data,
	PF_ParamDef*  params[],
	PF_LayerDef*  output,
	const PrPixelFormat&    pixelFormat
) noexcept
{
	bool bValue = true;

	/* acquire controls parameters */

	switch (pixelFormat)
	{
		case PrPixelFormat_BGRA_4444_8u:
			bValue = ProcessPrImage_BGRA_4444_8u(in_data, out_data, params, output);
		break;

		case PrPixelFormat_BGRA_4444_16u:
			bValue = ProcessPrImage_BGRA_4444_16u(in_data, out_data, params, output);
		break;

		case PrPixelFormat_BGRA_4444_32f:
			bValue = ProcessPrImage_BGRA_4444_32f(in_data, out_data, params, output);
		break;

		case PrPixelFormat_VUYA_4444_8u:
		case PrPixelFormat_VUYA_4444_8u_709:
			bValue = ProcessPrImage_VUYA_4444_8u(in_data, out_data, params, output, PrPixelFormat_VUYA_4444_8u_709 == pixelFormat);
		break;

		case PrPixelFormat_VUYA_4444_32f:
		case PrPixelFormat_VUYA_4444_32f_709:
			bValue = ProcessPrImage_VUYA_4444_32f(in_data, out_data, params, output, PrPixelFormat_VUYA_4444_8u_709 == pixelFormat);
		break;

		case PrPixelFormat_RGB_444_10u:
			bValue = ProcessPrImage_RGB_444_10u(in_data, out_data, params, output);
		break;

		default:
			bValue = false;
		break;
	}

	return (true == bValue ? PF_Err_NONE : PF_Err_INTERNAL_STRUCT_DAMAGED);
}