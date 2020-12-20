#include "ColorizeMe.hpp"
#include "CubeLUT.h"

static bool ProcessPrImage_BGRA_4444_8u
(
	PF_InData*   __restrict in_data,
	PF_OutData*  __restrict out_data,
	PF_ParamDef* __restrict params[],
	PF_LayerDef* __restrict output
) noexcept
{
	if (nullptr != in_data->sequence_data)
	{
		CubeLUT* pLut = static_cast<CubeLUT*>(*in_data->sequence_data);
	}

	return true;
}


static bool ProcessPrImage_BGRA_4444_16u
(
	PF_InData*   __restrict in_data,
	PF_OutData*  __restrict out_data,
	PF_ParamDef* __restrict params[],
	PF_LayerDef* __restrict output
) noexcept
{
	return true;
}


static bool ProcessPrImage_VUYA_4444_8u
(
	PF_InData*   __restrict in_data,
	PF_OutData*  __restrict out_data,
	PF_ParamDef* __restrict params[],
	PF_LayerDef* __restrict output,
	const bool isBT709
) noexcept
{
	return true;
}




PF_Err ProcessImgInPR
(
	PF_InData*   __restrict in_data,
	PF_OutData*  __restrict out_data,
	PF_ParamDef* __restrict params[],
	PF_LayerDef* __restrict output,
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

		case PrPixelFormat_VUYA_4444_8u:
		case PrPixelFormat_VUYA_4444_8u_709:
			bValue = ProcessPrImage_VUYA_4444_8u(in_data, out_data, params, output, PrPixelFormat_VUYA_4444_8u_709 == pixelFormat);
		break;

		default:
			bValue = false;
		break;
	}


	return (true == bValue ? PF_Err_NONE : PF_Err_INTERNAL_STRUCT_DAMAGED);
}