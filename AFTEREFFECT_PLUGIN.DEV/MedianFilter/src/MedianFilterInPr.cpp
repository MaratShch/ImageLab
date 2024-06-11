#include "MedianFilter.hpp"
#include "MedianFilterEnums.hpp"
#include "PrSDKAESupport.h"
#include "MedianFilterAvx2.hpp"


PF_Err MedianFilter_BGRA_4444_8u
(
	PF_InData*   __restrict in_data,
	PF_OutData*  __restrict out_data,
	PF_ParamDef* __restrict params[],
	PF_LayerDef* __restrict output
) noexcept
{
	const PF_LayerDef* __restrict pfLayer = reinterpret_cast<const PF_LayerDef* __restrict>(&params[MEDIAN_FILTER_INPUT]->u.ld);
	uint32_t*  __restrict localSrc = reinterpret_cast<uint32_t* __restrict>(pfLayer->data);
	uint32_t*  __restrict localDst = reinterpret_cast<uint32_t* __restrict>(output->data);

	const A_long height     = pfLayer->extent_hint.bottom - pfLayer->extent_hint.top;
	const A_long width      = pfLayer->extent_hint.right  - pfLayer->extent_hint.left;
	const A_long line_pitch = pfLayer->rowbytes / static_cast<A_long>(PF_Pixel_BGRA_8u_size);

	const A_long kernelSize = get_kernel_size(params);
	bool medianResult = false;

	switch (kernelSize)
	{
		case 0:
			/* median filter disabled - just copy from source to destination */
			PF_COPY(&params[MEDIAN_FILTER_INPUT]->u.ld, output, NULL, NULL);
			medianResult = true;
		break;

		case 3:
			/* manually optimized variant 3x3 */
			medianResult = AVX2::Median::median_filter_3x3_BGRA_4444_8u (localSrc, localDst, height, width, line_pitch, line_pitch);
		break;

		case 5:
			/* manually optimized variant 5x5 */
			medianResult = AVX2::Median::median_filter_5x5_BGRA_4444_8u(localSrc, localDst, height, width, line_pitch, line_pitch);
		break;

		case 7:
			/* manually optimized variant 7x7 */
			medianResult = AVX2::Median::median_filter_7x7_BGRA_4444_8u (localSrc, localDst, height, width, line_pitch, line_pitch);
		break;

		default:
			/* median via histogramm algo */
			medianResult = median_filter_constant_time_BGRA_4444_8u (localSrc, localDst, height, width, line_pitch, line_pitch, kernelSize);
		break;
	}

	return (true == medianResult ? PF_Err_NONE : PF_Err_INTERNAL_STRUCT_DAMAGED);
}


PF_Err MedianFilter_BGRA_4444_16u
(
	PF_InData*   __restrict in_data,
	PF_OutData*  __restrict out_data,
	PF_ParamDef* __restrict params[],
	PF_LayerDef* __restrict output
) noexcept
{
	const PF_LayerDef* __restrict pfLayer = reinterpret_cast<const PF_LayerDef* __restrict>(&params[MEDIAN_FILTER_INPUT]->u.ld);
	const uint32_t*    __restrict localSrc = reinterpret_cast<const uint32_t* __restrict>(pfLayer->data);
	      uint32_t*    __restrict localDst = reinterpret_cast<      uint32_t* __restrict>(output->data);

	const A_long height     = pfLayer->extent_hint.bottom - pfLayer->extent_hint.top;
	const A_long width      = pfLayer->extent_hint.right  - pfLayer->extent_hint.left;
	const A_long line_pitch = pfLayer->rowbytes / static_cast<A_long>(PF_Pixel_BGRA_8u_size);

	const A_long kernelSize = get_kernel_size(params);
	bool medianResult = false;

	switch (kernelSize)
	{
		case 0:
			/* median filter disabled - just copy from source to destination */
			PF_COPY(&params[MEDIAN_FILTER_INPUT]->u.ld, output, NULL, NULL);
		break;
	
	    case 3:
		/* manually optimized variant 3x3 */
		break;

		case 5:
		/* manually optimized variant 5x5 */
		break;

		case 7:
		/* manually optimized variant 7x7 */
		break;

		default:
			/* median via histogramm algo */
			medianResult = median_filter_constant_time_BGRA_4444_16u (localSrc, localDst, height, width, line_pitch, line_pitch, kernelSize);
		break;
	}

	return (true == medianResult ? PF_Err_NONE : PF_Err_INTERNAL_STRUCT_DAMAGED);
}


PF_Err MedianFilter_BGRA_4444_32f
(
	PF_InData*   __restrict in_data,
	PF_OutData*  __restrict out_data,
	PF_ParamDef* __restrict params[],
	PF_LayerDef* __restrict output
) noexcept
{
	const PF_LayerDef* __restrict pfLayer  = reinterpret_cast<const PF_LayerDef* __restrict>(&params[MEDIAN_FILTER_INPUT]->u.ld);
	const uint32_t*    __restrict localSrc = reinterpret_cast<const uint32_t* __restrict>(pfLayer->data);
	      uint32_t*    __restrict localDst = reinterpret_cast<      uint32_t* __restrict>(output->data);

	auto const height = pfLayer->extent_hint.bottom - pfLayer->extent_hint.top;
	auto const width  = pfLayer->extent_hint.right - pfLayer->extent_hint.left;
	auto const line_pitch = pfLayer->rowbytes / static_cast<A_long>(PF_Pixel_BGRA_32f_size);

	auto const kernelSize = get_kernel_size(params);
	bool medianResult = false;

	switch (kernelSize)
	{
		case 0:
			/* median filter disabled - just copy from source to destination */
			PF_COPY(&params[MEDIAN_FILTER_INPUT]->u.ld, output, NULL, NULL);
		break;
		
		case 3:
		/* manually optimized variant 3x3 */
		break;

		case 5:
		/* manually optimized variant 5x5 */
		break;

		case 7:
		/* manually optimized variant 7x7 */
		break;

		default:
			/* median via histogramm algo */
			medianResult = median_filter_constant_time_BGRA_4444_32f (localSrc, localDst, height, width, line_pitch, line_pitch, kernelSize);
		break;
	}

	return (true == medianResult ? PF_Err_NONE : PF_Err_INTERNAL_STRUCT_DAMAGED);
}


PF_Err ProcessImgInPR
(
	PF_InData*   __restrict in_data,
	PF_OutData*  __restrict out_data,
	PF_ParamDef* __restrict params[],
	PF_LayerDef* __restrict output
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
				err = MedianFilter_BGRA_4444_8u (in_data, out_data, params, output);
			break;

			case PrPixelFormat_BGRA_4444_16u:
				err = MedianFilter_BGRA_4444_16u (in_data, out_data, params, output);
			break;

			case PrPixelFormat_BGRA_4444_32f:
				err = MedianFilter_BGRA_4444_32f (in_data, out_data, params, output);
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
