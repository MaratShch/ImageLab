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

	auto const height     = pfLayer->extent_hint.bottom - pfLayer->extent_hint.top;
	auto const width      = pfLayer->extent_hint.right  - pfLayer->extent_hint.left;
	auto const line_pitch = pfLayer->rowbytes / static_cast<A_long>(PF_Pixel_BGRA_8u_size);

	auto const kernelSize   = get_kernel_size(params);

	bool avx2ProcReturn = false;

	switch (kernelSize)
	{
		case 3:
			/* manually optimized variant 3x3 */
				median_filter_3x3_BGRA_4444_8u (localSrc, localDst, height, width, line_pitch, line_pitch);
			break;

		case 5:
			/* manually optimized variant 5x5 */
				median_filter_5x5_BGRA_4444_8u(localSrc, localDst, height, width, line_pitch, line_pitch);
		break;

		case 7:
			/* manually optimized variant 7x7 */
				median_filter_7x7_BGRA_4444_8u (localSrc, localDst, height, width, line_pitch, line_pitch);
		break;

		case 9:
			/* manually optimized variant 9x9  */
			//			true == procLumaOnly ?
			//				median_filter_7x7_BGRA_4444_uint_luma_only (localSrc, localDst, height, width, line_pitch) :
			//				median_filter_7x7_BGRA_4444_uint (localSrc, localDst, height, width, line_pitch);
		break;

		default:
			/* median via histogramm algo */
		break;
	}


	return PF_Err_NONE;
}


PF_Err MedianFilter_BGRA_4444_16u
(
	PF_InData*   __restrict in_data,
	PF_OutData*  __restrict out_data,
	PF_ParamDef* __restrict params[],
	PF_LayerDef* __restrict output
) noexcept
{
	const PF_LayerDef*       __restrict pfLayer = reinterpret_cast<const PF_LayerDef* __restrict>(&params[MEDIAN_FILTER_INPUT]->u.ld);
	const PF_Pixel_BGRA_16u* __restrict localSrc = reinterpret_cast<const PF_Pixel_BGRA_16u* __restrict>(pfLayer->data);
	PF_Pixel_BGRA_16u*       __restrict localDst = reinterpret_cast<PF_Pixel_BGRA_16u* __restrict>(output->data);

	auto const height = pfLayer->extent_hint.bottom - pfLayer->extent_hint.top;
	auto const width  = pfLayer->extent_hint.right - pfLayer->extent_hint.left;
	auto const line_pitch = pfLayer->rowbytes / static_cast<A_long>(PF_Pixel_BGRA_16u_size);
	auto const kernelSize = get_kernel_size(params);

	switch (kernelSize)
	{
		case 3:
		/* manually optimized variant 3x3 */
		break;

		case 5:
		/* manually optimized variant 5x5 */
		break;

		case 7:
		/* manually optimized variant 7x7 */
		break;

		case 9:
			/* manually optimized variant 7x7 */
		break;

		default:
		/* median via histogramm algo */
		break;
	}

	return PF_Err_NONE;
}


PF_Err MedianFilter_BGRA_4444_32f
(
	PF_InData*   __restrict in_data,
	PF_OutData*  __restrict out_data,
	PF_ParamDef* __restrict params[],
	PF_LayerDef* __restrict output
) noexcept
{
	const PF_LayerDef*       __restrict pfLayer = reinterpret_cast<const PF_LayerDef* __restrict>(&params[MEDIAN_FILTER_INPUT]->u.ld);
	const PF_Pixel_BGRA_32f* __restrict localSrc = reinterpret_cast<const PF_Pixel_BGRA_32f* __restrict>(pfLayer->data);
	PF_Pixel_BGRA_32f*       __restrict localDst = reinterpret_cast<PF_Pixel_BGRA_32f* __restrict>(output->data);

	auto const height = pfLayer->extent_hint.bottom - pfLayer->extent_hint.top;
	auto const width  = pfLayer->extent_hint.right - pfLayer->extent_hint.left;
	auto const line_pitch = pfLayer->rowbytes / static_cast<A_long>(PF_Pixel_BGRA_32f_size);
	auto const kernelSize = get_kernel_size(params);

	switch (kernelSize)
	{
		case 3:
		/* manually optimized variant 3x3 */
		break;

		case 5:
		/* manually optimized variant 5x5 */
		break;

		case 7:
		/* manually optimized variant 7x7 */
		break;

		case 9:
		/* manually optimized variant 9x97 */
		break;

		default:
		/* median via histogramm algo */
		break;
	}

	return PF_Err_NONE;
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
	PF_Err errFormat = PF_Err_INVALID_INDEX;

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
