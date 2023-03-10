#include "ImageEqualization.hpp"
#include "ImageHistogram.hpp"
#include "ImageLUT.hpp"
#include "Avx2Histogram.hpp"


PF_Err PR_ImageEq_Linear_VUYA_4444_8u_709
(
	PF_InData*    in_data,
	PF_OutData*   out_data,
	PF_ParamDef*  params[],
	PF_LayerDef*  output
) noexcept
{
	constexpr int32_t histSize = 256;
	constexpr int32_t noiseLevel = 1;

	CACHE_ALIGN uint32_t histIn [histSize]{};
	CACHE_ALIGN uint32_t histOut[histSize]{};
	CACHE_ALIGN uint32_t lut[histSize]{};

	const PF_LayerDef*      __restrict pfLayer  = reinterpret_cast<const PF_LayerDef* __restrict>(&params[IMAGE_EQUALIZATION_FILTER_INPUT]->u.ld);
	const PF_Pixel_VUYA_8u* __restrict localSrc = reinterpret_cast<const PF_Pixel_VUYA_8u* __restrict>(pfLayer->data);
	PF_Pixel_VUYA_8u*       __restrict localDst = reinterpret_cast<PF_Pixel_VUYA_8u* __restrict>(output->data);

	auto const height = pfLayer->extent_hint.bottom - pfLayer->extent_hint.top;
	auto const width  = pfLayer->extent_hint.right  - pfLayer->extent_hint.left;
	auto const line_pitch = pfLayer->rowbytes / static_cast<A_long>(PF_Pixel_BGRA_8u_size);

	/* create histogram of the luminance channel */
	AVX2::Histogram::make_luma_histogram_VUYA4444_8u(localSrc, histIn, histSize, width, height, line_pitch);
	/* create histogram binarization and cumulative SUM */
	imgHistogramCumSum (histIn, histOut, noiseLevel, histSize);
	/* generate LUT */
	imgLinearLutGenerate (histOut, lut, histSize);
	/* apply LUT to the image */
	imgApplyLut (localSrc, localDst, lut, width, height, line_pitch);

	return PF_Err_NONE;
}


PF_Err PR_ImageEq_Linear_VUYA_4444_8u
(
	PF_InData*    in_data,
	PF_OutData*   out_data,
	PF_ParamDef*  params[],
	PF_LayerDef*  output
) noexcept
{
	return PR_ImageEq_Linear_VUYA_4444_8u_709(in_data, out_data, params, output);
}



PF_Err PR_ImageEq_Linear_BGRA_4444_8u
(
	PF_InData*    in_data,
	PF_OutData*   out_data,
	PF_ParamDef*  params[],
	PF_LayerDef*  output
) noexcept
{
	return PF_Err_NONE;
}

PF_Err PR_ImageEq_Linear_BGRA_4444_16u
(
	PF_InData*    in_data,
	PF_OutData*   out_data,
	PF_ParamDef*  params[],
	PF_LayerDef*  output
) noexcept
{
	return PF_Err_NONE;
}

PF_Err PR_ImageEq_Linear_BGRA_4444_32f
(
	PF_InData*    in_data,
	PF_OutData*   out_data,
	PF_ParamDef*  params[],
	PF_LayerDef*  output
) noexcept
{
	return PF_Err_NONE;
}


PF_Err PR_ImageEq_Linear_VUYA_4444_32f_709
(
	PF_InData*    in_data,
	PF_OutData*   out_data,
	PF_ParamDef*  params[],
	PF_LayerDef*  output
) noexcept
{
	return PF_Err_NONE;
}

PF_Err PR_ImageEq_Linear_VUYA_4444_32f
(
	PF_InData*    in_data,
	PF_OutData*   out_data,
	PF_ParamDef*  params[],
	PF_LayerDef*  output
) noexcept
{
	return PR_ImageEq_Linear_VUYA_4444_32f_709 (in_data, out_data, params, output);
}


PF_Err PR_ImageEq_Linear_RGB_444_10u
(
	PF_InData*    in_data,
	PF_OutData*   out_data,
	PF_ParamDef*  params[],
	PF_LayerDef*  output
) noexcept
{
	return PF_Err_NONE;
}