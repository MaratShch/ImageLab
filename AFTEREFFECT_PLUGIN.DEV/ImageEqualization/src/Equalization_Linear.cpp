#include "ImageEqualization.hpp"
#include "ImageHistogram.hpp"
#include "ImageLUT.hpp"
#include "Avx2Histogram.hpp"
#include "Avx2MiscUtils.hpp"


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
	CACHE_ALIGN uint32_t histBin[histSize]{};
	CACHE_ALIGN uint32_t cumSum [histSize]{};
	CACHE_ALIGN uint32_t lut    [histSize]{};

	const PF_LayerDef*      __restrict pfLayer  = reinterpret_cast<const PF_LayerDef* __restrict>(&params[IMAGE_EQUALIZATION_FILTER_INPUT]->u.ld);
	const PF_Pixel_VUYA_8u* __restrict localSrc = reinterpret_cast<const PF_Pixel_VUYA_8u* __restrict>(pfLayer->data);
	PF_Pixel_VUYA_8u*       __restrict localDst = reinterpret_cast<PF_Pixel_VUYA_8u* __restrict>(output->data);

	auto const height = pfLayer->extent_hint.bottom - pfLayer->extent_hint.top;
	auto const width  = pfLayer->extent_hint.right  - pfLayer->extent_hint.left;
	auto const line_pitch = pfLayer->rowbytes / static_cast<A_long>(PF_Pixel_BGRA_8u_size);

	/* create histogram of the luminance channel */
	AVX2::Histogram::make_luma_histogram_VUYA4444_8u(localSrc, histIn, histSize, width, height, line_pitch);
	/* make histogram binarization */
	AVX2::Histogram::make_histogram_binarization(histIn, histBin, histSize, noiseLevel);
	/* make cumulative SUM of binary histogram elements */
	AVX2::MiscUtils::cum_sum_uint32(histBin, histOut, histSize);
	/* generate LUT */
	constexpr int32_t lastHistElem = histSize - 1;
	const float coeff = static_cast<float>(lastHistElem) / static_cast<float>(histOut[lastHistElem]);
	AVX2::MiscUtils::generate_lut_uint32(histOut, lut, coeff, histSize);
	/* apply LUT to the image */
	imgApplyLut (localSrc, localDst, lut, width, height, line_pitch, line_pitch);

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

PF_Err PR_ImageEq_Linear_VUYA_4444_32f_709
(
	PF_InData*    in_data,
	PF_OutData*   out_data,
	PF_ParamDef*  params[],
	PF_LayerDef*  output
) noexcept
{
	constexpr uint32_t histSize = static_cast<int32_t>(u16_value_white + 1);
	constexpr uint32_t noiseLevel = 1;

	uint32_t histIn[histSize]  = {};
	uint32_t histOut[histSize] = {};
	uint32_t histBin[histSize] = {};
	uint32_t cumSum[histSize]  = {};
	float    lut[histSize]     = {};

	const PF_LayerDef*       __restrict pfLayer  = reinterpret_cast<const PF_LayerDef* __restrict>(&params[IMAGE_EQUALIZATION_FILTER_INPUT]->u.ld);
	const PF_Pixel_VUYA_32f* __restrict localSrc = reinterpret_cast<const PF_Pixel_VUYA_32f* __restrict>(pfLayer->data);
	      PF_Pixel_VUYA_32f* __restrict localDst = reinterpret_cast<PF_Pixel_VUYA_32f* __restrict>(output->data);

	auto const height = pfLayer->extent_hint.bottom - pfLayer->extent_hint.top;
	auto const width  = pfLayer->extent_hint.right  - pfLayer->extent_hint.left;
	auto const line_pitch = pfLayer->rowbytes / static_cast<A_long>(PF_Pixel_BGRA_32f_size);

	/* create histogram of the luminance channel [32 floating point pixel values converted to integer] */
	AVX2::Histogram::make_luma_histogram_VUYA4444_32f(localSrc, histIn, histSize, width, height, line_pitch);
	/* make histogram binarization */
	AVX2::Histogram::make_histogram_binarization(histIn, histBin, histSize, noiseLevel);
	/* make cumulative SUM of binary histogram elements */
	AVX2::MiscUtils::cum_sum_uint32(histBin, histOut, histSize);
	/* generate LUT */
	constexpr int32_t lastHistElem = histSize - 1;
	const float coeff = static_cast<float>(lastHistElem) / static_cast<float>(histOut[lastHistElem]);
	AVX2::MiscUtils::generate_lut_float32 (histOut, lut, coeff, histSize);

	/* apply LUT to the image */
	imgApplyLut(localSrc, localDst, lut, width, height, line_pitch, line_pitch);

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
	return PR_ImageEq_Linear_VUYA_4444_32f_709(in_data, out_data, params, output);
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