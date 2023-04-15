#include "ImageEqualization.hpp"
#include "ImageHistogram.hpp"
#include "ImageLUT.hpp"
#include "FastAriphmetics.hpp"
#include "Avx2Histogram.hpp"
#include "Avx2MiscUtils.hpp"
#include "ImageLabUtils.hpp"
#include "ColorTransform.hpp"


PF_Err PR_ImageEq_Linear_VUYA_4444_8u_709
(
	PF_InData*    in_data,
	PF_OutData*   out_data,
	PF_ParamDef*  params[],
	PF_LayerDef*  output
) noexcept
{
	constexpr int32_t histSize = u8_value_white + 1;
	constexpr int32_t noiseLevel = 1;

	CACHE_ALIGN uint32_t histIn [histSize]{};
	CACHE_ALIGN uint32_t histOut[histSize]{};
	CACHE_ALIGN uint32_t histBin[histSize]{};
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

	const PF_LayerDef*       __restrict pfLayer  = reinterpret_cast<const PF_LayerDef* __restrict>(&params[IMAGE_EQUALIZATION_FILTER_INPUT]->u.ld);
	const PF_Pixel_VUYA_32f* __restrict localSrc = reinterpret_cast<const PF_Pixel_VUYA_32f* __restrict>(pfLayer->data);
	      PF_Pixel_VUYA_32f* __restrict localDst = reinterpret_cast<PF_Pixel_VUYA_32f* __restrict>(output->data);

	auto const height = pfLayer->extent_hint.bottom - pfLayer->extent_hint.top;
	auto const width  = pfLayer->extent_hint.right  - pfLayer->extent_hint.left;
	auto const line_pitch = pfLayer->rowbytes / static_cast<A_long>(PF_Pixel_BGRA_32f_size);

	CACHE_ALIGN uint32_t histBin[histSize] = {};
	CACHE_ALIGN float    lut    [histSize] = {};

	{
		CACHE_ALIGN uint32_t histIn[histSize] = {};
		/* create histogram of the luminance channel [32 floating point pixel values converted to integer] */
		AVX2::Histogram::make_luma_histogram_VUYA4444_32f(localSrc, histIn, histSize, width, height, line_pitch);
		/* make histogram binarization */
		AVX2::Histogram::make_histogram_binarization(histIn, histBin, histSize, noiseLevel);
	}
	{
		CACHE_ALIGN uint32_t histOut[histSize] = {};
		/* make cumulative SUM of binary histogram elements */
		AVX2::MiscUtils::cum_sum_uint32(histBin, histOut, histSize);
		/* generate LUT */
		constexpr int32_t lastHistElem = histSize - 1;
		const float coeff = static_cast<float>(lastHistElem) / static_cast<float>(histOut[lastHistElem]);
		AVX2::MiscUtils::generate_lut_float32(histOut, lut, coeff, histSize);
	}
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
	PF_Err errCode = PF_Err_OUT_OF_MEMORY;
	constexpr int32_t histSize = u8_value_white + 1;
	constexpr int32_t noiseLevel = 1;

	CACHE_ALIGN uint32_t histIn [histSize]{};
	CACHE_ALIGN uint32_t histOut[histSize]{};
	CACHE_ALIGN uint32_t histBin[histSize]{};
	CACHE_ALIGN uint32_t lut    [histSize]{};

	const PF_LayerDef*      __restrict pfLayer  = reinterpret_cast<const PF_LayerDef* __restrict>(&params[IMAGE_EQUALIZATION_FILTER_INPUT]->u.ld);
	const PF_Pixel_BGRA_8u* __restrict localSrc = reinterpret_cast<const PF_Pixel_BGRA_8u* __restrict>(pfLayer->data);
	      PF_Pixel_BGRA_8u* __restrict localDst = reinterpret_cast<      PF_Pixel_BGRA_8u* __restrict>(output->data);

	auto const height = pfLayer->extent_hint.bottom - pfLayer->extent_hint.top;
	auto const width  = pfLayer->extent_hint.right  - pfLayer->extent_hint.left;
	auto const line_pitch = pfLayer->rowbytes / static_cast<A_long>(PF_Pixel_BGRA_8u_size);

	const size_t frameSize = height * FastCompute::Abs(line_pitch);
	const size_t requiredMemSize = CreateAlignment(frameSize * PF_Pixel_BGRA_8u_size, static_cast<size_t>(CACHE_LINE));

	/* Get memory block */
	void* pMemoryBlock = nullptr;
	const int32_t blockId = ::GetMemoryBlock(requiredMemSize, 0, &pMemoryBlock);

	if (-1 != blockId && nullptr != pMemoryBlock)
	{
		PF_Pixel_VUYA_8u* __restrict pSrcYUV = reinterpret_cast<PF_Pixel_VUYA_8u* __restrict>(pMemoryBlock);
		/* let's chek if we have negative line pitch */
		const int32_t startOffset = ((line_pitch < 0) ? frameSize + line_pitch : 0);
		pSrcYUV += startOffset;

		/* convert BGRA image to VUYA image format */
		AVX2::ColorConvert::BGRA8u_to_VUYA8u (localSrc, pSrcYUV, width, height, line_pitch);
		/* create histogram of the luminance channel */
		AVX2::Histogram::make_luma_histogram_VUYA4444_8u (pSrcYUV, histIn, histSize, width, height, line_pitch);
		/* make histogram binarization */
		AVX2::Histogram::make_histogram_binarization(histIn, histBin, histSize, noiseLevel);
		/* make cumulative SUM of binary histogram elements */
		AVX2::MiscUtils::cum_sum_uint32(histBin, histOut, histSize);
		/* generate LUT */
		constexpr int32_t lastHistElem = histSize - 1;
		const float coeff = static_cast<float>(lastHistElem) / static_cast<float>(histOut[lastHistElem]);
		AVX2::MiscUtils::generate_lut_uint32(histOut, lut, coeff, histSize);

		/* apply LUT to the image */
		imgApplyLut (pSrcYUV, localDst, lut, width, height, line_pitch, line_pitch);

		::FreeMemoryBlock (blockId);
		errCode = PF_Err_NONE;
	}/* if (nullptr != pMemInstance) */

	return errCode;
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


PF_Err AE_ImageEq_Linear_ARGB_4444_8u
(
	PF_InData*    in_data,
	PF_OutData*   out_data,
	PF_ParamDef*  params[],
	PF_LayerDef*  output
) noexcept
{
	constexpr int32_t histSize = u8_value_white + 1;
	constexpr int32_t noiseLevel = 1;
	CACHE_ALIGN uint32_t histIn [histSize]{};
	CACHE_ALIGN uint32_t histOut[histSize]{};
	CACHE_ALIGN uint32_t histBin[histSize]{};
	CACHE_ALIGN uint32_t lut    [histSize]{};

	PF_Err errCode = PF_Err_OUT_OF_MEMORY;
	const PF_EffectWorld*    __restrict input    = reinterpret_cast<const PF_EffectWorld* __restrict>(&params[IMAGE_EQUALIZATION_FILTER_INPUT]->u.ld);
	const PF_Pixel_ARGB_8u*  __restrict localSrc = reinterpret_cast<const PF_Pixel_ARGB_8u* __restrict>(input->data);
	      PF_Pixel_ARGB_8u*  __restrict localDst = reinterpret_cast<      PF_Pixel_ARGB_8u* __restrict>(output->data);

	auto const& src_line_pitch = input->rowbytes  / static_cast<A_long>(PF_Pixel_ARGB_8u_size);
	auto const& dst_line_pitch = output->rowbytes / static_cast<A_long>(PF_Pixel_ARGB_8u_size);

    auto const& height = output->height;
	auto const& width  = output->width;

	const int32_t yuv_line_pitch = src_line_pitch;
	const size_t frameSize = height * FastCompute::Abs(yuv_line_pitch);
	const size_t requiredMemSize = CreateAlignment (frameSize * PF_Pixel_VUYA_8u_size, static_cast<size_t>(CACHE_LINE));

	/* Get memory block */
	void* pMemoryBlock = nullptr;
	const int32_t blockId = ::GetMemoryBlock (requiredMemSize, 0, &pMemoryBlock);

	if (-1 != blockId && nullptr != pMemoryBlock)
	{
		PF_Pixel_VUYA_8u* __restrict pSrcYUV = reinterpret_cast<PF_Pixel_VUYA_8u* __restrict>(pMemoryBlock);
		/* let's chek if we have negative line pitch */
		const int32_t startOffset = ((src_line_pitch < 0) ? frameSize + src_line_pitch : 0);
		pSrcYUV += startOffset;

		constexpr int32_t convert_addendum = 128;
		imgRGB2YUV (localSrc, pSrcYUV, BT709, width, height, src_line_pitch, yuv_line_pitch, convert_addendum);

		/* create histogram of the luminance channel */
		AVX2::Histogram::make_luma_histogram_VUYA4444_8u(pSrcYUV, histIn, histSize, width, height, src_line_pitch);
		/* make histogram binarization */
		AVX2::Histogram::make_histogram_binarization(histIn, histBin, histSize, noiseLevel);
		/* make cumulative SUM of binary histogram elements */
		AVX2::MiscUtils::cum_sum_uint32 (histBin, histOut, histSize);
		/* generate LUT */
		constexpr int32_t lastHistElem = histSize - 1;
		const float coeff = static_cast<float>(lastHistElem) / static_cast<float>(histOut[lastHistElem]);
		AVX2::MiscUtils::generate_lut_uint32 (histOut, lut, coeff, histSize);

		/* apply LUT to the image */
		imgApplyLut (pSrcYUV, localDst, lut, width, height, yuv_line_pitch, dst_line_pitch, convert_addendum);

		::FreeMemoryBlock(blockId);
		errCode = PF_Err_NONE;
	}

	return errCode;
}


PF_Err AE_ImageEq_Linear_ARGB_4444_16u
(
	PF_InData*    in_data,
	PF_OutData*   out_data,
	PF_ParamDef*  params[],
	PF_LayerDef*  output
) noexcept
{
	constexpr int32_t histSize = u16_value_white + 1;
	constexpr int32_t noiseLevel = 1;

	CACHE_ALIGN uint32_t histIn [histSize]{};
	CACHE_ALIGN uint32_t histOut[histSize]{};
	CACHE_ALIGN uint32_t histBin[histSize]{};
	CACHE_ALIGN uint32_t lut    [histSize]{};

	PF_Err errCode = PF_Err_OUT_OF_MEMORY;
	const PF_EffectWorld*    __restrict input    = reinterpret_cast<const PF_EffectWorld* __restrict>(&params[IMAGE_EQUALIZATION_FILTER_INPUT]->u.ld);
	const PF_Pixel_ARGB_16u* __restrict localSrc = reinterpret_cast<const PF_Pixel_ARGB_16u* __restrict>(input->data);
	      PF_Pixel_ARGB_16u* __restrict localDst = reinterpret_cast<      PF_Pixel_ARGB_16u* __restrict>(output->data);

	auto const& src_line_pitch = input->rowbytes  / static_cast<A_long>(PF_Pixel_ARGB_16u_size);
	auto const& dst_line_pitch = output->rowbytes / static_cast<A_long>(PF_Pixel_ARGB_16u_size);

	auto const& height = output->height;
	auto const& width  = output->width;

	const int32_t yuv_line_pitch = src_line_pitch;
	const size_t frameSize = height * FastCompute::Abs(yuv_line_pitch);
	const size_t requiredMemSize = CreateAlignment(frameSize * PF_Pixel_VUYA_16u_size, static_cast<size_t>(CACHE_LINE));

	/* Get memory block */
	void* pMemoryBlock = nullptr;
	const int32_t blockId = ::GetMemoryBlock (requiredMemSize, 0, &pMemoryBlock);

	if (-1 != blockId && nullptr != pMemoryBlock)
	{
		PF_Pixel_VUYA_16u* __restrict pSrcYUV = reinterpret_cast<PF_Pixel_VUYA_16u* __restrict>(pMemoryBlock);
		/* let's chek if we have negative line pitch */
		const int32_t startOffset = ((src_line_pitch < 0) ? frameSize + src_line_pitch : 0);
		pSrcYUV += startOffset;

		constexpr int32_t convert_addendum = 16384;
		imgRGB2YUV(localSrc, pSrcYUV, BT709, width, height, src_line_pitch, yuv_line_pitch, convert_addendum);

		/* create histogram of the luminance channel (require AVX2 implementation in future) */
		imgHistogram (pSrcYUV, width, height, src_line_pitch, histIn);
		/* make histogram binarization */
		AVX2::Histogram::make_histogram_binarization (histIn, histBin, histSize, noiseLevel);
		/* make cumulative SUM of binary histogram elements */
		AVX2::MiscUtils::cum_sum_uint32 (histBin, histOut, histSize);
		/* generate LUT */
		constexpr int32_t lastHistElem = histSize - 1;
		const float coeff = static_cast<float>(lastHistElem) / static_cast<float>(histOut[lastHistElem]);
		AVX2::MiscUtils::generate_lut_uint32(histOut, lut, coeff, histSize);

		/* apply LUT to the image */
		imgApplyLut (pSrcYUV, localDst, lut, width, height, yuv_line_pitch, dst_line_pitch, convert_addendum);

		::FreeMemoryBlock(blockId);
		errCode = PF_Err_NONE;
	}

	return errCode;
}
