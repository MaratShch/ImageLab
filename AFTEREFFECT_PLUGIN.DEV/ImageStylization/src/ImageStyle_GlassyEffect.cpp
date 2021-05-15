#include "ImageStylization.hpp"
#include "PrSDKAESupport.h"


template <typename T>
inline PF_Err ImageStyle_GlassyEffect
(
	const T* __restrict srcBuffer,
	      T* __restrict dstBuffer,
	const A_long&       height,
	const A_long&       width,
	const A_long&       src_line_pitch,
	const A_long&       dst_line_pitch,
	const A_long&       dispersionSliderValue
) noexcept
{
	uint32_t randomBufSize = 0u;
	const float* __restrict pRandom = get_random_buffer(randomBufSize);
	const uint32_t randomBufMask = randomBufSize - 1u;
	float const& imgDispersion = static_cast<float const>(dispersionSliderValue);

	const A_long& short_height = height - dispersionSliderValue;
	const A_long& short_width  = width - dispersionSliderValue;

	int32_t i, j, idx;

	for (idx = j = 0; j < height; j++)
	{
		const auto& src_line_idx = src_line_pitch * j;
		const auto& dst_line_idx = dst_line_pitch * j;

		__VECTOR_ALIGNED__
		for (i = 0; i < width; i++)
		{
			const auto& src_pix_idx = src_line_idx + i;
			const auto& dst_pix_idx = dst_line_idx + i;

			const float& random1 = pRandom[idx++];
			idx &= randomBufMask;

			const float& random2 = pRandom[idx++];
			idx &= randomBufMask;

			const auto& xIdx = static_cast<int32_t>(random1 * (i < short_width  ? imgDispersion : (width  - i)));
			const auto& yIdx = static_cast<int32_t>(random2 * (j < short_height ? imgDispersion : (height - j)));

			const auto& src_pix_offset = src_pix_idx + yIdx * src_line_pitch + xIdx;
			const auto& dst_pix_offset = dst_pix_idx;

			dstBuffer[dst_pix_offset] = srcBuffer[src_pix_offset];
		} /* for (i = 0; i < width; i++) */
	}
	
	return PF_Err_NONE;
}


static PF_Err PR_ImageStyle_GlassyEffect_BGRA_8u
(
	PF_InData*   __restrict in_data,
	PF_OutData*  __restrict out_data,
	PF_ParamDef* __restrict params[],
	PF_LayerDef* __restrict output
) noexcept
{
	const PF_LayerDef*       __restrict pfLayer = reinterpret_cast<const PF_LayerDef* __restrict>(&params[IMAGE_STYLE_INPUT]->u.ld);
	const PF_Pixel_BGRA_8u*  __restrict localSrc = reinterpret_cast<const PF_Pixel_BGRA_8u* __restrict>(pfLayer->data);
	PF_Pixel_BGRA_8u*        __restrict localDst = reinterpret_cast<PF_Pixel_BGRA_8u* __restrict>(output->data);

	auto const& height = pfLayer->extent_hint.bottom - pfLayer->extent_hint.top;
	auto const& width  = pfLayer->extent_hint.right  - pfLayer->extent_hint.left;
	auto const& line_pitch = pfLayer->rowbytes / static_cast<A_long>(PF_Pixel_BGRA_8u_size);

	auto const& dispersionSliderValue = getDispersionSliderValue(params[IMAGE_STYLE_SLIDER1]->u.sd.value);

	if (dispersionSliderValue < height && dispersionSliderValue < width)
		ImageStyle_GlassyEffect(localSrc, localDst, height, width, line_pitch, line_pitch, dispersionSliderValue);
	else /* ROI buffer to small - make simple copy */
		Image_SimpleCopy(localSrc, localDst, height, width, line_pitch, line_pitch);

	return PF_Err_NONE;
}


static PF_Err PR_ImageStyleGlassyEffect_VUYA_8u
(
	PF_InData*   __restrict in_data,
	PF_OutData*  __restrict out_data,
	PF_ParamDef* __restrict params[],
	PF_LayerDef* __restrict output
) noexcept
{
	const PF_LayerDef*       __restrict pfLayer = reinterpret_cast<const PF_LayerDef* __restrict>(&params[IMAGE_STYLE_INPUT]->u.ld);
	const PF_Pixel_VUYA_8u*  __restrict localSrc = reinterpret_cast<const PF_Pixel_VUYA_8u* __restrict>(pfLayer->data);
	PF_Pixel_VUYA_8u*        __restrict localDst = reinterpret_cast<PF_Pixel_VUYA_8u* __restrict>(output->data);

	auto const& height = pfLayer->extent_hint.bottom - pfLayer->extent_hint.top;
	auto const& width  = pfLayer->extent_hint.right  - pfLayer->extent_hint.left;
	auto const& line_pitch = pfLayer->rowbytes / static_cast<A_long>(PF_Pixel_VUYA_8u_size);

	auto const& dispersionSliderValue = getDispersionSliderValue(params[IMAGE_STYLE_SLIDER1]->u.sd.value);

	if (dispersionSliderValue < height && dispersionSliderValue < width)
		ImageStyle_GlassyEffect(localSrc, localDst, height, width, line_pitch, line_pitch, dispersionSliderValue);
	else /* ROI buffer to small - make simple copy */
		Image_SimpleCopy(localSrc, localDst, height, width, line_pitch, line_pitch);

	return PF_Err_NONE;
}


static PF_Err PR_ImageStyle_GlassyEffect_VUYA_32f
(
	PF_InData*   __restrict in_data,
	PF_OutData*  __restrict out_data,
	PF_ParamDef* __restrict params[],
	PF_LayerDef* __restrict output
) noexcept
{
	const PF_LayerDef*       __restrict pfLayer = reinterpret_cast<const PF_LayerDef* __restrict>(&params[IMAGE_STYLE_INPUT]->u.ld);
	const PF_Pixel_VUYA_32f* __restrict localSrc = reinterpret_cast<const PF_Pixel_VUYA_32f* __restrict>(pfLayer->data);
	PF_Pixel_VUYA_32f*        __restrict localDst = reinterpret_cast<PF_Pixel_VUYA_32f* __restrict>(output->data);

	auto const& height = pfLayer->extent_hint.bottom - pfLayer->extent_hint.top;
	auto const& width  = pfLayer->extent_hint.right  - pfLayer->extent_hint.left;
	auto const& line_pitch = pfLayer->rowbytes / static_cast<A_long>(PF_Pixel_VUYA_32f_size);

	auto const& dispersionSliderValue = getDispersionSliderValue(params[IMAGE_STYLE_SLIDER1]->u.sd.value);

	if (dispersionSliderValue < height && dispersionSliderValue < width)
		ImageStyle_GlassyEffect(localSrc, localDst, height, width, line_pitch, line_pitch, dispersionSliderValue);
	else /* ROI buffer to small - make simple copy */
		Image_SimpleCopy(localSrc, localDst, height, width, line_pitch, line_pitch);

	return PF_Err_NONE;
}


static PF_Err PR_ImageStyle_GlassyEffect_BGRA_16u
(
	PF_InData*   __restrict in_data,
	PF_OutData*  __restrict out_data,
	PF_ParamDef* __restrict params[],
	PF_LayerDef* __restrict output
) noexcept
{
	const PF_LayerDef*       __restrict pfLayer = reinterpret_cast<const PF_LayerDef* __restrict>(&params[IMAGE_STYLE_INPUT]->u.ld);
	const PF_Pixel_BGRA_16u* __restrict localSrc = reinterpret_cast<const PF_Pixel_BGRA_16u* __restrict>(pfLayer->data);
	PF_Pixel_BGRA_16u*       __restrict localDst = reinterpret_cast<PF_Pixel_BGRA_16u* __restrict>(output->data);

	auto const& height = pfLayer->extent_hint.bottom - pfLayer->extent_hint.top;
	auto const& width  = pfLayer->extent_hint.right  - pfLayer->extent_hint.left;
	auto const& line_pitch = pfLayer->rowbytes / static_cast<A_long>(PF_Pixel_BGRA_16u_size);

	auto const& dispersionSliderValue = getDispersionSliderValue(params[IMAGE_STYLE_SLIDER1]->u.sd.value);

	if (dispersionSliderValue < height && dispersionSliderValue < width)
		ImageStyle_GlassyEffect(localSrc, localDst, height, width, line_pitch, line_pitch, dispersionSliderValue);
	else /* ROI buffer to small - make simple copy */
		Image_SimpleCopy(localSrc, localDst, height, width, line_pitch, line_pitch);

	return PF_Err_NONE;
}


static PF_Err PR_ImageStyle_GlassyEffect_BGRA_32f
(
	PF_InData*   __restrict in_data,
	PF_OutData*  __restrict out_data,
	PF_ParamDef* __restrict params[],
	PF_LayerDef* __restrict output
) noexcept
{
	const PF_LayerDef*       __restrict pfLayer = reinterpret_cast<const PF_LayerDef* __restrict>(&params[IMAGE_STYLE_INPUT]->u.ld);
	const PF_Pixel_BGRA_32f* __restrict localSrc = reinterpret_cast<const PF_Pixel_BGRA_32f* __restrict>(pfLayer->data);
	PF_Pixel_BGRA_32f*        __restrict localDst = reinterpret_cast<PF_Pixel_BGRA_32f* __restrict>(output->data);

	auto const& height = pfLayer->extent_hint.bottom - pfLayer->extent_hint.top;
	auto const& width  = pfLayer->extent_hint.right  - pfLayer->extent_hint.left;
	auto const& line_pitch = pfLayer->rowbytes / static_cast<A_long>(PF_Pixel_BGRA_32f_size);

	auto const& dispersionSliderValue = getDispersionSliderValue(params[IMAGE_STYLE_SLIDER1]->u.sd.value);

	if (dispersionSliderValue < height && dispersionSliderValue < width)
		ImageStyle_GlassyEffect(localSrc, localDst, height, width, line_pitch, line_pitch, dispersionSliderValue);
	else /* ROI buffer to small - make simple copy */
		Image_SimpleCopy(localSrc, localDst, height, width, line_pitch, line_pitch);

	return PF_Err_NONE;
}




PF_Err PR_ImageStyle_GlassyEffect
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
				err = PR_ImageStyle_GlassyEffect_BGRA_8u (in_data, out_data, params, output);
			break;

			case PrPixelFormat_VUYA_4444_8u_709:
			case PrPixelFormat_VUYA_4444_8u:
				err = PR_ImageStyleGlassyEffect_VUYA_8u (in_data, out_data, params, output);
			break;

			case PrPixelFormat_VUYA_4444_32f_709:
			case PrPixelFormat_VUYA_4444_32f:
				err = PR_ImageStyle_GlassyEffect_VUYA_32f (in_data, out_data, params, output);
			break;

			case PrPixelFormat_BGRA_4444_16u:
				err = PR_ImageStyle_GlassyEffect_BGRA_16u (in_data, out_data, params, output);
			break;

			case PrPixelFormat_BGRA_4444_32f:
				err = PR_ImageStyle_GlassyEffect_BGRA_32f (in_data, out_data, params, output);
			break;

			default:
				err = PF_Err_INVALID_INDEX;
			break;
		}
	}
	else
	{
		err = PF_Err_UNRECOGNIZED_PARAM_TYPE;
	}

	return err;
}



PF_Err AE_ImageStyle_GlassyEffect_ARGB_8u
(
	PF_InData*   __restrict in_data,
	PF_OutData*  __restrict out_data,
	PF_ParamDef* __restrict params[],
	PF_LayerDef* __restrict output
) noexcept
{
	const PF_EffectWorld* __restrict input = reinterpret_cast<const PF_EffectWorld* __restrict>(&params[IMAGE_STYLE_INPUT]->u.ld);
	PF_Pixel_ARGB_8u*     __restrict localSrc = reinterpret_cast<PF_Pixel_ARGB_8u* __restrict>(input->data);
	PF_Pixel_ARGB_8u*     __restrict localDst = reinterpret_cast<PF_Pixel_ARGB_8u* __restrict>(output->data);

	const A_long& height = output->height;
	const A_long& width = output->width;
	const A_long& src_line_pitch = input->rowbytes  / static_cast<A_long>(PF_Pixel_ARGB_8u_size);
	const A_long& dst_line_pitch = output->rowbytes / static_cast<A_long>(PF_Pixel_ARGB_8u_size);

	auto const& dispersionSliderValue = getDispersionSliderValue(params[IMAGE_STYLE_SLIDER1]->u.sd.value);

	if (dispersionSliderValue < height && dispersionSliderValue < width)
		ImageStyle_GlassyEffect(localSrc, localDst, height, width, src_line_pitch, dst_line_pitch, dispersionSliderValue);
	else /* ROI buffer to small - make simple copy */
		Image_SimpleCopy(localSrc, localDst, height, width, src_line_pitch, dst_line_pitch);

	return PF_Err_NONE;
}


PF_Err AE_ImageStyle_GlassyEffect_ARGB_16u
(
	PF_InData*   __restrict in_data,
	PF_OutData*  __restrict out_data,
	PF_ParamDef* __restrict params[],
	PF_LayerDef* __restrict output
) noexcept
{
	const PF_EffectWorld* __restrict input = reinterpret_cast<const PF_EffectWorld* __restrict>(&params[IMAGE_STYLE_INPUT]->u.ld);
	PF_Pixel_ARGB_16u*    __restrict localSrc = reinterpret_cast<PF_Pixel_ARGB_16u* __restrict>(input->data);
	PF_Pixel_ARGB_16u*    __restrict localDst = reinterpret_cast<PF_Pixel_ARGB_16u* __restrict>(output->data);

	const A_long& height = output->height;
	const A_long& width = output->width;
	const A_long& src_line_pitch = input->rowbytes  / static_cast<A_long>(PF_Pixel_ARGB_16u_size);
	const A_long& dst_line_pitch = output->rowbytes / static_cast<A_long>(PF_Pixel_ARGB_16u_size);

	auto const& dispersionSliderValue = getDispersionSliderValue(params[IMAGE_STYLE_SLIDER1]->u.sd.value);

	if (dispersionSliderValue < height && dispersionSliderValue < width)
		ImageStyle_GlassyEffect(localSrc, localDst, height, width, src_line_pitch, dst_line_pitch, dispersionSliderValue);
	else /* ROI buffer to small - make simple copy */
		Image_SimpleCopy(localSrc, localDst, height, width, src_line_pitch, dst_line_pitch);

	return PF_Err_NONE;
}