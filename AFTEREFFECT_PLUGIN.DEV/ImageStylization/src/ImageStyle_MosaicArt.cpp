#include "ImageStylization.hpp"
#include "StylizationStructs.hpp"
#include "PrSDKAESupport.h"
#include "ColorTransformMatrix.hpp"
#include "FastAriphmetics.hpp"
#include "SegmentationUtils.hpp"
#include "ImageAuxPixFormat.hpp"
#include "ImageMosaicUtils.hpp"


static PF_Err PR_ImageStyle_MosaicArt_BGRA_8u
(
	PF_InData*   __restrict in_data,
	PF_OutData*  __restrict out_data,
	PF_ParamDef* __restrict params[],
	PF_LayerDef* __restrict output
) noexcept
{
	const PF_LayerDef* __restrict pfLayer  = reinterpret_cast<const PF_LayerDef* __restrict>(&params[IMAGE_STYLE_INPUT]->u.ld);
	PF_Pixel_BGRA_8u*  __restrict localSrc = reinterpret_cast<PF_Pixel_BGRA_8u*  __restrict>(pfLayer->data);
	PF_Pixel_BGRA_8u*  __restrict localDst = reinterpret_cast<PF_Pixel_BGRA_8u*  __restrict>(output->data);

	auto const height = pfLayer->extent_hint.bottom - pfLayer->extent_hint.top;
	auto const width  = pfLayer->extent_hint.right  - pfLayer->extent_hint.left;
	auto const line_pitch = pfLayer->rowbytes / static_cast<A_long>(PF_Pixel_BGRA_8u_size);

	/* parameters */
	const float m = 40.f;
	A_long k = 1000, g = 0;
	constexpr int maxNorm = std::numeric_limits<unsigned char>::max();
	const ArtMosaic::Color WhiteColor(maxNorm, maxNorm, maxNorm);
	const ArtMosaic::Color GrayColor (maxNorm / 2, maxNorm / 2, maxNorm / 2);

	const bool bRet = ArtMosaic::SlicImage (localSrc, localDst, GrayColor, WhiteColor, m, k, g, width, height, line_pitch, line_pitch);

	return (true == bRet ? PF_Err_NONE : PF_Err_INVALID_INDEX);
}


static PF_Err PR_ImageStyle_MosaicArt_BGRA_16u
(
	PF_InData*   __restrict in_data,
	PF_OutData*  __restrict out_data,
	PF_ParamDef* __restrict params[],
	PF_LayerDef* __restrict output
) noexcept
{
	const PF_LayerDef* __restrict pfLayer  = reinterpret_cast<const PF_LayerDef* __restrict>(&params[IMAGE_STYLE_INPUT]->u.ld);
	PF_Pixel_BGRA_16u* __restrict localSrc = reinterpret_cast<PF_Pixel_BGRA_16u* __restrict>(pfLayer->data);
	PF_Pixel_BGRA_16u* __restrict localDst = reinterpret_cast<PF_Pixel_BGRA_16u* __restrict>(output->data);

	auto const height = pfLayer->extent_hint.bottom - pfLayer->extent_hint.top;
	auto const width  = pfLayer->extent_hint.right  - pfLayer->extent_hint.left;
	auto const line_pitch = pfLayer->rowbytes / static_cast<A_long>(PF_Pixel_BGRA_16u_size);

	/* parameters */
	const float m = 40.f;
	A_long k = 1000, g = 0;
	constexpr int maxNorm = std::numeric_limits<short int>::max();
	const ArtMosaic::Color WhiteColor(maxNorm, maxNorm, maxNorm);
	const ArtMosaic::Color GrayColor (maxNorm / 2, maxNorm / 2, maxNorm / 2);

	const bool bRet = ArtMosaic::SlicImage (localSrc, localDst, GrayColor, WhiteColor, m, k, g, width, height, line_pitch, line_pitch);

	return (true == bRet ? PF_Err_NONE : PF_Err_INVALID_INDEX);
}


static PF_Err PR_ImageStyle_MosaicArt_BGRA_32f
(
	PF_InData*   __restrict in_data,
	PF_OutData*  __restrict out_data,
	PF_ParamDef* __restrict params[],
	PF_LayerDef* __restrict output
) noexcept
{
	const PF_LayerDef* __restrict pfLayer  = reinterpret_cast<const PF_LayerDef* __restrict>(&params[IMAGE_STYLE_INPUT]->u.ld);
	PF_Pixel_BGRA_32f* __restrict localSrc = reinterpret_cast<PF_Pixel_BGRA_32f* __restrict>(pfLayer->data);
	PF_Pixel_BGRA_32f* __restrict localDst = reinterpret_cast<PF_Pixel_BGRA_32f* __restrict>(output->data);

	auto const height = pfLayer->extent_hint.bottom - pfLayer->extent_hint.top;
	auto const width  = pfLayer->extent_hint.right  - pfLayer->extent_hint.left;
	auto const line_pitch = pfLayer->rowbytes / static_cast<A_long>(PF_Pixel_BGRA_32f_size);

	/* parameters */
	const float m = 40.f;
	A_long k = 1000, g = 0;
	constexpr float maxNorm = f32_value_white;
	const ArtMosaic::Color WhiteColor(maxNorm, maxNorm, maxNorm);
	const ArtMosaic::Color GrayColor (maxNorm / 2.f, maxNorm / 2.f, maxNorm / 2.f);

	const bool bRet = ArtMosaic::SlicImage (localSrc, localDst, GrayColor, WhiteColor, m, k, g, width, height, line_pitch, line_pitch);

	return (true == bRet ? PF_Err_NONE : PF_Err_INVALID_INDEX);
	return PF_Err_NONE;
}



static PF_Err PR_ImageStyle_MosaicArt_VUYA_8u
(
	PF_InData*   __restrict in_data,
	PF_OutData*  __restrict out_data,
	PF_ParamDef* __restrict params[],
	PF_LayerDef* __restrict output
) noexcept
{
	const PF_LayerDef* __restrict pfLayer = reinterpret_cast<const PF_LayerDef* __restrict>(&params[IMAGE_STYLE_INPUT]->u.ld);
	PF_Pixel_VUYA_8u*  __restrict localSrc = reinterpret_cast<PF_Pixel_VUYA_8u* __restrict>(pfLayer->data);
	PF_Pixel_VUYA_8u*  __restrict localDst = reinterpret_cast<PF_Pixel_VUYA_8u* __restrict>(output->data);

	auto const height = pfLayer->extent_hint.bottom - pfLayer->extent_hint.top;
	auto const width = pfLayer->extent_hint.right - pfLayer->extent_hint.left;
	auto const line_pitch = pfLayer->rowbytes / static_cast<A_long>(PF_Pixel_VUYA_8u_size);

	/* parameters */
	const float m = 40.f;
	A_long k = 1000, g = 0;
	constexpr int maxNorm = std::numeric_limits<unsigned char>::max();
	const ArtMosaic::Color WhiteColor(maxNorm, maxNorm, maxNorm);
	const ArtMosaic::Color GrayColor (maxNorm / 2, maxNorm / 2, maxNorm / 2);

	const bool bRet = ArtMosaic::SlicImage (localSrc, localDst, GrayColor, WhiteColor, m, k, g, width, height, line_pitch, line_pitch);

	return (true == bRet ? PF_Err_NONE : PF_Err_INVALID_INDEX);
}


static PF_Err PR_ImageStyle_MosaicArt_VUYA_32f
(
	PF_InData*   __restrict in_data,
	PF_OutData*  __restrict out_data,
	PF_ParamDef* __restrict params[],
	PF_LayerDef* __restrict output
) noexcept
{
	const PF_LayerDef* __restrict pfLayer = reinterpret_cast<const PF_LayerDef*  __restrict>(&params[IMAGE_STYLE_INPUT]->u.ld);
	PF_Pixel_VUYA_32f* __restrict localSrc = reinterpret_cast<PF_Pixel_VUYA_32f* __restrict>(pfLayer->data);
	PF_Pixel_VUYA_32f* __restrict localDst = reinterpret_cast<PF_Pixel_VUYA_32f* __restrict>(output->data);

	auto const height = pfLayer->extent_hint.bottom - pfLayer->extent_hint.top;
	auto const width = pfLayer->extent_hint.right - pfLayer->extent_hint.left;
	auto const line_pitch = pfLayer->rowbytes / static_cast<A_long>(PF_Pixel_VUYA_32f_size);

	/* parameters */
	const float m = 40.f;
	A_long k = 1000, g = 0;
	constexpr float maxNorm = f32_value_white;
	const ArtMosaic::Color WhiteColor(maxNorm, maxNorm, maxNorm);
	const ArtMosaic::Color GrayColor (maxNorm / 2.f, maxNorm / 2.f, maxNorm / 2.f);

	const bool bRet = ArtMosaic::SlicImage(localSrc, localDst, GrayColor, WhiteColor, m, k, g, width, height, line_pitch, line_pitch);

	return (true == bRet ? PF_Err_NONE : PF_Err_INVALID_INDEX);
}



PF_Err PR_ImageStyle_MosaicArt
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
				err = PR_ImageStyle_MosaicArt_BGRA_8u (in_data, out_data, params, output);
			break;

			case PrPixelFormat_VUYA_4444_8u_709:
			case PrPixelFormat_VUYA_4444_8u:
				err = PR_ImageStyle_MosaicArt_VUYA_8u (in_data, out_data, params, output);
			break;

			case PrPixelFormat_VUYA_4444_32f_709:
			case PrPixelFormat_VUYA_4444_32f:
				err = PR_ImageStyle_MosaicArt_VUYA_32f (in_data, out_data, params, output);
			break;

			case PrPixelFormat_BGRA_4444_16u:
				err = PR_ImageStyle_MosaicArt_BGRA_16u (in_data, out_data, params, output);
			break;

			case PrPixelFormat_BGRA_4444_32f:
				err = PR_ImageStyle_MosaicArt_BGRA_32f (in_data, out_data, params, output);
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


PF_Err AE_ImageStyle_MosaicArt_ARGB_8u
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

	auto const height = output->height;
	auto const width = output->width;
	auto const src_line_pitch = input->rowbytes  / static_cast<A_long>(PF_Pixel_ARGB_8u_size);
	auto const dst_line_pitch = output->rowbytes / static_cast<A_long>(PF_Pixel_ARGB_8u_size);

	/* parameters */
	const float m = 40.f;
	A_long k = 1000, g = 0;
	constexpr int maxNorm = std::numeric_limits<unsigned char>::max();
	const ArtMosaic::Color WhiteColor(maxNorm, maxNorm, maxNorm);
	const ArtMosaic::Color GrayColor (maxNorm / 2, maxNorm / 2, maxNorm / 2);

	const bool bRet = ArtMosaic::SlicImage(localSrc, localDst, GrayColor, WhiteColor, m, k, g, width, height, src_line_pitch, dst_line_pitch);

	return (true == bRet ? PF_Err_NONE : PF_Err_INVALID_INDEX);
}


PF_Err AE_ImageStyle_MosaicArt_ARGB_16u
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

	const A_long height = output->height;
	const A_long width = output->width;
	const A_long src_line_pitch = input->rowbytes  / static_cast<A_long>(PF_Pixel_ARGB_16u_size);
	const A_long dst_line_pitch = output->rowbytes / static_cast<A_long>(PF_Pixel_ARGB_16u_size);

	/* parameters */
	const float m = 40.f;
	A_long k = 1000, g = 0;
	constexpr int maxNorm = std::numeric_limits<short int>::max();
	const ArtMosaic::Color WhiteColor(maxNorm, maxNorm, maxNorm);
	const ArtMosaic::Color GrayColor (maxNorm / 2, maxNorm / 2, maxNorm / 2);

	const bool bRet = ArtMosaic::SlicImage(localSrc, localDst, GrayColor, WhiteColor, m, k, g, width, height, src_line_pitch, dst_line_pitch);

	return (true == bRet ? PF_Err_NONE : PF_Err_INVALID_INDEX);
}


PF_Err AE_ImageStyle_MosaicArt_ARGB_32f
(
    PF_InData*   __restrict in_data,
    PF_OutData*  __restrict out_data,
    PF_ParamDef* __restrict params[],
    PF_LayerDef* __restrict output
) noexcept
{
    const PF_EffectWorld* __restrict input = reinterpret_cast<const PF_EffectWorld* __restrict>(&params[IMAGE_STYLE_INPUT]->u.ld);
    PF_Pixel_ARGB_32f*    __restrict localSrc = reinterpret_cast<PF_Pixel_ARGB_32f* __restrict>(input->data);
    PF_Pixel_ARGB_32f*    __restrict localDst = reinterpret_cast<PF_Pixel_ARGB_32f* __restrict>(output->data);

    const A_long height = output->height;
    const A_long width = output->width;
    const A_long src_line_pitch = input->rowbytes  / static_cast<A_long>(PF_Pixel_ARGB_32f_size);
    const A_long dst_line_pitch = output->rowbytes / static_cast<A_long>(PF_Pixel_ARGB_32f_size);

    /* parameters */
    const float m = 40.f;
    A_long k = 1000, g = 0;
    constexpr float maxNorm = f32_value_white;
    const ArtMosaic::Color WhiteColor(maxNorm, maxNorm, maxNorm);
    const ArtMosaic::Color GrayColor(maxNorm / 2.f, maxNorm / 2.f, maxNorm / 2.f);

    const bool bRet = ArtMosaic::SlicImage(localSrc, localDst, GrayColor, WhiteColor, m, k, g, width, height, src_line_pitch, dst_line_pitch);

    return (true == bRet ? PF_Err_NONE : PF_Err_INVALID_INDEX);
}