#include "ColorBandSelect.hpp"
#include "PrSDKAESupport.h"
#include "ColorBandSelectEnums.hpp"
#include "ColorBandSelectProc.hpp"

inline void ImgCopyByChannelMask
(
	const PF_Pixel_RGB_10u* __restrict pSrcImg,
	      PF_Pixel_RGB_10u* __restrict pDstImg,
	const A_long& srcPitch,
	const A_long& dstPitch,
	const A_long& sizeX,
	const A_long& sizeY,
	const A_long& Red,
	const A_long& Green,
	const A_long& Blue
) noexcept
{
	for (A_long j = 0; j < sizeY; j++)
	{
		const PF_Pixel_RGB_10u* __restrict pSrcLine{ pSrcImg + j * srcPitch };
		      PF_Pixel_RGB_10u* __restrict pDstLine{ pDstImg + j * dstPitch };

		__VECTOR_ALIGNED__
		for (A_long i = 0; i < sizeX; i++)
		{
			pDstLine[i].B = Blue  ? pSrcLine[i].B : 0;
			pDstLine[i].G = Green ? pSrcLine[i].G : 0;
			pDstLine[i].R = Red   ? pSrcLine[i].R : 0;
		}
	}
}


PF_Err ColorBandSelect_BGRA_4444_8u
(
	PF_InData*   __restrict in_data,
	PF_OutData*  __restrict out_data,
	PF_ParamDef* __restrict params[],
	PF_LayerDef* __restrict output
) noexcept
{
	const PF_LayerDef* __restrict pfLayer  = reinterpret_cast<const PF_LayerDef* __restrict>(&params[COLOR_BAND_FILTER_INPUT]->u.ld);
	PF_Pixel_BGRA_8u*  __restrict localSrc = reinterpret_cast<PF_Pixel_BGRA_8u*  __restrict>(pfLayer->data);
	PF_Pixel_BGRA_8u*  __restrict localDst = reinterpret_cast<PF_Pixel_BGRA_8u*  __restrict>(output->data);

	auto const height = pfLayer->extent_hint.bottom - pfLayer->extent_hint.top;
	auto const width  = pfLayer->extent_hint.right - pfLayer->extent_hint.left;
	auto const line_pitch = pfLayer->rowbytes / static_cast<A_long>(PF_Pixel_BGRA_8u_size);

	const A_long ChannelR = params[COLOR_BAND_CHANNEL_RED  ]->u.bd.value;
	const A_long ChannelG = params[COLOR_BAND_CHANNEL_GREEN]->u.bd.value;
	const A_long ChannelB = params[COLOR_BAND_CHANNEL_BLUE ]->u.bd.value;

	PF_Err err = PF_Err_NONE;

	if (0 != ChannelR && 0 != ChannelG && 0 != ChannelB)
		err = PF_COPY(&params[COLOR_BAND_FILTER_INPUT]->u.ld, output, NULL, NULL);
	else
		ImgCopyByChannelMask (localSrc, localDst, line_pitch, line_pitch, width, height, ChannelR, ChannelG, ChannelB);

	return err;
}


PF_Err ColorBandSelect_BGRA_4444_16u
(
	PF_InData*   __restrict in_data,
	PF_OutData*  __restrict out_data,
	PF_ParamDef* __restrict params[],
	PF_LayerDef* __restrict output
) noexcept
{
	const PF_LayerDef*  __restrict pfLayer  = reinterpret_cast<const PF_LayerDef* __restrict>(&params[COLOR_BAND_FILTER_INPUT]->u.ld);
	PF_Pixel_BGRA_16u*  __restrict localSrc = reinterpret_cast<PF_Pixel_BGRA_16u*  __restrict>(pfLayer->data);
	PF_Pixel_BGRA_16u*  __restrict localDst = reinterpret_cast<PF_Pixel_BGRA_16u*  __restrict>(output->data);

	auto const height = pfLayer->extent_hint.bottom - pfLayer->extent_hint.top;
	auto const width = pfLayer->extent_hint.right - pfLayer->extent_hint.left;
	auto const line_pitch = pfLayer->rowbytes / static_cast<A_long>(PF_Pixel_BGRA_16u_size);

	auto const ChannelR = params[COLOR_BAND_CHANNEL_RED  ]->u.bd.value;
	auto const ChannelG = params[COLOR_BAND_CHANNEL_GREEN]->u.bd.value;
	auto const ChannelB = params[COLOR_BAND_CHANNEL_BLUE ]->u.bd.value;

	PF_Err err = PF_Err_NONE;

	if (0 != ChannelR && 0 != ChannelG && 0 != ChannelB)
		err = PF_COPY(&params[COLOR_BAND_FILTER_INPUT]->u.ld, output, NULL, NULL);
	else
		ImgCopyByChannelMask (localSrc, localDst, line_pitch, line_pitch, width, height, ChannelR, ChannelG, ChannelB);

	return err;
}


PF_Err ColorBandSelect_BGRA_4444_32f
(
	PF_InData*   __restrict in_data,
	PF_OutData*  __restrict out_data,
	PF_ParamDef* __restrict params[],
	PF_LayerDef* __restrict output
) noexcept
{
	const PF_LayerDef*  __restrict pfLayer  = reinterpret_cast<const PF_LayerDef* __restrict>(&params[COLOR_BAND_FILTER_INPUT]->u.ld);
	PF_Pixel_BGRA_32f*  __restrict localSrc = reinterpret_cast<PF_Pixel_BGRA_32f*  __restrict>(pfLayer->data);
	PF_Pixel_BGRA_32f*  __restrict localDst = reinterpret_cast<PF_Pixel_BGRA_32f*  __restrict>(output->data);

	auto const height = pfLayer->extent_hint.bottom - pfLayer->extent_hint.top;
	auto const width = pfLayer->extent_hint.right - pfLayer->extent_hint.left;
	auto const line_pitch = pfLayer->rowbytes / static_cast<A_long>(PF_Pixel_BGRA_32f_size);

	auto const ChannelR = params[COLOR_BAND_CHANNEL_RED  ]->u.bd.value;
	auto const ChannelG = params[COLOR_BAND_CHANNEL_GREEN]->u.bd.value;
	auto const ChannelB = params[COLOR_BAND_CHANNEL_BLUE ]->u.bd.value;

	PF_Err err = PF_Err_NONE;

	if (0 != ChannelR && 0 != ChannelG && 0 != ChannelB)
		err = PF_COPY(&params[COLOR_BAND_FILTER_INPUT]->u.ld, output, NULL, NULL);
	else
		ImgCopyByChannelMask (localSrc, localDst, line_pitch, line_pitch, width, height, ChannelR, ChannelG, ChannelB);

	return err;
}


PF_Err ColorBandSelect_BGR_444_10u
(
	PF_InData*   __restrict in_data,
	PF_OutData*  __restrict out_data,
	PF_ParamDef* __restrict params[],
	PF_LayerDef* __restrict output
) noexcept
{
	const PF_LayerDef* __restrict pfLayer  = reinterpret_cast<const PF_LayerDef* __restrict>(&params[COLOR_BAND_FILTER_INPUT]->u.ld);
	PF_Pixel_RGB_10u*  __restrict localSrc = reinterpret_cast<PF_Pixel_RGB_10u*  __restrict>(pfLayer->data);
	PF_Pixel_RGB_10u*  __restrict localDst = reinterpret_cast<PF_Pixel_RGB_10u*  __restrict>(output->data);

	auto const height = pfLayer->extent_hint.bottom - pfLayer->extent_hint.top;
	auto const width = pfLayer->extent_hint.right - pfLayer->extent_hint.left;
	auto const line_pitch = pfLayer->rowbytes / static_cast<A_long>(PF_Pixel_RGB_10u_size);

	auto const ChannelR = params[COLOR_BAND_CHANNEL_RED  ]->u.bd.value;
	auto const ChannelG = params[COLOR_BAND_CHANNEL_GREEN]->u.bd.value;
	auto const ChannelB = params[COLOR_BAND_CHANNEL_BLUE ]->u.bd.value;

	PF_Err err = PF_Err_NONE;

	if (0 != ChannelR && 0 != ChannelG && 0 != ChannelB)
		err = PF_COPY(&params[COLOR_BAND_FILTER_INPUT]->u.ld, output, NULL, NULL);
	else
		ImgCopyByChannelMask (localSrc, localDst, line_pitch, line_pitch, width, height, ChannelR, ChannelG, ChannelB);

	return err;
}


PF_Err ColorBandSelect_VUYA_4444_8u
(
	PF_InData*   __restrict in_data,
	PF_OutData*  __restrict out_data,
	PF_ParamDef* __restrict params[],
	PF_LayerDef* __restrict output
) noexcept
{
	const PF_LayerDef* __restrict pfLayer  = reinterpret_cast<const PF_LayerDef* __restrict>(&params[COLOR_BAND_FILTER_INPUT]->u.ld);
	PF_Pixel_VUYA_8u*  __restrict localSrc = reinterpret_cast<PF_Pixel_VUYA_8u*  __restrict>(pfLayer->data);
	PF_Pixel_VUYA_8u*  __restrict localDst = reinterpret_cast<PF_Pixel_VUYA_8u*  __restrict>(output->data);

	auto const height = pfLayer->extent_hint.bottom - pfLayer->extent_hint.top;
	auto const width = pfLayer->extent_hint.right - pfLayer->extent_hint.left;
	auto const line_pitch = pfLayer->rowbytes / static_cast<A_long>(PF_Pixel_VUYA_8u_size);

	auto const ChannelR = params[COLOR_BAND_CHANNEL_RED]->u.bd.value;
	auto const ChannelG = params[COLOR_BAND_CHANNEL_GREEN]->u.bd.value;
	auto const ChannelB = params[COLOR_BAND_CHANNEL_BLUE]->u.bd.value;

	PF_Err err = PF_Err_NONE;

	if (0 != ChannelR && 0 != ChannelG && 0 != ChannelB)
		err = PF_COPY(&params[COLOR_BAND_FILTER_INPUT]->u.ld, output, NULL, NULL);
	else
		ImgCopyByChannelMask (localSrc, localDst, line_pitch, line_pitch, width, height, ChannelR, ChannelG, ChannelB);

	return err;
}


PF_Err ColorBandSelect_VUYA_4444_32f
(
	PF_InData*   __restrict in_data,
	PF_OutData*  __restrict out_data,
	PF_ParamDef* __restrict params[],
	PF_LayerDef* __restrict output
) noexcept
{
	const PF_LayerDef* __restrict pfLayer  = reinterpret_cast<const PF_LayerDef* __restrict>(&params[COLOR_BAND_FILTER_INPUT]->u.ld);
	PF_Pixel_VUYA_32f* __restrict localSrc = reinterpret_cast<PF_Pixel_VUYA_32f*  __restrict>(pfLayer->data);
	PF_Pixel_VUYA_32f* __restrict localDst = reinterpret_cast<PF_Pixel_VUYA_32f*  __restrict>(output->data);

	auto const height = pfLayer->extent_hint.bottom - pfLayer->extent_hint.top;
	auto const width = pfLayer->extent_hint.right - pfLayer->extent_hint.left;
	auto const line_pitch = pfLayer->rowbytes / static_cast<A_long>(PF_Pixel_VUYA_8u_size);

	auto const ChannelR = params[COLOR_BAND_CHANNEL_RED]->u.bd.value;
	auto const ChannelG = params[COLOR_BAND_CHANNEL_GREEN]->u.bd.value;
	auto const ChannelB = params[COLOR_BAND_CHANNEL_BLUE]->u.bd.value;

	PF_Err err = PF_Err_NONE;

	if (0 != ChannelR && 0 != ChannelG && 0 != ChannelB)
		err = PF_COPY(&params[COLOR_BAND_FILTER_INPUT]->u.ld, output, NULL, NULL);
	else
		ImgCopyByChannelMask (localSrc, localDst, line_pitch, line_pitch, width, height, ChannelR, ChannelG, ChannelB);

	return err;
}



PF_Err ProcessImgInPR
(
	PF_InData*   __restrict in_data,
	PF_OutData*  __restrict out_data,
	PF_ParamDef* __restrict params[],
	PF_LayerDef* __restrict output
) noexcept
{
	PF_Err err{ PF_Err_NONE };
	PF_Err errFormat{ PF_Err_INVALID_INDEX };
	PrPixelFormat destinationPixelFormat{ PrPixelFormat_Invalid };

	/* This plugin called frop PR - check video fomat */
	auto const pixelFormatSuite{ AEFX_SuiteScoper<PF_PixelFormatSuite1>(in_data, kPFPixelFormatSuite, kPFPixelFormatSuiteVersion1, out_data) };

	if (PF_Err_NONE == (errFormat = pixelFormatSuite->GetPixelFormat(output, &destinationPixelFormat)))
	{
		switch (destinationPixelFormat)
		{
			case PrPixelFormat_BGRA_4444_8u:
				err = ColorBandSelect_BGRA_4444_8u (in_data, out_data, params, output);
			break;

			case PrPixelFormat_BGRA_4444_16u:
				err = ColorBandSelect_BGRA_4444_16u (in_data, out_data, params, output);
			break;

			case PrPixelFormat_BGRA_4444_32f:
				err = ColorBandSelect_BGRA_4444_32f (in_data, out_data, params, output);
			break;

			case PrPixelFormat_VUYA_4444_8u_709:
			case PrPixelFormat_VUYA_4444_8u:
				err = ColorBandSelect_VUYA_4444_8u (in_data, out_data, params, output);
			break;
			
			case PrPixelFormat_VUYA_4444_32f_709:
			case PrPixelFormat_VUYA_4444_32f:
				err = ColorBandSelect_VUYA_4444_32f (in_data, out_data, params, output);
			break;

			case PrPixelFormat_RGB_444_10u:
				err = ColorBandSelect_BGR_444_10u (in_data, out_data, params, output);
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
