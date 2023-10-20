#include "BlackAndWhite.hpp"
#include "PrSDKAESupport.h"


template <typename U, typename T, std::enable_if_t<is_YUV_proc<T>::value>* = nullptr>
inline void ProcessImageYUV
(
	const T* __restrict pSrc,
	      T* __restrict pDst,
	A_long              sizeX,
	A_long              sizeY,
	A_long              linePitch,
	U                   noColor
) noexcept
{
	for (A_long j = 0; j < sizeY; j++)
	{
		const T* __restrict pSrcLine = pSrc + j * linePitch;
		      T* __restrict pDstLine = pDst + j * linePitch;
  	    for (A_long i = 0; i < sizeX; i++)
		{
			pDstLine[i].V = noColor;
			pDstLine[i].U = noColor;
			pDstLine[i].Y = pSrcLine[i].Y;
			pDstLine[i].A = pSrcLine[i].A;
		}
	}

	return;
}


PF_Err ProcessImgInPR
(
	PF_InData*    in_data,
	PF_OutData*   out_data,
	PF_ParamDef*  params[],
	PF_LayerDef*  output
) noexcept
{
	PF_Err err { PF_Err_NONE };
	PrPixelFormat destinationPixelFormat { PrPixelFormat_Invalid };

	const PF_LayerDef* pfLayer = reinterpret_cast<const PF_LayerDef*>(&params[IMAGE_BW_FILTER_INPUT]->u.ld);

	/* This plugin called from PR - check video format */
	if (PF_Err_NONE == AEFX_SuiteScoper<PF_PixelFormatSuite1>(in_data, kPFPixelFormatSuite, kPFPixelFormatSuiteVersion1, out_data)->GetPixelFormat(output, &destinationPixelFormat))
	{
		switch (destinationPixelFormat)
		{
			case PrPixelFormat_BGRA_4444_8u:
			{
			}
			break;

			case PrPixelFormat_BGRA_4444_16u:
			{
			}
			break;

			case PrPixelFormat_BGRA_4444_32f:
			{
			}
			break;

			case PrPixelFormat_VUYA_4444_8u:
			case PrPixelFormat_VUYA_4444_8u_709:
			{
				constexpr A_long noColor = 0x80;
				const PF_Pixel_VUYA_8u* __restrict localSrc = reinterpret_cast<const PF_Pixel_VUYA_8u* __restrict>(pfLayer->data);
				      PF_Pixel_VUYA_8u* __restrict localDst = reinterpret_cast<      PF_Pixel_VUYA_8u* __restrict>(output->data);

				auto const height = pfLayer->extent_hint.bottom - pfLayer->extent_hint.top;
				auto const width  = pfLayer->extent_hint.right  - pfLayer->extent_hint.left;
				auto const line_pitch = pfLayer->rowbytes / static_cast<A_long>(PF_Pixel_VUYA_8u_size);

				ProcessImageYUV (localSrc, localDst, width, height, line_pitch, noColor);
			}
			break;

			case PrPixelFormat_VUYA_4444_32f:
			case PrPixelFormat_VUYA_4444_32f_709:
			{
				constexpr PF_FpShort noColor = 0.f;
				const PF_Pixel_VUYA_32f* __restrict localSrc = reinterpret_cast<const PF_Pixel_VUYA_32f* __restrict>(pfLayer->data);
				      PF_Pixel_VUYA_32f* __restrict localDst = reinterpret_cast<      PF_Pixel_VUYA_32f* __restrict>(output->data);

				auto const height = pfLayer->extent_hint.bottom - pfLayer->extent_hint.top;
				auto const width  = pfLayer->extent_hint.right  - pfLayer->extent_hint.left;
				auto const line_pitch = pfLayer->rowbytes / static_cast<A_long>(PF_Pixel_VUYA_32f_size);

				ProcessImageYUV(localSrc, localDst, width, height, line_pitch, noColor);
			}
			break;

			case PrPixelFormat_RGB_444_10u:
			{
			}
			break;

			default:
			{
				err = PF_Err_UNRECOGNIZED_PARAM_TYPE;
				break;
			}
		} /* switch (destinationPixelFormat) */

	} /* if (PF_Err_NONE == (errFormat = pixelFormatSuite->GetPixelFormat(output, &destinationPixelFormat))) */
	else
	{
		/* error in determine pixel format */
		err = PF_Err_UNRECOGNIZED_PARAM_TYPE;
	}

	return err;
}