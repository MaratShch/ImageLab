#include "BlackAndWhiteProc.hpp"
#include "PrSDKAESupport.h"


PF_Err ProcessImgInPR
(
	PF_InData*    in_data,
	PF_OutData*   out_data,
	PF_ParamDef*  params[],
	PF_LayerDef*  output
) noexcept
{
	const PF_LayerDef* pfLayer = reinterpret_cast<const PF_LayerDef*>(&params[IMAGE_BW_FILTER_INPUT]->u.ld);
	PrPixelFormat destinationPixelFormat{ PrPixelFormat_Invalid };
	PF_Err err{ PF_Err_NONE };

	const A_long algoAdvanced = params[IMAGE_BW_ADVANCED_ALGO]->u.bd.value;

	/* This plugin called from PR - check video format */
	if (PF_Err_NONE == AEFX_SuiteScoper<PF_PixelFormatSuite1>(in_data, kPFPixelFormatSuite, kPFPixelFormatSuiteVersion1, out_data)->GetPixelFormat(output, &destinationPixelFormat))
	{
		switch (destinationPixelFormat)
		{
			case PrPixelFormat_BGRA_4444_8u:
			{
				const PF_Pixel_BGRA_8u* __restrict localSrc = reinterpret_cast<const PF_Pixel_BGRA_8u* __restrict>(pfLayer->data);
				      PF_Pixel_BGRA_8u* __restrict localDst = reinterpret_cast<      PF_Pixel_BGRA_8u* __restrict>(output->data);

				auto const height = pfLayer->extent_hint.bottom - pfLayer->extent_hint.top;
				auto const width  = pfLayer->extent_hint.right  - pfLayer->extent_hint.left;
				auto const line_pitch = pfLayer->rowbytes / static_cast<A_long>(PF_Pixel_BGRA_8u_size);

				if (0 != algoAdvanced)
					ProcessImageAdvanced (localSrc, localDst, width, height, line_pitch, line_pitch);
				else
					ProcessImage (localSrc, localDst, width, height, line_pitch, line_pitch, 0);
			}
			break;

			case PrPixelFormat_BGRA_4444_16u:
			{
				const PF_Pixel_BGRA_16u* __restrict localSrc = reinterpret_cast<const PF_Pixel_BGRA_16u* __restrict>(pfLayer->data);
				      PF_Pixel_BGRA_16u* __restrict localDst = reinterpret_cast<      PF_Pixel_BGRA_16u* __restrict>(output->data);

				auto const height = pfLayer->extent_hint.bottom - pfLayer->extent_hint.top;
				auto const width  = pfLayer->extent_hint.right  - pfLayer->extent_hint.left;
				auto const line_pitch = pfLayer->rowbytes / static_cast<A_long>(PF_Pixel_BGRA_16u_size);

				if (0 != algoAdvanced)
					ProcessImageAdvanced (localSrc, localDst, width, height, line_pitch, line_pitch);
				else
					ProcessImage (localSrc, localDst, width, height, line_pitch, line_pitch, 0);
			}
			break;

			case PrPixelFormat_BGRA_4444_32f:
			{
				const PF_Pixel_BGRA_32f* __restrict localSrc = reinterpret_cast<const PF_Pixel_BGRA_32f* __restrict>(pfLayer->data);
				      PF_Pixel_BGRA_32f* __restrict localDst = reinterpret_cast<      PF_Pixel_BGRA_32f* __restrict>(output->data);

				auto const height = pfLayer->extent_hint.bottom - pfLayer->extent_hint.top;
				auto const width  = pfLayer->extent_hint.right  - pfLayer->extent_hint.left;
				auto const line_pitch = pfLayer->rowbytes / static_cast<A_long>(PF_Pixel_BGRA_32f_size);

				if (0 != algoAdvanced)
					ProcessImageAdvanced (localSrc, localDst, width, height, line_pitch, line_pitch);
				else
					ProcessImage (localSrc, localDst, width, height, line_pitch, line_pitch, 0);
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

				ProcessImage (localSrc, localDst, width, height, line_pitch, line_pitch, noColor);
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

				ProcessImage (localSrc, localDst, width, height, line_pitch, line_pitch, noColor);
			}
			break;

			case PrPixelFormat_RGB_444_10u:
			{
				const PF_Pixel_RGB_10u* __restrict localSrc = reinterpret_cast<const PF_Pixel_RGB_10u* __restrict>(pfLayer->data);
				      PF_Pixel_RGB_10u* __restrict localDst = reinterpret_cast<      PF_Pixel_RGB_10u* __restrict>(output->data);

				auto const height = pfLayer->extent_hint.bottom - pfLayer->extent_hint.top;
				auto const width  = pfLayer->extent_hint.right  - pfLayer->extent_hint.left;
				auto const line_pitch = pfLayer->rowbytes / static_cast<A_long>(PF_Pixel_RGB_10u_size);

				if (0 != algoAdvanced)
					ProcessImageAdvanced (localSrc, localDst, width, height, line_pitch, line_pitch);
				else
					ProcessImage (localSrc, localDst, width, height, line_pitch, line_pitch, 0);
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
