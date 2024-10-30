#include "BilateralFilter.hpp"
#include "BilateralFilterAlgo.hpp"
#include "BilateralFilterEnum.hpp"
#include "PrSDKAESupport.h"


PF_Err ProcessImgInPR
(
	PF_InData*   __restrict in_data,
	PF_OutData*  __restrict out_data,
	PF_ParamDef* __restrict params[],
	PF_LayerDef* __restrict output
) noexcept
{
    // Get Bilateral Filter Radius value from slider
    const A_long sliderFilterRadius = params[eBILATERAL_FILTER_RADIUS]->u.sd.value;
    if (0 == sliderFilterRadius) // Filter Radius equal to zero, so algorithm disabled  - let's make simple copy
        return PF_COPY(&params[eBILATERAL_FILTER_INPUT]->u.ld, output, NULL, NULL);

    PF_Err err{ PF_Err_NONE };
    PF_Err errFormat{ PF_Err_INVALID_INDEX };
    PrPixelFormat destinationPixelFormat{ PrPixelFormat_Invalid };

    // This plugin called from Pr - check video format
	auto const& pixelFormatSuite{ AEFX_SuiteScoper<PF_PixelFormatSuite1>(in_data, kPFPixelFormatSuite, kPFPixelFormatSuiteVersion1, out_data) };

	if (PF_Err_NONE == (errFormat = pixelFormatSuite->GetPixelFormat(output, &destinationPixelFormat)))
	{
        A_long sizeY = 0, sizeX = 0, srcPitch = 0, dstPitch = 0;
        const PF_LayerDef* pfLayer = reinterpret_cast<const PF_LayerDef*>(&params[eBILATERAL_FILTER_INPUT]->u.ld);

		switch (destinationPixelFormat)
		{
			case PrPixelFormat_BGRA_4444_8u:
            {
                const PF_Pixel_BGRA_8u* __restrict localSrc = reinterpret_cast<const PF_Pixel_BGRA_8u* __restrict>(pfLayer->data);
                      PF_Pixel_BGRA_8u* __restrict localDst = reinterpret_cast<      PF_Pixel_BGRA_8u* __restrict>(output->data);
                sizeY = pfLayer->extent_hint.bottom - pfLayer->extent_hint.top;
                sizeX = pfLayer->extent_hint.right  - pfLayer->extent_hint.left;
                dstPitch = srcPitch = pfLayer->rowbytes / static_cast<A_long>(PF_Pixel_BGRA_8u_size);

                PF_Pixel_BGRA_8u white{}, black{};
                pixelFormatSuite->GetBlackForPixelFormat(PrPixelFormat_BGRA_4444_8u, &black);
                pixelFormatSuite->GetWhiteForPixelFormat(PrPixelFormat_BGRA_4444_8u, &white);

                BilateralFilterAlgorithm (localSrc, localDst, sizeX, sizeY, srcPitch, dstPitch, sliderFilterRadius, black, white);
            }
			break;

			case PrPixelFormat_BGRA_4444_16u:
            {
                const PF_Pixel_BGRA_16u* __restrict localSrc = reinterpret_cast<const PF_Pixel_BGRA_16u* __restrict>(pfLayer->data);
                      PF_Pixel_BGRA_16u* __restrict localDst = reinterpret_cast<      PF_Pixel_BGRA_16u* __restrict>(output->data);
                sizeY = pfLayer->extent_hint.bottom - pfLayer->extent_hint.top;
                sizeX = pfLayer->extent_hint.right - pfLayer->extent_hint.left;
                dstPitch = srcPitch = pfLayer->rowbytes / static_cast<A_long>(PF_Pixel_BGRA_16u_size);

                PF_Pixel_BGRA_16u white{}, black{};
                pixelFormatSuite->GetBlackForPixelFormat(PrPixelFormat_BGRA_4444_16u, &black);
                pixelFormatSuite->GetWhiteForPixelFormat(PrPixelFormat_BGRA_4444_16u, &white);

                BilateralFilterAlgorithm (localSrc, localDst, sizeX, sizeY, srcPitch, dstPitch, sliderFilterRadius, black, white);
            }
			break;

			case PrPixelFormat_BGRA_4444_32f:
            {
                const PF_Pixel_BGRA_32f* __restrict localSrc = reinterpret_cast<const PF_Pixel_BGRA_32f* __restrict>(pfLayer->data);
                      PF_Pixel_BGRA_32f* __restrict localDst = reinterpret_cast<      PF_Pixel_BGRA_32f* __restrict>(output->data);
                sizeY = pfLayer->extent_hint.bottom - pfLayer->extent_hint.top;
                sizeX = pfLayer->extent_hint.right  - pfLayer->extent_hint.left;
                dstPitch = srcPitch = pfLayer->rowbytes / static_cast<A_long>(PF_Pixel_BGRA_32f_size);

                PF_Pixel_BGRA_32f white{}, black{};
                pixelFormatSuite->GetBlackForPixelFormat(PrPixelFormat_BGRA_4444_32f, &black);
                pixelFormatSuite->GetWhiteForPixelFormat(PrPixelFormat_BGRA_4444_32f, &white);

                BilateralFilterAlgorithm (localSrc, localDst, sizeX, sizeY, srcPitch, dstPitch, sliderFilterRadius, black, white);
            }
			break;

            case PrPixelFormat_ARGB_4444_8u:
            {
                const PF_Pixel_ARGB_8u* __restrict localSrc = reinterpret_cast<const PF_Pixel_ARGB_8u* __restrict>(pfLayer->data);
                      PF_Pixel_ARGB_8u* __restrict localDst = reinterpret_cast<      PF_Pixel_ARGB_8u* __restrict>(output->data);
                sizeY = pfLayer->extent_hint.bottom - pfLayer->extent_hint.top;
                sizeX = pfLayer->extent_hint.right - pfLayer->extent_hint.left;
                dstPitch = srcPitch = pfLayer->rowbytes / static_cast<A_long>(PF_Pixel_ARGB_8u_size);

                PF_Pixel_ARGB_8u white{}, black{};
                pixelFormatSuite->GetBlackForPixelFormat(PrPixelFormat_ARGB_4444_8u, &black);
                pixelFormatSuite->GetWhiteForPixelFormat(PrPixelFormat_ARGB_4444_8u, &white);

                BilateralFilterAlgorithm (localSrc, localDst, sizeX, sizeY, srcPitch, dstPitch, sliderFilterRadius, black, white);
            }
            break;

            case PrPixelFormat_ARGB_4444_16u:
            {
                const PF_Pixel_ARGB_16u* __restrict localSrc = reinterpret_cast<const PF_Pixel_ARGB_16u* __restrict>(pfLayer->data);
                      PF_Pixel_ARGB_16u* __restrict localDst = reinterpret_cast<      PF_Pixel_ARGB_16u* __restrict>(output->data);
                sizeY = pfLayer->extent_hint.bottom - pfLayer->extent_hint.top;
                sizeX = pfLayer->extent_hint.right  - pfLayer->extent_hint.left;
                dstPitch = srcPitch = pfLayer->rowbytes / static_cast<A_long>(PF_Pixel_ARGB_16u_size);

                PF_Pixel_ARGB_16u white{}, black{};
                pixelFormatSuite->GetBlackForPixelFormat(PrPixelFormat_ARGB_4444_16u, &black);
                pixelFormatSuite->GetWhiteForPixelFormat(PrPixelFormat_ARGB_4444_16u, &white);

                BilateralFilterAlgorithm (localSrc, localDst, sizeX, sizeY, srcPitch, dstPitch, sliderFilterRadius, black, white);
            }
            break;
            
            case PrPixelFormat_ARGB_4444_32f:
            {
                const PF_Pixel_ARGB_32f* __restrict localSrc = reinterpret_cast<const PF_Pixel_ARGB_32f* __restrict>(pfLayer->data);
                      PF_Pixel_ARGB_32f* __restrict localDst = reinterpret_cast<      PF_Pixel_ARGB_32f* __restrict>(output->data);
                sizeY = pfLayer->extent_hint.bottom - pfLayer->extent_hint.top;
                sizeX = pfLayer->extent_hint.right - pfLayer->extent_hint.left;
                dstPitch = srcPitch = pfLayer->rowbytes / static_cast<A_long>(PF_Pixel_ARGB_32f_size);

                PF_Pixel_ARGB_32f white{}, black{};
                pixelFormatSuite->GetBlackForPixelFormat(PrPixelFormat_ARGB_4444_32f, &black);
                pixelFormatSuite->GetWhiteForPixelFormat(PrPixelFormat_ARGB_4444_32f, &white);

                BilateralFilterAlgorithm (localSrc, localDst, sizeX, sizeY, srcPitch, dstPitch, sliderFilterRadius, black, white);
            }
            break;

			default:
                // not supported pixel format
                err = PF_Err_INVALID_INDEX;
			break;
		} // switch (destinationPixelFormat) 

	} // if (PF_Err_NONE == (errFormat = pixelFormatSuite->GetPixelFormat(output, &destinationPixelFormat)))
	else
	{
		// error in determine pixel format
		err = PF_Err_UNRECOGNIZED_PARAM_TYPE;
	}

	return err;
}
