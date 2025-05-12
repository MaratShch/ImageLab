#include "ColorTemperature.hpp"
#include "ColorTemperatureEnums.hpp"
#include "PrSDKAESupport.h"


PF_Err ProcessImgInPR_BGRA_4444_8u
(
	PF_InData*   __restrict in_data,
	PF_OutData*  __restrict out_data,
	PF_ParamDef* __restrict params[],
	PF_LayerDef* __restrict output
) noexcept
{
	PF_Err err = PF_Err_NONE;
	return err;
}


PF_Err ProcessImgInPR
(
	PF_InData*   __restrict in_data,
	PF_OutData*  __restrict out_data,
	PF_ParamDef* __restrict params[],
	PF_LayerDef* __restrict output
) 
{
	PF_Err err{ PF_Err_NONE };
    PrPixelFormat destinationPixelFormat{ PrPixelFormat_Invalid };

    // This plugin called frop PR - check video fomat
    auto const pixelFormatSuite{ AEFX_SuiteScoper<PF_PixelFormatSuite1>(in_data, kPFPixelFormatSuite, kPFPixelFormatSuiteVersion1, out_data) };

    if (PF_Err_NONE == (err = pixelFormatSuite->GetPixelFormat(output, &destinationPixelFormat)))
    {
        A_long sizeY = 0, sizeX = 0, linePitch = 0;
        const PF_LayerDef* pfLayer = reinterpret_cast<const PF_LayerDef*>(&params[COLOR_TEMPERATURE_FILTER_INPUT]->u.ld);
        
        switch (destinationPixelFormat)
        {
            case PrPixelFormat_BGRA_4444_8u:
            {
                const PF_Pixel_BGRA_8u* __restrict localSrc = reinterpret_cast<const PF_Pixel_BGRA_8u* __restrict>(pfLayer->data);
                      PF_Pixel_BGRA_8u* __restrict localDst = reinterpret_cast<      PF_Pixel_BGRA_8u* __restrict>(output->data);
                sizeY = pfLayer->extent_hint.bottom - pfLayer->extent_hint.top;
                sizeX = pfLayer->extent_hint.right  - pfLayer->extent_hint.left;
                linePitch = pfLayer->rowbytes / static_cast<A_long>(PF_Pixel_BGRA_8u_size);
            }
            break;

            case PrPixelFormat_BGRA_4444_16u:
            {
                const PF_Pixel_BGRA_16u* __restrict localSrc = reinterpret_cast<const PF_Pixel_BGRA_16u* __restrict>(pfLayer->data);
                      PF_Pixel_BGRA_16u* __restrict localDst = reinterpret_cast<      PF_Pixel_BGRA_16u* __restrict>(output->data);
                sizeY = pfLayer->extent_hint.bottom - pfLayer->extent_hint.top;
                sizeX = pfLayer->extent_hint.right  - pfLayer->extent_hint.left;
                linePitch = pfLayer->rowbytes / static_cast<A_long>(PF_Pixel_BGRA_16u_size);
            }
            break;

            case PrPixelFormat_BGRA_4444_32f:
            {
                const PF_Pixel_BGRA_32f* __restrict localSrc = reinterpret_cast<const PF_Pixel_BGRA_32f* __restrict>(pfLayer->data);
                      PF_Pixel_BGRA_32f* __restrict localDst = reinterpret_cast<      PF_Pixel_BGRA_32f* __restrict>(output->data);
                sizeY = pfLayer->extent_hint.bottom - pfLayer->extent_hint.top;
                sizeX = pfLayer->extent_hint.right  - pfLayer->extent_hint.left;
                linePitch = pfLayer->rowbytes / static_cast<A_long>(PF_Pixel_BGRA_32f_size);
            }
            break;

            case PrPixelFormat_VUYA_4444_8u_709:
            case PrPixelFormat_VUYA_4444_8u:
            {
                const PF_Pixel_VUYA_8u* __restrict localSrc = reinterpret_cast<const PF_Pixel_VUYA_8u* __restrict>(pfLayer->data);
                      PF_Pixel_VUYA_8u* __restrict localDst = reinterpret_cast<      PF_Pixel_VUYA_8u* __restrict>(output->data);
                sizeY = pfLayer->extent_hint.bottom - pfLayer->extent_hint.top;
                sizeX = pfLayer->extent_hint.right  - pfLayer->extent_hint.left;
                linePitch = pfLayer->rowbytes / static_cast<A_long>(PF_Pixel_VUYA_8u_size);
            }
            break;

            case PrPixelFormat_VUYA_4444_32f_709:
            case PrPixelFormat_VUYA_4444_32f:
            {
                const PF_Pixel_VUYA_32f* __restrict localSrc = reinterpret_cast<const PF_Pixel_VUYA_32f* __restrict>(pfLayer->data);
                      PF_Pixel_VUYA_32f* __restrict localDst = reinterpret_cast<      PF_Pixel_VUYA_32f* __restrict>(output->data);
                sizeY = pfLayer->extent_hint.bottom - pfLayer->extent_hint.top;
                sizeX = pfLayer->extent_hint.right  - pfLayer->extent_hint.left;
                linePitch = pfLayer->rowbytes / static_cast<A_long>(PF_Pixel_VUYA_32f_size);
            }
            break;

            case PrPixelFormat_RGB_444_10u:
            {
                const PF_Pixel_RGB_10u* __restrict localSrc = reinterpret_cast<const PF_Pixel_RGB_10u* __restrict>(pfLayer->data);
                      PF_Pixel_RGB_10u* __restrict localDst = reinterpret_cast<      PF_Pixel_RGB_10u* __restrict>(output->data);
                sizeY = pfLayer->extent_hint.bottom - pfLayer->extent_hint.top;
                sizeX = pfLayer->extent_hint.right  - pfLayer->extent_hint.left;
                linePitch = pfLayer->rowbytes / static_cast<A_long>(PF_Pixel_RGB_10u_size);
            }
            break;

            default:
                err = PF_Err_INTERNAL_STRUCT_DAMAGED;
            break;
        }
    }


	return err;
}
