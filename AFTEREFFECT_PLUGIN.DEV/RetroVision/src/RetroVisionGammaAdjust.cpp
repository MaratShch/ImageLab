#include "RetroVision.hpp"
#include "RetroVisionEnum.hpp"
#include "CommonAuxPixFormat.hpp"
#include "PrSDKAESupport.h"
#include "FastAriphmetics.hpp"
#include "AdjustGamma.hpp"


PF_Err AdjustGammaValue
(
    PF_InData*   __restrict in_data,
    PF_OutData*  __restrict out_data,
    PF_ParamDef* __restrict params[],
    PF_LayerDef* __restrict output,
    const float fGamma
)
{
    PF_Err errFormat{ PF_Err_INVALID_INDEX };
    PF_Err err{ PF_Err_NONE };
    PrPixelFormat destinationPixelFormat{ PrPixelFormat_Invalid };

    const PF_LayerDef* pfLayer = reinterpret_cast<const PF_LayerDef*>(&params[UnderlyingType(RetroVision::eRETRO_VISION_INPUT)]->u.ld);
    const A_long sizeY = pfLayer->extent_hint.bottom - pfLayer->extent_hint.top;
    const A_long sizeX = pfLayer->extent_hint.right - pfLayer->extent_hint.left;

    // This plugin called frop PR - check video fomat
    auto const pixelFormatSuite{ AEFX_SuiteScoper<PF_PixelFormatSuite1>(in_data, kPFPixelFormatSuite, kPFPixelFormatSuiteVersion1, out_data) };
    if (PF_Err_NONE == (errFormat = pixelFormatSuite->GetPixelFormat(output, &destinationPixelFormat)))
    {
        switch (destinationPixelFormat)
        {
            case PrPixelFormat_BGRA_4444_8u:
            {
                const PF_Pixel_BGRA_8u* __restrict localSrc = reinterpret_cast<const PF_Pixel_BGRA_8u* __restrict>(pfLayer->data);
                      PF_Pixel_BGRA_8u* __restrict localDst = reinterpret_cast<      PF_Pixel_BGRA_8u* __restrict>(output->data);
                const A_long linePitch = pfLayer->rowbytes / static_cast<A_long>(PF_Pixel_BGRA_8u_size);
                constexpr float fCoeff{ static_cast<float>(u8_value_white) };

                err = AdjustGammaValue(localSrc, localDst, sizeX, sizeY, linePitch, linePitch, fGamma, fCoeff);
            }
            break;

            case PrPixelFormat_ARGB_4444_8u:
            {
                const PF_Pixel_ARGB_8u* __restrict localSrc = reinterpret_cast<const PF_Pixel_ARGB_8u* __restrict>(pfLayer->data);
                      PF_Pixel_ARGB_8u* __restrict localDst = reinterpret_cast<      PF_Pixel_ARGB_8u* __restrict>(output->data);
                const A_long linePitch = pfLayer->rowbytes / static_cast<A_long>(PF_Pixel_ARGB_8u_size);
                constexpr float fCoeff{ static_cast<float>(u8_value_white) };

                err = AdjustGammaValue(localSrc, localDst, sizeX, sizeY, linePitch, linePitch, fGamma, fCoeff);
            }
            break;

            case PrPixelFormat_BGRA_4444_16u:
            {
                const PF_Pixel_BGRA_16u* __restrict localSrc = reinterpret_cast<const PF_Pixel_BGRA_16u* __restrict>(pfLayer->data);
                      PF_Pixel_BGRA_16u* __restrict localDst = reinterpret_cast<      PF_Pixel_BGRA_16u* __restrict>(output->data);
                const A_long linePitch = pfLayer->rowbytes / static_cast<A_long>(PF_Pixel_BGRA_16u_size);
                constexpr float fCoeff{ static_cast<float>(u16_value_white) };
 
                err = AdjustGammaValue(localSrc, localDst, sizeX, sizeY, linePitch, linePitch, fGamma, fCoeff);
            }
            break;

            case PrPixelFormat_ARGB_4444_16u:
            {
                const PF_Pixel_ARGB_16u* __restrict localSrc = reinterpret_cast<const PF_Pixel_ARGB_16u* __restrict>(pfLayer->data);
                      PF_Pixel_ARGB_16u* __restrict localDst = reinterpret_cast<      PF_Pixel_ARGB_16u* __restrict>(output->data);
                const A_long linePitch = pfLayer->rowbytes / static_cast<A_long>(PF_Pixel_ARGB_16u_size);
                constexpr float fCoeff{ static_cast<float>(u16_value_white) };

                err = AdjustGammaValue(localSrc, localDst, sizeX, sizeY, linePitch, linePitch, fGamma, fCoeff);
            }
            break;

            case PrPixelFormat_BGRA_4444_32f:
            {
                const PF_Pixel_BGRA_32f* __restrict localSrc = reinterpret_cast<const PF_Pixel_BGRA_32f* __restrict>(pfLayer->data);
                      PF_Pixel_BGRA_32f* __restrict localDst = reinterpret_cast<      PF_Pixel_BGRA_32f* __restrict>(output->data);
                const A_long linePitch = pfLayer->rowbytes / static_cast<A_long>(PF_Pixel_BGRA_32f_size);
                constexpr float fCoeff{ static_cast<float>(1) };

                err = AdjustGammaValue(localSrc, localDst, sizeX, sizeY, linePitch, linePitch, fGamma, fCoeff);
            }
            break;

            case PrPixelFormat_ARGB_4444_32f:
            {
                const PF_Pixel_ARGB_32f* __restrict localSrc = reinterpret_cast<const PF_Pixel_ARGB_32f* __restrict>(pfLayer->data);
                      PF_Pixel_ARGB_32f* __restrict localDst = reinterpret_cast<      PF_Pixel_ARGB_32f* __restrict>(output->data);
                const A_long linePitch = pfLayer->rowbytes / static_cast<A_long>(PF_Pixel_ARGB_32f_size);
                constexpr float fCoeff{ static_cast<float>(1) };

                err = AdjustGammaValue(localSrc, localDst, sizeX, sizeY, linePitch, linePitch, fGamma, fCoeff);
            }
            break;

            case PrPixelFormat_VUYA_4444_8u_709:
            case PrPixelFormat_VUYA_4444_8u:
            {
                const PF_Pixel_VUYA_8u* __restrict localSrc = reinterpret_cast<const PF_Pixel_VUYA_8u* __restrict>(pfLayer->data);
                      PF_Pixel_VUYA_8u* __restrict localDst = reinterpret_cast<      PF_Pixel_VUYA_8u* __restrict>(output->data);
                const A_long linePitch = pfLayer->rowbytes / static_cast<A_long>(PF_Pixel_VUYA_8u_size);
                const bool isBT709 = (PrPixelFormat_VUYA_4444_8u_709 == destinationPixelFormat);
                constexpr float fCoeff{ static_cast<float>(u8_value_white) };

     //         err = AdjustGammaValue(localSrc, localDst, sizeX, sizeY, linePitch, linePitch, fGamma, fCoeff);
            }
            break;

            case PrPixelFormat_VUYA_4444_32f_709:
            case PrPixelFormat_VUYA_4444_32f:
            {
                const PF_Pixel_VUYA_32f* __restrict localSrc = reinterpret_cast<const PF_Pixel_VUYA_32f* __restrict>(pfLayer->data);
                      PF_Pixel_VUYA_32f* __restrict localDst = reinterpret_cast<      PF_Pixel_VUYA_32f* __restrict>(output->data);
                const A_long linePitch = pfLayer->rowbytes / static_cast<A_long>(PF_Pixel_VUYA_32f_size);
                const bool isBT709 = (PrPixelFormat_VUYA_4444_32f_709 == destinationPixelFormat);
                constexpr float fCoeff{ static_cast<float>(1) };

    //          err = AdjustGammaValue(localSrc, localDst, sizeX, sizeY, linePitch, linePitch, fGamma, fCoeff);
            }
            break;

            case PrPixelFormat_RGB_444_10u:
            {
                const PF_Pixel_RGB_10u* __restrict localSrc = reinterpret_cast<const PF_Pixel_RGB_10u* __restrict>(pfLayer->data);
                      PF_Pixel_RGB_10u* __restrict localDst = reinterpret_cast<      PF_Pixel_RGB_10u* __restrict>(output->data);
                const A_long linePitch = pfLayer->rowbytes / static_cast<A_long>(PF_Pixel_RGB_10u_size);
                constexpr float fCoeff{ static_cast<float>(u10_value_white) };

                err = AdjustGammaValue(localSrc, localDst, sizeX, sizeY, linePitch, linePitch, fGamma, fCoeff);
            }
            break;

            default:
                err = PF_Err_INTERNAL_STRUCT_DAMAGED;
            break;
        }
    }

    return err;
}

