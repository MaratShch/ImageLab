#include "RetroVision.hpp"
#include "RetroVisionEnum.hpp"
#include "PrSDKAESupport.h"
#include "FastAriphmetics.hpp"

template<typename T, typename U, typename std::enable_if<is_RGB_proc<T>::value && std::is_floating_point<U>::value>::type* = nullptr>
inline T gamma_adjust
(
    const T& in,
    const U& gamma,
    const U& maxVal
) noexcept
{
    T out;

    out.R = static_cast<decltype(out.R)>(FastCompute::Min(FastCompute::Pow(in.R / maxVal, gamma) * maxVal, maxVal));
    out.G = static_cast<decltype(out.G)>(FastCompute::Min(FastCompute::Pow(in.G / maxVal, gamma) * maxVal, maxVal));
    out.B = static_cast<decltype(out.B)>(FastCompute::Min(FastCompute::Pow(in.B / maxVal, gamma) * maxVal, maxVal));
    out.A = in.A;

    return out;
}

template<typename U, typename std::enable_if<std::is_floating_point<U>::value>::type* = nullptr>
inline PF_Pixel_RGB_10u gamma_adjust
(
    const PF_Pixel_RGB_10u& in,
    const U& gamma,
    const U& maxVal
) noexcept
{
    PF_Pixel_RGB_10u out;

    out.R = static_cast<decltype(out.R)>(FastCompute::Min(FastCompute::Pow(in.R, gamma), maxVal));
    out.G = static_cast<decltype(out.G)>(FastCompute::Min(FastCompute::Pow(in.G, gamma), maxVal));
    out.B = static_cast<decltype(out.B)>(FastCompute::Min(FastCompute::Pow(in.B, gamma), maxVal));

    return out;
}


template<typename T, typename U, typename std::enable_if<is_RGB_Variants<T>::value && std::is_floating_point<U>::value>::type* = nullptr>
PF_Err AdjustGammaValue
(
    const T* __restrict srcBuf,
          T* __restrict dstBuf,
    A_long sizeX, 
    A_long sizeY,
    A_long srcPitch,
    A_long dstPitch,
    const U gamma,
    const U maxVal
)
{
    for (A_long j = 0; j < sizeY; j++)
    {
        const T* pSrcLine = srcBuf + j * srcPitch;
              T* pDstLine = dstBuf + j * dstPitch;

        for (A_long i = 0; i < sizeX; i++)
            pDstLine[i] = gamma_adjust (pSrcLine[i], gamma, maxVal);
    }
    return PF_Err_NONE;
}


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

