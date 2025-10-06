#ifndef __IMAGE_LAB_RETRO_VISION_ADJUST_GAMMA_METHODS__
#define __IMAGE_LAB_RETRO_VISION_ADJUST_GAMMA_METHODS__

#include "RetroVisionEnum.hpp"
#include "CommonAuxPixFormat.hpp"
#include "PrSDKAESupport.h"
#include "FastAriphmetics.hpp"

PF_Err AdjustGammaValue
(
    PF_InData*   __restrict in_data,
    PF_OutData*  __restrict out_data,
    PF_ParamDef* __restrict params[],
    PF_LayerDef* __restrict output,
    const float fGamma
);


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

    out.R = static_cast<decltype(out.R)>(FastCompute::Min(FastCompute::Pow(in.R / maxVal, gamma) * maxVal, maxVal));
    out.G = static_cast<decltype(out.G)>(FastCompute::Min(FastCompute::Pow(in.G / maxVal, gamma) * maxVal, maxVal));
    out.B = static_cast<decltype(out.B)>(FastCompute::Min(FastCompute::Pow(in.B / maxVal, gamma) * maxVal, maxVal));

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

        __VECTORIZATION__
            for (A_long i = 0; i < sizeX; i++)
                pDstLine[i] = gamma_adjust(pSrcLine[i], gamma, maxVal);
    }
    return PF_Err_NONE;
}



#endif // __IMAGE_LAB_RETRO_VISION_ADJUST_GAMMA_METHODS__