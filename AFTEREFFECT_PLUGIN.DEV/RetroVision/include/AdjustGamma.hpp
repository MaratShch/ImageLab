#ifndef __IMAGE_LAB_RETRO_VISION_ADJUST_GAMMA_METHODS__
#define __IMAGE_LAB_RETRO_VISION_ADJUST_GAMMA_METHODS__

#include <type_traits>
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

// -----------------------------------------------------------------------------
// 1. RGB GAMMA KERNEL
// -----------------------------------------------------------------------------
template<typename T, typename U, typename std::enable_if<is_RGB_proc<T>::value && std::is_floating_point<U>::value>::type* = nullptr>
inline T gamma_adjust
(
    const T& in,
    const U& gamma,
    const U& maxVal,
    const U& invMaxVal
) noexcept
{
    T out;

    // Optimized: Multiplication by invMaxVal is much faster than division
    out.R = static_cast<decltype(out.R)>(FastCompute::Min(FastCompute::Pow(in.R * invMaxVal, gamma) * maxVal, maxVal));
    out.G = static_cast<decltype(out.G)>(FastCompute::Min(FastCompute::Pow(in.G * invMaxVal, gamma) * maxVal, maxVal));
    out.B = static_cast<decltype(out.B)>(FastCompute::Min(FastCompute::Pow(in.B * invMaxVal, gamma) * maxVal, maxVal));
    out.A = in.A;

    return out;
}

// -----------------------------------------------------------------------------
// 2. YUV GAMMA KERNEL (Luma Gamma + Chroma Scale)
// -----------------------------------------------------------------------------
template<typename T, typename U, typename std::enable_if<is_YUV_proc<T>::value && std::is_floating_point<U>::value>::type* = nullptr>
inline T gamma_adjust
(
    const T& in,
    const U& gamma,
    const U& maxVal,
    const U& invMaxVal
) noexcept
{
    T out;

    // 1. Calculate the Chroma Offset based on bit-depth
    // 8u YUV uses 128 as the zero-color center. 32f YUV uses 0.0.
    const U chromaOffset = (maxVal > 2.0) ? (maxVal / (U)2.0) : (U)0.0;

    // 2. Apply Gamma strictly to the Luma (Y) channel
    U normY = FastCompute::Max(in.Y * invMaxVal, (U)0.0); // Clamp to avoid pow(<0)
    U newY = FastCompute::Pow(normY, gamma) * maxVal;

    out.Y = static_cast<decltype(out.Y)>(FastCompute::Min(newY, maxVal));

    // 3. Scale Chroma (U, V) proportionally to preserve color saturation
    // If we just apply gamma to Y without scaling U/V, the image loses saturation.
    U ratio = (in.Y > (maxVal * (U)0.01)) ? (newY / (U)in.Y) : (U)0.0;

    U scaledU = ((in.U - chromaOffset) * ratio) + chromaOffset;
    U scaledV = ((in.V - chromaOffset) * ratio) + chromaOffset;

    out.U = static_cast<decltype(out.U)>(FastCompute::Min(FastCompute::Max(scaledU, (U)0.0), maxVal));
    out.V = static_cast<decltype(out.V)>(FastCompute::Min(FastCompute::Max(scaledV, (U)0.0), maxVal));
    out.A = in.A;

    return out;
}

// -----------------------------------------------------------------------------
// 3. 10-BIT RGB GAMMA KERNEL
// -----------------------------------------------------------------------------
template<typename U, typename std::enable_if<std::is_floating_point<U>::value>::type* = nullptr>
inline PF_Pixel_RGB_10u gamma_adjust
(
    const PF_Pixel_RGB_10u& in,
    const U& gamma,
    const U& maxVal,
    const U& invMaxVal
) noexcept
{
    PF_Pixel_RGB_10u out;

    out.R = static_cast<decltype(out.R)>(FastCompute::Min(FastCompute::Pow(in.R * invMaxVal, gamma) * maxVal, maxVal));
    out.G = static_cast<decltype(out.G)>(FastCompute::Min(FastCompute::Pow(in.G * invMaxVal, gamma) * maxVal, maxVal));
    out.B = static_cast<decltype(out.B)>(FastCompute::Min(FastCompute::Pow(in.B * invMaxVal, gamma) * maxVal, maxVal));

    return out;
}

// -----------------------------------------------------------------------------
// 4. MAIN PROCESSING LOOP (RGB Variants)
// -----------------------------------------------------------------------------
template<typename T, typename U, typename std::enable_if<is_RGB_Variants<T>::value && std::is_floating_point<U>::value>::type* = nullptr>
inline PF_Err AdjustGammaValue
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
    // PRE-CALCULATE DIVISION: Eliminates ~24 million division ops for a 4K frame
    const U invMaxVal = (U)1.0 / maxVal;

    for (A_long j = 0; j < sizeY; j++)
    {
        // BUG FIX: Adobe pitch is ALWAYS in BYTES, not pixels. 
        // You MUST cast to char* before doing pitch math!
        const T* pSrcLine = reinterpret_cast<const T*>(srcBuf + (j * srcPitch));
              T* pDstLine = reinterpret_cast<T*>(dstBuf + (j * dstPitch));

        __VECTORIZATION__
        for (A_long i = 0; i < sizeX; i++)
        {
            pDstLine[i] = gamma_adjust(pSrcLine[i], gamma, maxVal, invMaxVal);
        }
    }
    return PF_Err_NONE;
}

// -----------------------------------------------------------------------------
// 5. MAIN PROCESSING LOOP (YUV Variants)
// -----------------------------------------------------------------------------
template<typename T, typename U, typename std::enable_if<is_YUV_proc<T>::value && std::is_floating_point<U>::value>::type* = nullptr>
inline PF_Err AdjustGammaValue
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
    const U invMaxVal = (U)1.0 / maxVal;

    for (A_long j = 0; j < sizeY; j++)
    {
        const T* pSrcLine = reinterpret_cast<const T*>(srcBuf + (j * srcPitch));
              T* pDstLine = reinterpret_cast<T*>(dstBuf + (j * dstPitch));

        __VECTORIZATION__
        for (A_long i = 0; i < sizeX; i++)
        {
            pDstLine[i] = gamma_adjust(pSrcLine[i], gamma, maxVal, invMaxVal);
        }
    }
    return PF_Err_NONE;
}

#endif // __IMAGE_LAB_RETRO_VISION_ADJUST_GAMMA_METHODS__