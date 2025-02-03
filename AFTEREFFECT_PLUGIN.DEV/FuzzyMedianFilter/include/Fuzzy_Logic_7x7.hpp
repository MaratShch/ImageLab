#ifndef __FUZZY_ALGO_LOGIC_KERNEL_7x7__
#define __FUZZY_ALGO_LOGIC_KERNEL_7x7__

#include "FuzzyRules.hpp"

/*
    NWNWNW  NNWNW  NNNW   NNN    NNNE   NNENE  NENENE
    WNWNW   NWNW   NNW    NN     N      NNE    NENE
    WNW     NW     NWN    N      N      NEN    NEE
    WWW     WNW    W      C      E      ENE    EEE
    SWWW    SWW    SW     S      SE     SEE    SEEE
    SWSWW   SWSE   SSW    SS     SSE    SSEE   SESE
    SWSWSW  SWESE  SSSW   SSS    SSE    SSESE  SESESE
*/

template <typename T, std::enable_if_t<is_RGB_proc<T>::value>* = nullptr>
inline void FuzzyLogic_7x7
(
    const fCIELabPix* __restrict pLabIn,
    const T* __restrict pIn, /* add Input original (non-filtered) image for get Alpha channels values onl y*/
          T* __restrict pOut,
    const A_long&          sizeX,
    const A_long&          sizeY,
    const A_long&          labInPitch,
    const A_long&          imgInPitch,
    const A_long&          imgOutPitch,
    const T&               blackPix, // black (minimal) color pixel value - used for clamping
    const T&               whitePix, // white (maximal) color pixel value - used for clamping
    const float&           fSigma = 2.f
)
{
    return;
}


template <typename T, std::enable_if_t<is_YUV_proc<T>::value>* = nullptr>
inline void FuzzyLogic_7x7
(
    const fCIELabPix* __restrict pLabIn,
    const T* __restrict pIn, /* add Input original (non-filtered) image for get Alpha channels values onl y*/
          T* __restrict pOut,
    const A_long&          sizeX,
    const A_long&          sizeY,
    const A_long&          labInPitch,
    const A_long&          imgInPitch,
    const A_long&          imgOutPitch,
    const T&               blackPix, // black (minimal) color pixel value - used for clamping
    const T&               whitePix, // white (maximal) color pixel value - used for clamping
    const float&           fSigma = 2.f
)
{
    return;
}


inline void FuzzyLogic_7x7
(
    const fCIELabPix* __restrict pLabIn,
    const PF_Pixel_RGB_10u* __restrict pIn, /* add Input original (non-filtered) image for get Alpha channels values onl y*/
          PF_Pixel_RGB_10u* __restrict pOut,
    const A_long&          sizeX,
    const A_long&          sizeY,
    const A_long&          labInPitch,
    const A_long&          imgInPitch,
    const A_long&          imgOutPitch,
    const PF_Pixel_RGB_10u& blackPix, // black (minimal) color pixel value - used for clamping
    const PF_Pixel_RGB_10u& whitePix, // white (maximal) color pixel value - used for clamping
    const float&           fSigma = 2.f
)
{
    return;
}



#endif // __FUZZY_ALGO_LOGIC_KERNEL_7x7__