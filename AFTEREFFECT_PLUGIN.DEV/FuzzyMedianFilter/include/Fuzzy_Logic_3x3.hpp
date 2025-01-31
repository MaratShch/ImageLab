#ifndef __FUZZY_ALGO_LOGIC_KERNEL_3x3__
#define __FUZZY_ALGO_LOGIC_KERNEL_3x3__

#include "FuzzyRules.hpp"

/*
    NW  N   NE
    W   C   E
    SW  S   SE
*/

template <typename T, std::enable_if_t<is_RGB_proc<T>::value>* = nullptr>
inline void FuzzyLogic_3x3
(
    const fCIELabPix* __restrict pLabIn,
    const T* __restrict pIn, /* add Input original (non-filtered) image for get Alpha channels values onl y*/
          T* __restrict pOut,
    const A_long&          sizeX,
    const A_long&          sizeY,
    const A_long&          labInPitch,
    const A_long&          imgInPitch,
    const A_long&          imgOutPitch
)
{
    A_long i, j;
    const A_long lastPix  = sizeX - 1;
    const A_long lastLine = sizeY - 1;
    float dNW, dN, dNE, dW, dE, dSW, dS, dSE;

    for (j = 0; j < sizeY; j++)
    {
        const fCIELabPix* __restrict labLinePrv = pLabIn + (j - 1) * labInPitch; // line -1
        const fCIELabPix* __restrict labLineCur = pLabIn +  j * labInPitch;      // cureent line
        const fCIELabPix* __restrict labLineNxt = pLabIn + (j + 1) * labInPitch; // line +1
        const T* __restrict inOrgLine = pIn  + j * imgInPitch;
              T* __restrict outLine   = pOut + j * imgOutPitch;
        
        for (i = 0; i < sizeX; i++)
        {
            // CURRENT pixel
            const float C = labLineCur[i].L;

            // NORTH pixels
            if (0 == j)
                dNW = dN = dNE = 0.f;
            else
            {
                dNW = (0 != i ? FastCompute::Abs(C - labLinePrv[i - 1].L) : 0.f);
                dN  = FastCompute::Abs(C - labLinePrv[i].L);
                dNE = (lastPix < i ?  FastCompute::Abs(C - labLinePrv[i + 1].L) : 0.f);
            }

            // CENTRAL pixels
            dW = (0 != i ? FastCompute::Abs(C - labLineCur[i - 1].L) : 0.f);
            dE = (lastPix < i ? FastCompute::Abs(C - labLineCur[i + 1].L) : 0.f);

            // SOUTH pixels
            if (lastLine == j)
                dSW = dS = dSE = 0.f;
            else
            {
                dSW = (0 != i ? FastCompute::Abs(C - labLineNxt[i - 1].L) : 0.f);
                dS  = FastCompute::Abs(C - labLineNxt[i].L);
                dSE = (lastPix < i ? FastCompute::Abs(C - labLineNxt[i + 1].L) : 0.f);
            }

            // PROCESS FUZZY RULES

        } // for (i = 0; i < sizeX; i++)

    } // for (j = 0; j < sizeY; j++)

    return;
}


template <typename T, std::enable_if_t<is_YUV_proc<T>::value>* = nullptr>
inline void FuzzyLogic_3x3
(
    const fCIELabPix* __restrict pLabIn,
    const T* __restrict pIn, /* add Input original (non-filtered) image for get Alpha channels values onl y*/
    const T* __restrict pOut,
    const A_long&          sizeX,
    const A_long&          sizeY,
    const A_long&          labInPitch,
    const A_long&          imgInPitch,
    const A_long&          imgOutPitch
)
{
    return;
}


inline void FuzzyLogic_3x3
(
    const fCIELabPix* __restrict pLabIn,
    const PF_Pixel_RGB_10u* __restrict pIn, /* add Input original (non-filtered) image for get Alpha channels values onl y*/
    const PF_Pixel_RGB_10u* __restrict pOut,
    const A_long&          sizeX,
    const A_long&          sizeY,
    const A_long&          labInPitch,
    const A_long&          imgInPitch,
    const A_long&          imgOutPitch
)
{
}



#endif // __FUZZY_ALGO_LOGIC_KERNEL_3x3__