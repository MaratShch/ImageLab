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
    const A_long&          imgOutPitch,
    const float&           fSigma = 2.f 
)
{
    A_long i, j;
    const A_long lastPix  = sizeX - 1;
    const A_long lastLine = sizeY - 1;
    float iNW, iN, iNE, iW, iE, iSW, iS, iSE;
    float dNW, dN, dNE, dW, dE, dSW, dS, dSE;
    float fNW, fN, fNE, fW, fE, fSW, fS, fSE;

    const float sqSigma = fSigma * fSigma;

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
                iNW = iN = iNE = C;
            else
            {
                iNW = (0 != i ? labLinePrv[i - 1].L : C);
                iN  = labLinePrv[i].L;
                iNE = (lastPix < i ? labLinePrv[i + 1].L : C);
            }

            // CENTRAL pixels
            iW = (0 != i ? labLineCur[i - 1].L : C);
            iE = (lastPix < i ? labLineCur[i + 1].L : C);

            // SOUTH pixels
            if (lastLine == j)
                iSW = iS = iSE = C;
            else
            {
                iSW = (0 != i ? labLineNxt[i - 1].L : C);
                iS  = labLineNxt[i].L;
                iSE = (lastPix < i ? labLineNxt[i + 1].L : C);
            }

            // PROCESS FUZZY RULES - compute absolute differences
            dNW = FastCompute::Abs(C - iNW);
            dN  = FastCompute::Abs(C - iN);
            dNE = FastCompute::Abs(C - iNE);
            dW  = FastCompute::Abs(C - iW);
            dE  = FastCompute::Abs(C - iE);
            dSW = FastCompute::Abs(C - iSW);
            dS  = FastCompute::Abs(C - iS);
            dSE = FastCompute::Abs(C - iSE);

            fNW = gaussian_sim(dNW, sqSigma);
            fN  = gaussian_sim(dN,  sqSigma);
            fNW = gaussian_sim(dNE, sqSigma);
            fW  = gaussian_sim(dW,  sqSigma);
            fE  = gaussian_sim(dE,  sqSigma);
            fSW = gaussian_sim(dSW, sqSigma);
            fS  = gaussian_sim(dS,  sqSigma);
            fSE = gaussian_sim(dSE, sqSigma);

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
    const A_long&          imgOutPitch,
    const float            fSigma = 2.f
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
    const A_long&          imgOutPitch,
    const float            fSigma = 2.f
)
{
}



#endif // __FUZZY_ALGO_LOGIC_KERNEL_3x3__