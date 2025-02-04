#ifndef __FUZZY_ALGO_LOGIC_KERNEL_5x5__
#define __FUZZY_ALGO_LOGIC_KERNEL_5x5__

#include "FuzzyRules.hpp"

/*
    NWW NNW NN NNE NEE
    WNW NW  N   NE ENE
    WW  W   C   E  EE
    WSW SW  S   SE ESE
    SWW SSW SS SSE SEE
*/

template <typename T, std::enable_if_t<is_RGB_proc<T>::value>* = nullptr>
inline void FuzzyLogic_5x5
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
    A_long i, j;
    const A_long lastPix  = sizeX - 1;
    const A_long lastLine = sizeY - 1;
    const A_long preLastPix  = lastPix  - 1;
    const A_long preLastLine = lastLine - 1;

    float iNWW, iNNW, iNN, iNNE, iNEE,
          iWNW, iNW,  iN,  iNE,  iENE,
          iWW,  iW,   iC,  iE,   iEE,
          iWSW, iSW,  iS,  iSE,  iESE,
          iSWW, iSSW, iSS, iSSE, iSEE;

    float dNWW, dNNW, dNN, dNNE, dNEE,
          dWNW, dNW,  dN,  dNE,  dENE,
          dWW,  dW,   dC,  dE,   dEE,
          dWSW, dSW,  dS,  dSE,  dESE,
          dSWW, dSSW, dSS, dSSE, dSEE;

    float fNWW, fNNW, fNN, fNNE, fNEE,
          fWNW, fNW,  fN,  fNE,  fENE,
          fWW,  fW,   fC,  fE,   fEE,
          fWSW, fSW,  fS,  fSE,  fESE,
          fSWW, fSSW, fSS, fSSE, fSEE;

    const float sqSigma = fSigma * fSigma;

    for (j = 0; j < sizeY; j++)
    {
        const fCIELabPix* labLinePrv2 = pLabIn + (j - 2) * labInPitch; // line - 2
        const fCIELabPix* labLinePrv1 = pLabIn + (j - 1) * labInPitch; // line - 1
        const fCIELabPix* labLineCur  = pLabIn + j * labInPitch;       // current line
        const fCIELabPix* labLineNxt1 = pLabIn + (j + 1) * labInPitch; // line + 1
        const fCIELabPix* labLineNxt2 = pLabIn + (j + 2) * labInPitch; // line + 2
        const T* __restrict inOrgLine = pIn  + j * imgInPitch;
              T* __restrict outLine   = pOut + j * imgOutPitch;

        for (i = 0; i < sizeX; i++)
        {
            // CURRENT pixel
            const float C = labLineCur[i].L;


        } // for (i = 0; i < sizeX; i++)

    } // for (j = 0; j < sizeY; j++)

    return;
}


template <typename T, std::enable_if_t<is_YUV_proc<T>::value>* = nullptr>
inline void FuzzyLogic_5x5
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


inline void FuzzyLogic_5x5
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



#endif // __FUZZY_ALGO_LOGIC_KERNEL_5x5__