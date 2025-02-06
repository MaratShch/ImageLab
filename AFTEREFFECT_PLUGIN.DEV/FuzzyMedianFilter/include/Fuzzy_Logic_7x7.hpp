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
    // Don't mark this pixels as Nort, South, etc... just assign numbers from left to rigth and from top to down
    CACHE_ALIGN float iVal[48];
    CACHE_ALIGN float dVal[48];
    CACHE_ALIGN float fVal[48];

    A_long i, j;
    const A_long lastPix     = sizeX - 1;
    const A_long lastLine    = sizeY - 1;
    const A_long preLastPix  = sizeX - 2;
    const A_long preLastLine = sizeY - 2;
    const A_long prepreLastPix  = sizeX - 3;
    const A_long prepreLastLine = sizeY - 3;

    float val1, val2;
    const float sqSigma = fSigma * fSigma;

    for (j = 0; j < sizeY; j++)
    {
        const fCIELabPix* labLinePrv3 = pLabIn + (j - 3) * labInPitch; // line - 3
        const fCIELabPix* labLinePrv2 = pLabIn + (j - 2) * labInPitch; // line - 2
        const fCIELabPix* labLinePrv1 = pLabIn + (j - 1) * labInPitch; // line - 1
        const fCIELabPix* labLineCur  = pLabIn +  j      * labInPitch; // current line
        const fCIELabPix* labLineNxt1 = pLabIn + (j + 1) * labInPitch; // line + 1
        const fCIELabPix* labLineNxt2 = pLabIn + (j + 2) * labInPitch; // line + 2
        const fCIELabPix* labLineNxt3 = pLabIn + (j + 3) * labInPitch; // line + 3
        const T* __restrict inOrgLine = pIn  + j * imgInPitch;
              T* __restrict outLine   = pOut + j * imgOutPitch;

        for (i = 0; i < sizeX; i++)
        {
            // CURRENT pixel
            const float C = labLineCur[i].L;

            __VECTOR_ALIGNED__
            for (int k = 0; k < 48; k += 8)
            {
                dVal[k + 0] = FastCompute::Abs(C - iVal[k + 0]);
                dVal[k + 1] = FastCompute::Abs(C - iVal[k + 1]);
                dVal[k + 2] = FastCompute::Abs(C - iVal[k + 2]);
                dVal[k + 3] = FastCompute::Abs(C - iVal[k + 3]);
                dVal[k + 4] = FastCompute::Abs(C - iVal[k + 4]);
                dVal[k + 5] = FastCompute::Abs(C - iVal[k + 5]);
                dVal[k + 6] = FastCompute::Abs(C - iVal[k + 6]);
                dVal[k + 7] = FastCompute::Abs(C - iVal[k + 7]);
            }

            __VECTOR_ALIGNED__
            for (int k = 0; k < 48; k += 8)
            {
                fVal[k + 0] = gaussian_sim(dVal[k + 0], 0.f, sqSigma);
                fVal[k + 1] = gaussian_sim(dVal[k + 1], 0.f, sqSigma);
                fVal[k + 2] = gaussian_sim(dVal[k + 2], 0.f, sqSigma);
                fVal[k + 3] = gaussian_sim(dVal[k + 3], 0.f, sqSigma);
                fVal[k + 4] = gaussian_sim(dVal[k + 4], 0.f, sqSigma);
                fVal[k + 5] = gaussian_sim(dVal[k + 5], 0.f, sqSigma);
                fVal[k + 6] = gaussian_sim(dVal[k + 6], 0.f, sqSigma);
                fVal[k + 7] = gaussian_sim(dVal[k + 7], 0.f, sqSigma);
            }

            val1 = val2 = 0.f;
            __VECTOR_ALIGNED__
            for (int k = 0; k < 48; k ++)
            {
                val1 += fVal[k];
                val2 += (fVal[k] * iVal[k]);
            }

            fCIELabPix filteredPix;
            filteredPix.L = val2 / val1;
            filteredPix.a = labLineCur[i].a;
            filteredPix.b = labLineCur[i].b;

            const fRGB outPix = Xyz2Rgb(CieLab2Xyz(filteredPix));

            outLine[i].A = inOrgLine[i].A; // copy Alpha-channel from sources buffer 'as-is'
            outLine[i].R = static_cast<decltype(outLine[i].R)>(CLAMP_VALUE(outPix.R * whitePix.R, static_cast<float>(blackPix.R), static_cast<float>(whitePix.R)));
            outLine[i].G = static_cast<decltype(outLine[i].G)>(CLAMP_VALUE(outPix.G * whitePix.G, static_cast<float>(blackPix.G), static_cast<float>(whitePix.G)));
            outLine[i].B = static_cast<decltype(outLine[i].B)>(CLAMP_VALUE(outPix.B * whitePix.B, static_cast<float>(blackPix.B), static_cast<float>(whitePix.B)));

        } // for (i = 0; i < sizeX; i++)

    } // for (j = 0; j < sizeY; j++)
            
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
    // Don't mark this pixels as Nort, South, etc... just assign numbers from left to rigth and from top to down
    CACHE_ALIGN float iVal[48];
    CACHE_ALIGN float dVal[48];
    CACHE_ALIGN float fVal[48];


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
    // Don't mark this pixels as Nort, South, etc... just assign numbers from left to rigth and from top to down
    CACHE_ALIGN float iVal[48];
    CACHE_ALIGN float dVal[48];
    CACHE_ALIGN float fVal[48];


    return;
}



#endif // __FUZZY_ALGO_LOGIC_KERNEL_7x7__