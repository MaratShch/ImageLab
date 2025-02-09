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
    const T&               blackPix,  // black (minimal) color pixel value - used for clamping
    const T&               whitePix,  // white (maximal) color pixel value - used for clamping
    const float&           fSigma = 2.f 
) noexcept
{
    A_long i, j;
    const A_long lastPix  = sizeX - 1;
    const A_long lastLine = sizeY - 1;
    float iNW, iN, iNE, iW, iE, iSW, iS, iSE;
    float dNW, dN, dNE, dW, dE, dSW, dS, dSE;
    float fNW, fN, fNE, fW, fE, fSW, fS, fSE;
    float val1, val2;

    const float sqSigma = fSigma * fSigma;

    for (j = 0; j < sizeY; j++)
    {
        const fCIELabPix* __restrict labLinePrv = pLabIn + (j - 1) * labInPitch; // line -1
        const fCIELabPix* __restrict labLineCur = pLabIn +  j * labInPitch;      // current line
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

            fNW = gaussian_sim(dNW, 0.f, sqSigma);
            fN  = gaussian_sim(dN,  0.f, sqSigma);
            fNE = gaussian_sim(dNE, 0.f, sqSigma);
            fW  = gaussian_sim(dW,  0.f, sqSigma);
            fE  = gaussian_sim(dE,  0.f, sqSigma);
            fSW = gaussian_sim(dSW, 0.f, sqSigma);
            fS  = gaussian_sim(dS,  0.f, sqSigma);
            fSE = gaussian_sim(dSE, 0.f, sqSigma);

            // weighted average
            val1 = fNW + fN + fNE + fW + fE + fSW + fS + fSE;
             
            val2 = fNW * iNW + fN  * iN  + fNE * iNE + fW  * iW +
                   fE  * iE  + fSW * iSW + fS  * iS  + fSE * iSE;

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
    const T&               blackPix,  // black (minimal) color pixel value  - used for clamping
    const T&               whitePix,  // white (maximal) color pixel value - used for clamping
    const float            fSigma = 2.f,
    const eCOLOR_SPACE&    colorSpace = BT709
) noexcept
{
    A_long i, j;
    const A_long lastPix = sizeX - 1;
    const A_long lastLine = sizeY - 1;
    float iNW, iN, iNE, iW, iE, iSW, iS, iSE;
    float dNW, dN, dNE, dW, dE, dSW, dS, dSE;
    float fNW, fN, fNE, fW, fE, fSW, fS, fSE;
    float val1, val2;

    float fUVAdd = 128.f;
    if (std::is_same<T, PF_Pixel_VUYA_16u>::value)
        fUVAdd = 0.f;
    else if (std::is_same<T, PF_Pixel_VUYA_32f>::value)
        fUVAdd = 0.f;

    const float sqSigma = fSigma * fSigma;

    for (j = 0; j < sizeY; j++)
    {
        const fCIELabPix* __restrict labLinePrv = pLabIn + (j - 1) * labInPitch; // line -1
        const fCIELabPix* __restrict labLineCur = pLabIn +  j * labInPitch;      // current line
        const fCIELabPix* __restrict labLineNxt = pLabIn + (j + 1) * labInPitch; // line +1
        const T* __restrict inOrgLine = pIn + j * imgInPitch;
              T* __restrict outLine  = pOut + j * imgOutPitch;

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
                iN = labLinePrv[i].L;
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
                iS = labLineNxt[i].L;
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

            fNW = gaussian_sim(dNW, 0.f, sqSigma);
            fN  = gaussian_sim(dN, 0.f, sqSigma);
            fNE = gaussian_sim(dNE, 0.f, sqSigma);
            fW  = gaussian_sim(dW, 0.f, sqSigma);
            fE  = gaussian_sim(dE, 0.f, sqSigma);
            fSW = gaussian_sim(dSW, 0.f, sqSigma);
            fS  = gaussian_sim(dS, 0.f, sqSigma);
            fSE = gaussian_sim(dSE, 0.f, sqSigma);

            // weighted average
            val1 = fNW + fN + fNE + fW + fE + fSW + fS + fSE;

            val2 = fNW * iNW + fN  * iN  + fNE * iNE + fW  * iW +
                   fE  * iE  + fSW * iSW + fS  * iS  + fSE * iSE;

            fCIELabPix filteredPix;
            filteredPix.L = val2 / val1;
            filteredPix.a = labLineCur[i].a;
            filteredPix.b = labLineCur[i].b;

            const fYUV outPix = fRgb2Yuv(Xyz2Rgb(CieLab2Xyz(filteredPix)), colorSpace);

            outLine[i].A = inOrgLine[i].A; // copy Alpha-channel from sources buffer 'as-is'
            outLine[i].Y = static_cast<decltype(outLine[i].Y)>(CLAMP_VALUE(outPix.Y * whitePix.Y,          static_cast<float>(blackPix.Y), static_cast<float>(whitePix.Y)));
            outLine[i].U = static_cast<decltype(outLine[i].U)>(CLAMP_VALUE(outPix.U * whitePix.U + fUVAdd, static_cast<float>(blackPix.U), static_cast<float>(whitePix.U)));
            outLine[i].V = static_cast<decltype(outLine[i].V)>(CLAMP_VALUE(outPix.V * whitePix.V + fUVAdd, static_cast<float>(blackPix.V), static_cast<float>(whitePix.V)));

        } // for (i = 0; i < sizeX; i++)

    } // for (j = 0; j < sizeY; j++)

    return;
}


inline void FuzzyLogic_3x3
(
    const fCIELabPix* __restrict pLabIn,
    const PF_Pixel_RGB_10u* __restrict pIn, /* add Input original (non-filtered) image for get Alpha channels values onl y*/
          PF_Pixel_RGB_10u* __restrict pOut,
    const A_long&          sizeX,
    const A_long&          sizeY,
    const A_long&          labInPitch,
    const A_long&          imgInPitch,
    const A_long&          imgOutPitch,
    const PF_Pixel_RGB_10u&  blackPix, // black (minimal) color pixel value - used for clamping
    const PF_Pixel_RGB_10u&  whitePix, // white (maximal) color pixel value - used for clamping
    const float            fSigma = 2.f
) noexcept
{
    A_long i, j;
    const A_long lastPix  = sizeX - 1;
    const A_long lastLine = sizeY - 1;
    float iNW, iN, iNE, iW, iE, iSW, iS, iSE;
    float dNW, dN, dNE, dW, dE, dSW, dS, dSE;
    float fNW, fN, fNE, fW, fE, fSW, fS, fSE;
    float val1, val2;

    const float sqSigma = fSigma * fSigma;

    for (j = 0; j < sizeY; j++)
    {
        const fCIELabPix* __restrict labLinePrv = pLabIn + (j - 1) * labInPitch; // line -1
        const fCIELabPix* __restrict labLineCur = pLabIn + j * labInPitch;       // current line
        const fCIELabPix* __restrict labLineNxt = pLabIn + (j + 1) * labInPitch; // line +1
        const PF_Pixel_RGB_10u* __restrict inOrgLine = pIn  + j * imgInPitch;
              PF_Pixel_RGB_10u* __restrict outLine   = pOut + j * imgOutPitch;

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
                iN = labLinePrv[i].L;
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
                iS = labLineNxt[i].L;
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

            fNW = gaussian_sim(dNW, 0.f, sqSigma);
            fN  = gaussian_sim(dN,  0.f, sqSigma);
            fNE = gaussian_sim(dNE, 0.f, sqSigma);
            fW  = gaussian_sim(dW,  0.f, sqSigma);
            fE  = gaussian_sim(dE,  0.f, sqSigma);
            fSW = gaussian_sim(dSW, 0.f, sqSigma);
            fS  = gaussian_sim(dS,  0.f, sqSigma);
            fSE = gaussian_sim(dSE, 0.f, sqSigma);

            // weighted average
            val1 = fNW + fN + fNE + fW + fE + fSW + fS + fSE;

            val2 = fNW * iNW + fN  * iN  + fNE * iNE + fW  * iW +
                   fE  * iE  + fSW * iSW + fS  * iS  + fSE * iSE;

            fCIELabPix filteredPix;
            filteredPix.L = val2 / val1;
            filteredPix.a = labLineCur[i].a;
            filteredPix.b = labLineCur[i].b;

            const fRGB outPix = Xyz2Rgb(CieLab2Xyz(filteredPix));

            outLine[i].R = static_cast<decltype(outLine[i].R)>(CLAMP_VALUE(outPix.R * whitePix.R, static_cast<float>(blackPix.R), static_cast<float>(whitePix.R)));
            outLine[i].G = static_cast<decltype(outLine[i].G)>(CLAMP_VALUE(outPix.G * whitePix.G, static_cast<float>(blackPix.G), static_cast<float>(whitePix.G)));
            outLine[i].B = static_cast<decltype(outLine[i].B)>(CLAMP_VALUE(outPix.B * whitePix.B, static_cast<float>(blackPix.B), static_cast<float>(whitePix.B)));

        } // for (i = 0; i < sizeX; i++)

    } // for (j = 0; j < sizeY; j++)

    return;
}


#endif // __FUZZY_ALGO_LOGIC_KERNEL_3x3__