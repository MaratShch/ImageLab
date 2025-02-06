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
    const A_long lastPix     = sizeX - 1;
    const A_long lastLine    = sizeY - 1;
    const A_long preLastPix  = sizeX - 2;
    const A_long preLastLine = sizeY - 2;

    float iNWW, iNNW, iNN, iNNE, iNEE,
          iWNW, iNW,  iN,  iNE,  iENE,
          iWW,  iW,        iE,   iEE,
          iWSW, iSW,  iS,  iSE,  iESE,
          iSWW, iSSW, iSS, iSSE, iSEE;

    float dNWW, dNNW, dNN, dNNE, dNEE,
          dWNW, dNW,  dN,  dNE,  dENE,
          dWW,  dW,        dE,   dEE,
          dWSW, dSW,  dS,  dSE,  dESE,
          dSWW, dSSW, dSS, dSSE, dSEE;

    float fNWW, fNNW, fNN, fNNE, fNEE,
          fWNW, fNW,  fN,  fNE,  fENE,
          fWW,  fW,        fE,   fEE,
          fWSW, fSW,  fS,  fSE,  fESE,
          fSWW, fSSW, fSS, fSSE, fSEE;

    float val1, val2;

    const float sqSigma = fSigma * fSigma;

    for (j = 0; j < sizeY; j++)
    {
        const fCIELabPix* labLinePrv2 = pLabIn + (j - 2) * labInPitch; // line - 2
        const fCIELabPix* labLinePrv1 = pLabIn + (j - 1) * labInPitch; // line - 1
        const fCIELabPix* labLineCur  = pLabIn +  j      * labInPitch; // current line
        const fCIELabPix* labLineNxt1 = pLabIn + (j + 1) * labInPitch; // line + 1
        const fCIELabPix* labLineNxt2 = pLabIn + (j + 2) * labInPitch; // line + 2
        const T* __restrict inOrgLine = pIn  + j * imgInPitch;
              T* __restrict outLine   = pOut + j * imgOutPitch;

        for (i = 0; i < sizeX; i++)
        {
            // CURRENT pixel
            const float C = labLineCur[i].L;

            // NORTH pixels
            if (0 == j)
                iNWW = iNNW = iNN = iNNE = iNEE = iWNW = iNW = iN = iNE = iENE = C;
            else if (1 == j)
            {
                iNWW = iNWW = iNN = iNNE = iNEE = C;

                iWNW = (i > 1 ? labLinePrv1[i - 2].L : C);
                iNW  = (i > 0 ? labLinePrv1[i - 1].L : C);
                iN   = labLinePrv1[i].L;
                iNE  = (i < lastPix    ? labLinePrv1[i + 1].L : C);
                iENE = (i < preLastPix ? labLinePrv1[i + 2].L : C);
            }
            else
            {
                iNWW = (i > 1 ? labLinePrv2[i - 2].L : C);
                iNNW = (i > 0 ? labLinePrv2[i - 1].L : C);
                iNN = labLinePrv2[i].L;
                iNNE = (i < lastPix    ? labLinePrv2[i + 1].L : C);
                iNEE = (i < preLastPix ? labLinePrv2[i + 2].L : C);

                iWNW = (i > 1 ? labLinePrv1[i - 2].L : C);
                iNW  = (i > 0 ? labLinePrv1[i - 1].L : C);
                iN   = labLinePrv1[i].L;
                iNE  = (i < lastPix    ? labLinePrv1[i + 1].L : C);
                iENE = (i < preLastPix ? labLinePrv1[i + 2].L : C);
            }

            // CENTRAL PIXELS
            iWW = (i > 1 ? labLineCur[i - 2].L : C);
            iW  = (i > 0 ? labLineCur[i - 1].L : C);
            iE  = (i < lastPix    ? labLineCur[i + 1].L : C);
            iEE = (i < preLastPix ? labLineCur[i + 2].L : C);

            // SOUTH PIXELS
            if (j == lastLine)
                iWSW = iSW = iS = iSE = iESE = iSWW = iSSW = iSS = iSSE = iSEE = C;
            else if (j == preLastLine)
            {
                iSWW = iSSW = iSS = iSSE = iSEE = C;
                iWSW = (i > 1 ? labLineNxt1[i - 2].L : C);
                iSW  = (i > 0 ? labLineNxt1[i - 1].L : C);
                iS   = labLineNxt1[i].L;
                iSE  = (i < lastPix    ? labLineNxt1[i + 1].L : C);
                iESE = (i < preLastPix ? labLineNxt1[i + 2].L : C);
            }
            else
            {
                iWSW = (i > 1 ? labLineNxt1[i - 2].L : C);
                iSW  = (i > 0 ? labLineNxt1[i - 1].L : C);
                iS   = labLineNxt1[i].L;
                iSE  = (i < lastPix    ? labLineNxt1[i + 1].L : C);
                iESE = (i < preLastPix ? labLineNxt1[i + 2].L : C);
                iSWW = (i > 1 ? labLineNxt2[i - 2].L : C); 
                iSSW = (i > 0 ? labLineNxt2[i - 1].L : C); 
                iSS  = labLineNxt2[i].L;
                iSSE = (i < lastPix    ? labLineNxt2[i + 1].L : C); 
                iSEE = (i < preLastPix ? labLineNxt2[i + 2].L : C);
            }

            // PROCESS FUZZY RULES - compute absolute differences
            dNWW = FastCompute::Abs(C - iNWW);
            dNNW = FastCompute::Abs(C - iNNW);
            dNN  = FastCompute::Abs(C - iNN);
            dNNE = FastCompute::Abs(C - iNNE);
            dNEE = FastCompute::Abs(C - iNEE);
            dWNW = FastCompute::Abs(C - iWNW);
            dNW  = FastCompute::Abs(C - iNW);
            dN   = FastCompute::Abs(C - iN);
            dNE  = FastCompute::Abs(C - iNE);
            dENE = FastCompute::Abs(C - iENE);
            dWW  = FastCompute::Abs(C - iWW);
            dW   = FastCompute::Abs(C - iW);
            dE   = FastCompute::Abs(C - iE);
            dEE  = FastCompute::Abs(C - iEE);
            dWSW = FastCompute::Abs(C - iWSW);
            dSW  = FastCompute::Abs(C - iSW);
            dS   = FastCompute::Abs(C - iS);
            dSE  = FastCompute::Abs(C - iSE);
            dESE = FastCompute::Abs(C - iESE);
            dSWW = FastCompute::Abs(C - iSWW);
            dSSW = FastCompute::Abs(C - iSSW);
            dSS  = FastCompute::Abs(C - iSS);
            dSSE = FastCompute::Abs(C - iSSE);
            dSEE = FastCompute::Abs(C - iSEE);

            fNWW = gaussian_sim(dNWW, 0.f, sqSigma);
            fNNW = gaussian_sim(dNNW, 0.f, sqSigma);
            fNN  = gaussian_sim(dNN,  0.f, sqSigma);
            fNNE = gaussian_sim(dNNE, 0.f, sqSigma);
            fNEE = gaussian_sim(dNEE, 0.f, sqSigma);
            fWNW = gaussian_sim(dWNW, 0.f, sqSigma);
            fNW  = gaussian_sim(dNW,  0.f, sqSigma);
            fN   = gaussian_sim(dN,   0.f, sqSigma);
            fNE  = gaussian_sim(dNE,  0.f, sqSigma);
            fENE = gaussian_sim(dENE, 0.f, sqSigma);
            fWW  = gaussian_sim(dWW,  0.f, sqSigma);
            fW   = gaussian_sim(dW,   0.f, sqSigma);
            fE   = gaussian_sim(dE,   0.f, sqSigma);
            fEE  = gaussian_sim(dEE,  0.f, sqSigma);
            fWSW = gaussian_sim(dWSW, 0.f, sqSigma);
            fSW  = gaussian_sim(dSW,  0.f, sqSigma);
            fS   = gaussian_sim(dS,   0.f, sqSigma);
            fSE  = gaussian_sim(dSE,  0.f, sqSigma);
            fESE = gaussian_sim(dESE, 0.f, sqSigma);
            fSWW = gaussian_sim(dSWW, 0.f, sqSigma);
            fSSW = gaussian_sim(dSSW, 0.f, sqSigma);
            fSS  = gaussian_sim(dSS,  0.f, sqSigma);
            fSSE = gaussian_sim(dSSE, 0.f, sqSigma);
            fSEE = gaussian_sim(dSEE, 0.f, sqSigma);

            // weighted average
            val1 = fNWW + fNNW + fNN + fNNE + fNEE + fWNW + fNW  + fN  + fNE  + fENE + fWW + fW + fE + fEE +
                   fWSW + fSW  + fS  + fSE  + fESE + fSWW + fSSW + fSS + fSSE + fSEE;

            val2 = fNWW * iNWW + fNNW * iNNW + fNN  * iNN  +  fNNE * iNNE + 
                   fNEE * iNEE + fWNW * iWNW + fNW  * iNW  +  fN   * iN   + 
                   fNE  * iNE  + fENE * iENE + fWW  * iWW  +  fW   * iW   +
                   fE   * iE   + fEE  * iEE  + fWSW * iWSW +  fSW  * iSW  + 
                   fS   * iS   + fSE  * iSE  + fESE * iESE +  fSWW * iSWW + 
                   fSSW * iSSW + fSS  * iSS  + fSSE * iSSE +  fSEE * iSEE;

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
    A_long i, j;
    const A_long lastPix     = sizeX - 1;
    const A_long lastLine    = sizeY - 1;
    const A_long preLastPix  = sizeX - 2;
    const A_long preLastLine = sizeY - 2;

    float iNWW, iNNW, iNN, iNNE, iNEE,
          iWNW, iNW,  iN,  iNE,  iENE,
          iWW,  iW,        iE,   iEE,
          iWSW, iSW,  iS,  iSE,  iESE,
          iSWW, iSSW, iSS, iSSE, iSEE;

    float dNWW, dNNW, dNN, dNNE, dNEE,
          dWNW, dNW,  dN,  dNE,  dENE,
          dWW,  dW,        dE,   dEE,
          dWSW, dSW,  dS,  dSE,  dESE,
          dSWW, dSSW, dSS, dSSE, dSEE;

    float fNWW, fNNW, fNN, fNNE, fNEE,
          fWNW, fNW,  fN,  fNE,  fENE,
          fWW,  fW,        fE,   fEE,
          fWSW, fSW,  fS,  fSE,  fESE,
          fSWW, fSSW, fSS, fSSE, fSEE;

    float val1, val2;

    const float sqSigma = fSigma * fSigma;

    for (j = 0; j < sizeY; j++)
    {
        const fCIELabPix* labLinePrv2 = pLabIn + (j - 2) * labInPitch; // line - 2
        const fCIELabPix* labLinePrv1 = pLabIn + (j - 1) * labInPitch; // line - 1
        const fCIELabPix* labLineCur  = pLabIn +  j      * labInPitch; // current line
        const fCIELabPix* labLineNxt1 = pLabIn + (j + 1) * labInPitch; // line + 1
        const fCIELabPix* labLineNxt2 = pLabIn + (j + 2) * labInPitch; // line + 2
        const PF_Pixel_RGB_10u* __restrict inOrgLine = pIn + j * imgInPitch;
              PF_Pixel_RGB_10u* __restrict outLine = pOut + j * imgOutPitch;

        for (i = 0; i < sizeX; i++)
        {
            // CURRENT pixel
            const float C = labLineCur[i].L;

            // NORTH pixels
            if (0 == j)
                iNWW = iNNW = iNN = iNNE = iNEE = iWNW = iNW = iN = iNE = iENE = C;
            else if (1 == j)
            {
                iNWW = iNWW = iNN = iNNE = iNEE = C;

                iWNW = (i > 1 ? labLinePrv1[i - 2].L : C);
                iNW  = (i > 0 ? labLinePrv1[i - 1].L : C);
                iN   = labLinePrv1[i].L;
                iNE  = (i < lastPix    ? labLinePrv1[i + 1].L : C);
                iENE = (i < preLastPix ? labLinePrv1[i + 2].L : C);
            }
            else
            {
                iNWW = (i > 1 ? labLinePrv2[i - 2].L : C);
                iNNW = (i > 0 ? labLinePrv2[i - 1].L : C);
                iNN  = labLinePrv2[i].L;
                iNNE = (i < lastPix    ? labLinePrv2[i + 1].L : C);
                iNEE = (i < preLastPix ? labLinePrv2[i + 2].L : C);

                iWNW = (i > 1 ? labLinePrv1[i - 2].L : C);
                iNW  = (i > 0 ? labLinePrv1[i - 1].L : C);
                iN   = labLinePrv1[i].L;
                iNE  = (i < lastPix    ? labLinePrv1[i + 1].L : C);
                iENE = (i < preLastPix ? labLinePrv1[i + 2].L : C);
            }

            // CENTRAL PIXELS
            iWW = (i > 1 ? labLineCur[i - 2].L : C);
            iW  = (i > 0 ? labLineCur[i - 1].L : C);
            iE  = (i < lastPix    ? labLineCur[i + 1].L : C);
            iEE = (i < preLastPix ? labLineCur[i + 2].L : C);

            // SOUTH PIXELS
            if (j == lastLine)
                iWSW = iSW = iS = iSE = iESE = iSWW = iSSW = iSS = iSSE = iSEE = C;
            else if (j == preLastLine)
            {
                iSWW = iSSW = iSS = iSSE = iSEE = C;
                iWSW = (i > 1 ? labLineNxt1[i - 2].L : C);
                iSW  = (i > 0 ? labLineNxt1[i - 1].L : C);
                iS   = labLineNxt1[i].L;
                iSE  = (i < lastPix    ? labLineNxt1[i + 1].L : C);
                iESE = (i < preLastPix ? labLineNxt1[i + 2].L : C);
            }
            else
            {
                iWSW = (i > 1 ? labLineNxt1[i - 2].L : C);
                iSW  = (i > 0 ? labLineNxt1[i - 1].L : C);
                iS   = labLineNxt1[i].L;
                iSE  = (i < lastPix    ? labLineNxt1[i + 1].L : C);
                iESE = (i < preLastPix ? labLineNxt1[i + 2].L : C);
                iSWW = (i > 1 ? labLineNxt2[i - 2].L : C);
                iSSW = (i > 0 ? labLineNxt2[i - 1].L : C);
                iSS  = labLineNxt2[i].L;
                iSSE = (i < lastPix    ? labLineNxt2[i + 1].L : C);
                iSEE = (i < preLastPix ? labLineNxt2[i + 2].L : C);
            }

            // PROCESS FUZZY RULES - compute absolute differences
            dNWW = FastCompute::Abs(C - iNWW);
            dNNW = FastCompute::Abs(C - iNNW);
            dNN  = FastCompute::Abs(C - iNN);
            dNNE = FastCompute::Abs(C - iNNE);
            dNEE = FastCompute::Abs(C - iNEE);
            dWNW = FastCompute::Abs(C - iWNW);
            dNW  = FastCompute::Abs(C - iNW);
            dN   = FastCompute::Abs(C - iN);
            dNE  = FastCompute::Abs(C - iNE);
            dENE = FastCompute::Abs(C - iENE);
            dWW  = FastCompute::Abs(C - iWW);
            dW   = FastCompute::Abs(C - iW);
            dE   = FastCompute::Abs(C - iE);
            dEE  = FastCompute::Abs(C - iEE);
            dWSW = FastCompute::Abs(C - iWSW);
            dSW  = FastCompute::Abs(C - iSW);
            dS   = FastCompute::Abs(C - iS);
            dSE  = FastCompute::Abs(C - iSE);
            dESE = FastCompute::Abs(C - iESE);
            dSWW = FastCompute::Abs(C - iSWW);
            dSSW = FastCompute::Abs(C - iSSW);
            dSS  = FastCompute::Abs(C - iSS);
            dSSE = FastCompute::Abs(C - iSSE);
            dSEE = FastCompute::Abs(C - iSEE);

            fNWW = gaussian_sim(dNWW, 0.f, sqSigma);
            fNNW = gaussian_sim(dNNW, 0.f, sqSigma);
            fNN  = gaussian_sim(dNN,  0.f, sqSigma);
            fNNE = gaussian_sim(dNNE, 0.f, sqSigma);
            fNEE = gaussian_sim(dNEE, 0.f, sqSigma);
            fWNW = gaussian_sim(dWNW, 0.f, sqSigma);
            fNW  = gaussian_sim(dNW,  0.f, sqSigma);
            fN   = gaussian_sim(dN,   0.f, sqSigma);
            fNE  = gaussian_sim(dNE,  0.f, sqSigma);
            fENE = gaussian_sim(dENE, 0.f, sqSigma);
            fWW  = gaussian_sim(dWW,  0.f, sqSigma);
            fW   = gaussian_sim(dW,   0.f, sqSigma);
            fE   = gaussian_sim(dE,   0.f, sqSigma);
            fEE  = gaussian_sim(dEE,  0.f, sqSigma);
            fWSW = gaussian_sim(dWSW, 0.f, sqSigma);
            fSW  = gaussian_sim(dSW,  0.f, sqSigma);
            fS   = gaussian_sim(dS,   0.f, sqSigma);
            fSE  = gaussian_sim(dSE,  0.f, sqSigma);
            fESE = gaussian_sim(dESE, 0.f, sqSigma);
            fSWW = gaussian_sim(dSWW, 0.f, sqSigma);
            fSSW = gaussian_sim(dSSW, 0.f, sqSigma);
            fSS  = gaussian_sim(dSS,  0.f, sqSigma);
            fSSE = gaussian_sim(dSSE, 0.f, sqSigma);
            fSEE = gaussian_sim(dSEE, 0.f, sqSigma);

            // weighted average
            val1 = fNWW + fNNW + fNN + fNNE + fNEE + fWNW + fNW  + fN  + fNE  + fENE + fWW + fW + fE + fEE +
                   fWSW + fSW  + fS  + fSE  + fESE + fSWW + fSSW + fSS + fSSE + fSEE;

            val2 = fNWW * iNWW + fNNW * iNNW + fNN  * iNN  + fNNE * iNNE +
                   fNEE * iNEE + fWNW * iWNW + fNW  * iNW  + fN   * iN   +
                   fNE  * iNE  + fENE * iENE + fWW  * iWW  + fW   * iW   +
                   fE   * iE   + fEE  * iEE  + fWSW * iWSW + fSW  * iSW  +
                   fS   * iS   + fSE  * iSE  + fESE * iESE + fSWW * iSWW +
                   fSSW * iSSW + fSS  * iSS  + fSSE * iSSE + fSEE * iSEE;

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



#endif // __FUZZY_ALGO_LOGIC_KERNEL_5x5__