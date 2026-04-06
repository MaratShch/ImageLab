#include <cstring>
#include <algorithm>
#include <xmmintrin.h> // Required for _mm_prefetch
#include "PaintAlgoAVX2.hpp"


bool erode_max_plus
(
    const float* RESTRICT imIn,
    float* RESTRICT imOut,
    const A_long* RESTRICT I,
    const A_long* RESTRICT J,
    const A_long nLines,
    const A_long frameSize
) noexcept
{
    bool change = false;
    const size_t bytesSize = frameSize * sizeof(float);
    std::memcpy(imOut, imIn, bytesSize); 

    for (A_long l = 0; l < nLines; l++)
    {
        const A_long i = I[l];
        const A_long j = J[l];
        
        const float val_i = imIn[i];
        const float val_j = imIn[j];

        const float out_i = imOut[i];
        const float out_j = imOut[j];

        // Hardware vector min (vminss) - NO BRANCHING
        const float new_j = std::min(out_j, val_i);
        const float new_i = std::min(out_i, val_j);

        // Bitwise change tracking - NO BRANCHING
        if (new_j != out_j || new_i != out_i) {
            change = true;
        }

        imOut[j] = new_j;
        imOut[i] = new_i;
    }
    return change;
}

bool dilate_max_plus
(
    const float* RESTRICT imIn,
    float* RESTRICT imOut,
    const A_long* RESTRICT I,
    const A_long* RESTRICT J,
    const A_long nLines,
    const A_long frameSize
) noexcept
{
    bool change = false;
    const size_t bytesSize = frameSize * sizeof(float);
    std::memcpy(imOut, imIn, bytesSize);

    for (A_long l = 0; l < nLines; l++)
    {
        const A_long i = I[l];
        const A_long j = J[l];
        
        const float val_i = imIn[i];
        const float val_j = imIn[j];

        const float out_i = imOut[i];
        const float out_j = imOut[j];

        // Hardware vector max (vmaxss) - NO BRANCHING
        const float new_j = std::max(out_j, val_i);
        const float new_i = std::max(out_i, val_j);

        // Bitwise change tracking - NO BRANCHING
        if (new_j != out_j || new_i != out_i) {
            change = true;
        }

        imOut[j] = new_j;
        imOut[i] = new_i;
    }
    return change;
}

// ==================================================================================
// --- ITERATORS ---
// ==================================================================================

float* run_erode_iterated
(
    const float* RESTRICT imIn,
    float* RESTRICT bufA,
    float* RESTRICT bufB,
    const A_long* RESTRICT I,
    const A_long* RESTRICT J,
    const A_long iterations,
    const A_long nonZeros,
    const A_long frameSize
) noexcept
{
    const float* src = imIn;
    float* dst = bufA;
    float* alt = bufB;
    bool changed = true;

    for (A_long k = 0; k < iterations && changed; ++k)
    {
        changed = erode_max_plus(src, dst, I, J, nonZeros, frameSize);
        src = dst;
        std::swap(dst, alt);
    }
    return const_cast<float*>(src); 
}

float* run_dilate_iterated
(
    const float* RESTRICT imIn,
    float* RESTRICT bufA,
    float* RESTRICT bufB,
    const A_long* RESTRICT I,
    const A_long* RESTRICT J,
    const A_long iterations,
    const A_long nonZeros,
    const A_long frameSize
) noexcept
{
    const float* src = imIn;
    float* dst = bufA;
    float* alt = bufB;
    bool changed = true;

    for (A_long k = 0; k < iterations && changed; ++k)
    {
        changed = dilate_max_plus(src, dst, I, J, nonZeros, frameSize);
        src = dst;
        std::swap(dst, alt);
    }
    return const_cast<float*>(src);
}

// ==================================================================================
// --- MAIN AGGREGATORS ---
// ==================================================================================

void morpho_open
(
    float* RESTRICT imInOut,
    const A_long* RESTRICT I,
    const A_long* RESTRICT J,
    const A_long iter,
    const A_long nonZeros,
    const A_long width,
    const A_long height,
    const MemHandler& memHndl
) noexcept
{
    const A_long frameSize = width * height;

    float* pEroded = run_erode_iterated(imInOut, memHndl.imProc1, memHndl.imProc2, I, J, iter, nonZeros, frameSize);
    
    float* pDilateDst = (pEroded == memHndl.imProc1) ? memHndl.imProc2 : memHndl.imProc1;
    float* pDilateAlt = (pEroded == memHndl.imProc1) ? memHndl.imProc1 : memHndl.imProc2;

    float* pDilated = run_dilate_iterated(pEroded, pDilateDst, pDilateAlt, I, J, iter, nonZeros, frameSize);

    std::memcpy(imInOut, pDilated, frameSize * sizeof(float));
}

void morpho_close
(
    float* RESTRICT imInOut,
    const A_long* RESTRICT I,
    const A_long* RESTRICT J,
    const A_long iter,
    const A_long nonZeros,
    const A_long width,
    const A_long height,
    const MemHandler& memHndl
) noexcept
{
    const A_long frameSize = width * height;

    float* pDilated = run_dilate_iterated(imInOut, memHndl.imProc1, memHndl.imProc2, I, J, iter, nonZeros, frameSize);
    
    float* pErodeDst = (pDilated == memHndl.imProc1) ? memHndl.imProc2 : memHndl.imProc1;
    float* pErodeAlt = (pDilated == memHndl.imProc1) ? memHndl.imProc1 : memHndl.imProc2;

    float* pEroded = run_erode_iterated(pDilated, pErodeDst, pErodeAlt, I, J, iter, nonZeros, frameSize);

    std::memcpy(imInOut, pEroded, frameSize * sizeof(float));
}

void morpho_asf
(
    float* RESTRICT imInOut,
    const A_long* RESTRICT I,
    const A_long* RESTRICT J,
    const A_long iter,
    const A_long nonZeros,
    const A_long width,
    const A_long height,
    const MemHandler& memHndl
) noexcept
{
    const A_long frameSize = width * height;
    const float* pCurrentSrc = imInOut;
    
    for (A_long k = 1; k <= iter; ++k)
    {
        float* pDilated = run_dilate_iterated(pCurrentSrc, memHndl.imProc1, memHndl.imProc2, I, J, k, nonZeros, frameSize);
        float* pErodeDst = (pDilated == memHndl.imProc1) ? memHndl.imProc2 : memHndl.imProc1;
        float* pErodeAlt = (pDilated == memHndl.imProc1) ? memHndl.imProc1 : memHndl.imProc2;
        float* pClosed = run_erode_iterated(pDilated, pErodeDst, pErodeAlt, I, J, k, nonZeros, frameSize);

        float* pEroded = run_erode_iterated(pClosed, memHndl.imProc1, memHndl.imProc2, I, J, k, nonZeros, frameSize);
        float* pDilateDst = (pEroded == memHndl.imProc1) ? memHndl.imProc2 : memHndl.imProc1;
        float* pDilateAlt = (pEroded == memHndl.imProc1) ? memHndl.imProc1 : memHndl.imProc2;
        float* pOpened = run_dilate_iterated(pEroded, pDilateDst, pDilateAlt, I, J, k, nonZeros, frameSize);

        pCurrentSrc = pOpened; 
    }

    std::memcpy(imInOut, pCurrentSrc, frameSize * sizeof(float));
}