#include <cstring>
#include <algorithm>
#include "PaintAlgoAVX2.hpp"

// --- CORE PRIMITIVES (Zero Transcendental Math) ---

bool erode_max_plus_optimized
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

        // Instant memory read without calculating 'w'
        if (imOut[j] > imIn[i])
        {
            imOut[j] = imIn[i];
            change = true;
        }
        if (imOut[i] > imIn[j])
        {
            imOut[i] = imIn[j];
            change = true;
        }
    }
    return change;
}

bool dilate_max_plus_optimized
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

        // Instant memory read without calculating '+ w'
        if (imOut[i] < imIn[j])
        {
            imOut[i] = imIn[j];
            change = true;
        }
        if (imOut[j] < imIn[i])
        {
            imOut[j] = imIn[i];
            change = true;
        }
    }
    return change;
}


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
        changed = erode_max_plus_optimized(src, dst, I, J, nonZeros, frameSize);
        src = dst;
        std::swap(dst, alt); // Swap buffers for next iteration
    }
    return const_cast<float*>(src); // Returns the pointer holding the final result
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
        changed = dilate_max_plus_optimized(src, dst, I, J, nonZeros, frameSize);
        src = dst;
        std::swap(dst, alt);
    }
    return const_cast<float*>(src);
}

void morpho_open_optimized
(
    const float* RESTRICT imIn,
    float* RESTRICT imOut,
    const float* RESTRICT pLogW,
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

    // 1. Erode
    float* pEroded = run_erode_iterated(imIn, memHndl.imProc1, memHndl.imProc2, I, J, iter, nonZeros, frameSize);
    
    // Determine which buffer is free to use as the destination for Dilation
    float* pDilateDst = (pEroded == memHndl.imProc1) ? memHndl.imProc2 : memHndl.imProc1;
    float* pDilateAlt = (pEroded == memHndl.imProc1) ? memHndl.imProc1 : memHndl.imProc2;

    // 2. Dilate
    float* pDilated = run_dilate_iterated(pEroded, pDilateDst, pDilateAlt, I, J, iter, nonZeros, frameSize);

    // 3. Copy final result to output
    std::memcpy(imOut, pDilated, frameSize * sizeof(float));
}

void morpho_close_optimized
(
    const float* RESTRICT imIn,
    float* RESTRICT imOut,
    const float* RESTRICT pLogW,
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

    // 1. Dilate
    float* pDilated = run_dilate_iterated(imIn, memHndl.imProc1, memHndl.imProc2, I, J, iter, nonZeros, frameSize);
    
    // Determine free buffer
    float* pErodeDst = (pDilated == memHndl.imProc1) ? memHndl.imProc2 : memHndl.imProc1;
    float* pErodeAlt = (pDilated == memHndl.imProc1) ? memHndl.imProc1 : memHndl.imProc2;

    // 2. Erode
    float* pEroded = run_erode_iterated(pDilated, pErodeDst, pErodeAlt, I, J, iter, nonZeros, frameSize);

    // 3. Copy final result to output
    std::memcpy(imOut, pEroded, frameSize * sizeof(float));
}

void morpho_asf_optimized
(
    const float* RESTRICT imIn,
    float* RESTRICT imOut,
    const float* RESTRICT pLogW,
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
    const float* pCurrentSrc = imIn;
    
    // Alternating Sequential Filter: For k = 1 up to iter...
    // Execute a Close of size k, then an Open of size k
    for (A_long k = 1; k <= iter; ++k)
    {
        // --- CLOSE STAGE (Dilate k times, Erode k times) ---
        float* pDilated = run_dilate_iterated(pCurrentSrc, memHndl.imProc1, memHndl.imProc2, I, J, k, nonZeros, frameSize);
        float* pErodeDst = (pDilated == memHndl.imProc1) ? memHndl.imProc2 : memHndl.imProc1;
        float* pErodeAlt = (pDilated == memHndl.imProc1) ? memHndl.imProc1 : memHndl.imProc2;
        float* pClosed = run_erode_iterated(pDilated, pErodeDst, pErodeAlt, I, J, k, nonZeros, frameSize);

        // --- OPEN STAGE (Erode k times, Dilate k times) ---
        float* pEroded = run_erode_iterated(pClosed, memHndl.imProc1, memHndl.imProc2, I, J, k, nonZeros, frameSize);
        float* pDilateDst = (pEroded == memHndl.imProc1) ? memHndl.imProc2 : memHndl.imProc1;
        float* pDilateAlt = (pEroded == memHndl.imProc1) ? memHndl.imProc1 : memHndl.imProc2;
        float* pOpened = run_dilate_iterated(pEroded, pDilateDst, pDilateAlt, I, J, k, nonZeros, frameSize);

        pCurrentSrc = pOpened; // Feed result into the next scale of the ASF loop
    }

    // Copy final ASF result to output
    std::memcpy(imOut, pCurrentSrc, frameSize * sizeof(float));
}

