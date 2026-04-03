#include "PaintMorpologyKernels.hpp"


bool erode_max_plus_symmetric
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
    memcpy(imOut, imIn, bytesSize);

    for (A_long l = 0; l < nLines; l++)
    {
        const A_long i = I[l];
        const A_long j = J[l];

        const float out_j = imOut[j];
        const float in_i  = imIn[i];
        
        if (out_j > in_i)
        {
            imOut[j] = in_i;
            change = true;
        }

        const float out_i = imOut[i];
        const float in_j  = imIn[j];
        
        if (out_i > in_j)
        {
            imOut[i] = in_j;
            change = true;
        }
    }
    return change;
}

bool dilate_max_plus_symmetric
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
    memcpy(imOut, imIn, bytesSize);

    for (A_long l = 0; l < nLines; l++)
    {
        const A_long i = I[l];
        const A_long j = J[l];

        const float out_i = imOut[i];
        const float in_j  = imIn[j];

        if (out_i < in_j)
        {
            imOut[i] = in_j;
            change = true;
        }

        const float out_j = imOut[j];
        const float in_i  = imIn[i];

        if (out_j < in_i)
        {
            imOut[j] = in_i;
            change = true;
        }
    }
    return change;
}


int erode_max_plus_symmetric_iterated
(
    const A_long* RESTRICT I,
    const A_long* RESTRICT J,
    const float* RESTRICT imIn,
    float* RESTRICT imOut[],
    const A_long k,
    const A_long n_lines,
    float** pOut,
    const A_long frameSize
) noexcept
{
    const float* RESTRICT imgSrc = imIn;
    float* RESTRICT imgDst = imOut[0];
    A_long iteration = 0;
    bool changed = true;

    while (iteration < k && true == changed)
    {
        changed = erode_max_plus_symmetric(imgSrc, imgDst, I, J, n_lines, frameSize);

        if (nullptr != pOut) { *pOut = imgDst; }

        iteration++;
        imgSrc = imgDst;
        imgDst = imOut[iteration & 0x1];
    }

    return (true == changed ? k : iteration - 1);
}


int dilate_max_plus_symmetric_iterated
(
    const A_long* RESTRICT I,
    const A_long* RESTRICT J,
    const float* RESTRICT imIn,
    float* RESTRICT imOut[],
    const A_long k,
    const A_long n_lines,
    float** pOut,
    const A_long frameSize
) noexcept
{
    A_long iteration = 0;
    bool changed = true;

    const float* RESTRICT imgSrc = imIn;
    float* RESTRICT imgDst = imOut[0];

    while (iteration < k && true == changed)
    {
        changed = dilate_max_plus_symmetric(imgSrc, imgDst, I, J, n_lines, frameSize);

        if (nullptr != pOut) { *pOut = imgDst; }

        iteration++;
        imgSrc = imgDst;
        imgDst = imOut[iteration & 0x1];
    }

    return (true == changed ? k : iteration - 1);
}

A_long morpho_open
(
    float* RESTRICT imIn,
    float* RESTRICT imOut,
    const A_long* RESTRICT I,
    const A_long* RESTRICT J,
    A_long it,
    A_long nonZeros,
    A_long sizeX,
    A_long sizeY,
    const MemHandler& memHndl
) noexcept
{
    const A_long frameSize = sizeX * sizeY;
    float* RESTRICT im_proc[2] { memHndl.imProc1, memHndl.imProc2 };
    float* pOut = nullptr;

    // 1. Erode
    const A_long kMax = erode_max_plus_symmetric_iterated(I, J, imIn, im_proc, it, nonZeros, &pOut, frameSize);
    const size_t memSize = frameSize * sizeof(float);

    if (kMax == it)
    {
        // 2. Dilate
        memcpy(imIn, pOut, memSize);
        dilate_max_plus_symmetric_iterated(I, J, imIn, im_proc, it, nonZeros, &pOut, frameSize);
    } 

    memcpy(imOut, pOut, memSize);
    return 0;
}

A_long morpho_close
(
    float* RESTRICT imIn,
    float* RESTRICT imOut,
    const A_long* RESTRICT I,
    const A_long* RESTRICT J,
    A_long it,
    A_long nonZeros,
    A_long sizeX,
    A_long sizeY,
    const MemHandler& memHndl
) noexcept
{
    const A_long frameSize = sizeX * sizeY;
    float* RESTRICT im_proc[2] { memHndl.imProc1, memHndl.imProc2 };
    float* pOut = nullptr;

    // 1. Dilate
    const A_long kMax = dilate_max_plus_symmetric_iterated(I, J, imIn, im_proc, it, nonZeros, &pOut, frameSize);
    const size_t memSize = frameSize * sizeof(float);

    if (kMax == it)
    {
        // 2. Erode
        memcpy(imIn, pOut, memSize);
        erode_max_plus_symmetric_iterated(I, J, imIn, im_proc, it, nonZeros, &pOut, frameSize);
    } 

    memcpy(imOut, pOut, memSize);
    return 0;
}

A_long morpho_asf
(
    float* RESTRICT imIn,
    float* RESTRICT imOut,
    const A_long* RESTRICT I,
    const A_long* RESTRICT J,
    A_long it,
    A_long nonZeros,
    A_long sizeX,
    A_long sizeY,
    const MemHandler& memHndl
) noexcept
{
    // Alternating Sequential Filter: Open followed by Close
    morpho_open(imIn, imOut, I, J, it, nonZeros, sizeX, sizeY, memHndl);
    
    // Use imOut as the input for the close operation to chain them
    morpho_close(imOut, imOut, I, J, it, nonZeros, sizeX, sizeY, memHndl);
    
    return 0;
}


