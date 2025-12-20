#include <immintrin.h>
#include "dct.hpp"

CACHE_ALIGN constexpr float forward_DCT8_2D_f32[64] =
{
    0.353553f,  0.353553f,  0.353553f,  0.353553f,  0.353553f,  0.353553f,  0.353553f,  0.353553f,
    0.490393f,  0.415735f,  0.277785f,  0.097545f, -0.097545f, -0.277785f, -0.415735f, -0.490393f,
    0.461940f,  0.191342f, -0.191342f, -0.461940f, -0.461940f, -0.191342f,  0.191342f,  0.461940f,
    0.415735f, -0.097545f, -0.490393f, -0.277785f,  0.277785f,  0.490393f,  0.097545f, -0.415735f,
    0.353553f, -0.353553f, -0.353553f,  0.353553f,  0.353553f, -0.353553f, -0.353553f,  0.353553f,
    0.277785f, -0.490393f,  0.097545f,  0.415735f, -0.415735f, -0.097545f,  0.490393f, -0.277785f,
    0.191342f, -0.461940f,  0.461940f, -0.191342f, -0.191342f,  0.461940f, -0.461940f,  0.191342f,
    0.097545f, -0.277785f,  0.415735f, -0.490393f,  0.490393f, -0.415735f,  0.277785f, -0.097545f
};

CACHE_ALIGN constexpr float inverse_DCT8_2D_f32[64] =
{
    0.353553f,  0.490393f,  0.461940f,  0.415735f,  0.353553f,  0.277785f,  0.191342f,  0.097545f,
    0.353553f,  0.415735f,  0.191342f, -0.097545f, -0.353553f, -0.490393f, -0.461940f, -0.277785f,
    0.353553f,  0.277785f, -0.191342f, -0.490393f, -0.353553f,  0.097545f,  0.461940f,  0.415735f,
    0.353553f,  0.097545f, -0.461940f, -0.277785f,  0.353553f,  0.415735f, -0.191342f, -0.490393f,
    0.353553f, -0.097545f, -0.461940f,  0.277785f,  0.353553f, -0.415735f, -0.191342f,  0.490393f,
    0.353553f, -0.277785f, -0.191342f,  0.490393f, -0.353553f, -0.097545f,  0.461940f, -0.415735f,
    0.353553f, -0.415735f,  0.191342f,  0.097545f, -0.353553f,  0.490393f, -0.461940f,  0.277785f,
    0.353553f, -0.490393f,  0.461940f, -0.415735f,  0.353553f, -0.277785f,  0.191342f, -0.097545f
};

CACHE_ALIGN constexpr double forward_DCT8_2D_f64[64] =
{
    0.353553390593273762,  0.353553390593273762,  0.353553390593273762,  0.353553390593273762,
    0.353553390593273762,  0.353553390593273762,  0.353553390593273762,  0.353553390593273762,
    0.490392640201615225,  0.415734806151272619,  0.277785116509801112,  0.097545161008064133,
   -0.097545161008064133, -0.277785116509801112, -0.415734806151272619, -0.490392640201615225,
    0.461939766255643378,  0.191341716182544909, -0.191341716182544909, -0.461939766255643378,
   -0.461939766255643378, -0.191341716182544909,  0.191341716182544909,  0.461939766255643378,
    0.415734806151272619, -0.097545161008064133, -0.490392640201615225, -0.277785116509801112,
    0.277785116509801112,  0.490392640201615225,  0.097545161008064133, -0.415734806151272619,
    0.353553390593273762, -0.353553390593273762, -0.353553390593273762,  0.353553390593273762,
    0.353553390593273762, -0.353553390593273762, -0.353553390593273762,  0.353553390593273762,
    0.277785116509801112, -0.490392640201615225,  0.097545161008064133,  0.415734806151272619,
   -0.415734806151272619, -0.097545161008064133,  0.490392640201615225, -0.277785116509801112,
    0.191341716182544909, -0.461939766255643378,  0.461939766255643378, -0.191341716182544909,
   -0.191341716182544909,  0.461939766255643378, -0.461939766255643378,  0.191341716182544909,
    0.097545161008064133, -0.277785116509801112,  0.415734806151272619, -0.490392640201615225,
    0.490392640201615225, -0.415734806151272619,  0.277785116509801112, -0.097545161008064133
};

CACHE_ALIGN constexpr double inverse_DCT8_2D_f64[64] =
{
    0.353553390593273762,  0.490392640201615225,  0.461939766255643378,  0.415734806151272619,
    0.353553390593273762,  0.277785116509801112,  0.191341716182544909,  0.097545161008064133,
    0.353553390593273762,  0.415734806151272619,  0.191341716182544909, -0.097545161008064133,
   -0.353553390593273762, -0.490392640201615225, -0.461939766255643378, -0.277785116509801112,
    0.353553390593273762,  0.277785116509801112, -0.191341716182544909, -0.490392640201615225,
   -0.353553390593273762,  0.097545161008064133,  0.461939766255643378,  0.415734806151272619,
    0.353553390593273762,  0.097545161008064133, -0.461939766255643378, -0.277785116509801112,
    0.353553390593273762,  0.415734806151272619, -0.191341716182544909, -0.490392640201615225,
    0.353553390593273762, -0.097545161008064133, -0.461939766255643378,  0.277785116509801112,
    0.353553390593273762, -0.415734806151272619, -0.191341716182544909,  0.490392640201615225,
    0.353553390593273762, -0.277785116509801112, -0.191341716182544909,  0.490392640201615225,
   -0.353553390593273762, -0.097545161008064133,  0.461939766255643378, -0.415734806151272619,
    0.353553390593273762, -0.415734806151272619,  0.191341716182544909,  0.097545161008064133,
   -0.353553390593273762,  0.490392640201615225, -0.461939766255643378,  0.277785116509801112,
    0.353553390593273762, -0.490392640201615225,  0.461939766255643378, -0.415734806151272619,
    0.353553390593273762, -0.277785116509801112,  0.191341716182544909, -0.097545161008064133 
};


void FourierTransform::dct_2D
(
    const float* RESTRICT in,
    float* RESTRICT scratch,
    float* RESTRICT out,
    ptrdiff_t width,
    ptrdiff_t height
) noexcept
{
    // PIPELINE STRATEGY:
    // 1. Horizontal DCT:  In (WxH)      -> Scratch (WxH)
    // 2. Transpose:       Scratch (WxH) -> Out (HxW)
    // 3. Vertical DCT:    Out (HxW)     -> Scratch (HxW)  <-- Treated as Horizontal rows
    // 4. Transpose Back:  Scratch (HxW) -> Out (WxH)

    // --- PASS 1: Horizontal Rows ---
    for (ptrdiff_t y = 0; y < height; ++y)
    {
        const float* row_src = in + (y * width);
        float*       row_dst = scratch + (y * width);

        FourierTransform::dct_1D (row_src, row_dst, width);
    }

    // --- PASS 2: Transpose (Flip to Out) ---
    // We move data to 'out' so 'out' holds the Transposed Matrix
    FourierTransform::dct_transpose_block_2D (scratch, out, width, height);

    // --- PASS 3: Vertical Columns (Processing Rows of Transposed Data) ---
    // Note: The image in 'out' has width = height, and height = width
    for (ptrdiff_t i = 0; i < width; ++i)
    {
        // We are reading 'out', processing, and writing back to 'scratch'
        const float* row_src = out + (i * height);
        float*       row_dst = scratch + (i * height);

        // Apply 1D DCT on the "Columns" (which are now Rows)
        FourierTransform::dct_1D (row_src, row_dst, height);
    }

    // --- PASS 4: Transpose Back (Flip to Out) ---
    // We take the result from 'scratch' (HxW) and flip it back to 'out' (WxH)
    FourierTransform::dct_transpose_block_2D (scratch, out, height, width);

    return;
}


void FourierTransform::dct_2D
(
    const double* RESTRICT in,
    double* RESTRICT scratch,
    double* RESTRICT out,
    ptrdiff_t width,
    ptrdiff_t height
) noexcept
{
    // PIPELINE STRATEGY:
    // 1. Horizontal DCT:  In (WxH)      -> Scratch (WxH)
    // 2. Transpose:       Scratch (WxH) -> Out (HxW)
    // 3. Vertical DCT:    Out (HxW)     -> Scratch (HxW)  <-- Treated as Horizontal rows
    // 4. Transpose Back:  Scratch (HxW) -> Out (WxH)

    // --- PASS 1: Horizontal Rows ---
    for (ptrdiff_t y = 0; y < height; ++y)
    {
        const double* row_src = in + (y * width);
        double*       row_dst = scratch + (y * width);

        FourierTransform::dct_1D (row_src, row_dst, width);
    }

    // --- PASS 2: Transpose (Flip to Out) ---
    // We move data to 'out' so 'out' holds the Transposed Matrix
    FourierTransform::dct_transpose_block_2D (scratch, out, width, height);

    // --- PASS 3: Vertical Columns (Processing Rows of Transposed Data) ---
    // Note: The image in 'out' has width = height, and height = width
    for (ptrdiff_t i = 0; i < width; ++i)
    {
        // We are reading 'out', processing, and writing back to 'scratch'
        const double* row_src = out + (i * height);
        double*       row_dst = scratch + (i * height);

        // Apply 1D DCT on the "Columns" (which are now Rows)
        FourierTransform::dct_1D (row_src, row_dst, height);
    }

    // --- PASS 4: Transpose Back (Flip to Out) ---
    // We take the result from 'scratch' (HxW) and flip it back to 'out' (WxH)
    FourierTransform::dct_transpose_block_2D (scratch, out, height, width);

    return;
}


void FourierTransform::idct_2D
(
    const float* RESTRICT in,
    float* RESTRICT scratch,
    float* RESTRICT out,
    ptrdiff_t width,
    ptrdiff_t height
) noexcept
{
    // PIPELINE STRATEGY (Identical to Forward, but using IDCT kernel):
    // 1. Horizontal IDCT: In (WxH)      -> Scratch (WxH)
    // 2. Transpose:       Scratch (WxH) -> Out (HxW)
    // 3. Vertical IDCT:   Out (HxW)     -> Scratch (HxW)
    // 4. Transpose Back:  Scratch (HxW) -> Out (WxH)

    // --- PASS 1: Horizontal Rows (Inverse) ---
    for (int32_t y = 0; y < height; ++y)
    {
        const float* row_src = in + (y * width);
        float*       row_dst = scratch + (y * width);

        FourierTransform::idct_1D(row_src, row_dst, width);
    }

    // --- PASS 2: Transpose (Flip to Out) ---
    FourierTransform::dct_transpose_block_2D(scratch, out, width, height);

    // --- PASS 3: Vertical Columns (Inverse on Transposed Rows) ---
    for (int32_t i = 0; i < width; ++i)
    {
        const float* row_src = out + (i * height);
        float*       row_dst = scratch + (i * height);

        FourierTransform::idct_1D(row_src, row_dst, height);
    }

    // --- PASS 4: Transpose Back (Flip to Out) ---
    FourierTransform::dct_transpose_block_2D(scratch, out, height, width);

    return;
}

void FourierTransform::idct_2D
(
    const double* RESTRICT in,
    double* RESTRICT scratch,
    double* RESTRICT out,
    ptrdiff_t width,
    ptrdiff_t height
) noexcept
{
    // PIPELINE STRATEGY (Identical to Forward, but using IDCT kernel):
    // 1. Horizontal IDCT: In (WxH)      -> Scratch (WxH)
    // 2. Transpose:       Scratch (WxH) -> Out (HxW)
    // 3. Vertical IDCT:   Out (HxW)     -> Scratch (HxW)
    // 4. Transpose Back:  Scratch (HxW) -> Out (WxH)

    // --- PASS 1: Horizontal Rows (Inverse) ---
    for (ptrdiff_t y = 0; y < height; ++y)
    {
        const double* row_src = in + (y * width);
        double*       row_dst = scratch + (y * width);

        FourierTransform::idct_1D(row_src, row_dst, width);
    }

    // --- PASS 2: Transpose (Flip to Out) ---
    FourierTransform::dct_transpose_block_2D(scratch, out, width, height);

    // --- PASS 3: Vertical Columns (Inverse on Transposed Rows) ---
    for (ptrdiff_t i = 0; i < width; ++i)
    {
        const double* row_src = out + (i * height);
        double*       row_dst = scratch + (i * height);

        FourierTransform::idct_1D(row_src, row_dst, height);
    }

    // --- PASS 4: Transpose Back (Flip to Out) ---
    FourierTransform::dct_transpose_block_2D(scratch, out, height, width);

    return;
}


void FourierTransform::dct_generate_transform_matrix_f32
(
    const int N,
    float* RESTRICT pMatrix
)
{
    FourierTransform::GenerateDCTMatrix (N, pMatrix);
    return;
}

void FourierTransform::dct_generate_transform_matrix_f64
(
    const int N,
    double* RESTRICT pMatrix
)
{
    FourierTransform::GenerateDCTMatrix (N, pMatrix);
    return;
}


// Special case DCT 8x8 manually AVX2 optimized
void FourierTransform::dct_2D_8x8
(
    const float* RESTRICT in,
    float* RESTRICT scratch,
    float* RESTRICT out
) noexcept
{
    // TODO ...
    return;
}


// Special case DCT 8x8 manually AVX optimized
void FourierTransform::dct_2D_8x8
(
    const double* RESTRICT in,
    double* RESTRICT scratch,
    double* RESTRICT out
) noexcept
{
    // TODO ...
    return;
}

// Special case IDCT 8x8 manually AVX optimized
void FourierTransform::idct_2D_8x8
(
    const float* RESTRICT in,
    float* RESTRICT scratch,
    float* RESTRICT out
) noexcept
{
    // TODO ...
    return;
}


// Special case DCT 8x8 manually AVX optimized
void FourierTransform::idct_2D_8x8
(
    const double* RESTRICT in,
    double* RESTRICT scratch,
    double* RESTRICT out
) noexcept
{
    // TODO ...
    return;
}

