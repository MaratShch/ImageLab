#ifndef __IMAGE_LAB_AUTHOMATIC_WB_ALGO_COOMON_FUNCTIONS__
#define __IMAGE_LAB_AUTHOMATIC_WB_ALGO_COOMON_FUNCTIONS__

#include <immintrin.h>
#include "ColorTransformMatrix.hpp"
#include "FastAriphmetics.hpp"


template <typename T>
inline void simple_image_copy
(
	const T* __restrict srcPix,
	      T* __restrict dstPix,
	const A_long width,
	const A_long height,
	const A_long srcPitch,
	const A_long dstPitch
) noexcept
{
	const size_t line_size = width * sizeof(T);
	__VECTOR_ALIGNED__
	for (A_long i = 0; i < height; i++)
	{
		std::memcpy(&dstPix[i*dstPitch], &srcPix[i*srcPitch], line_size);
	}

	return;
}


template <typename T, std::enable_if_t<is_RGB_proc<T>::value>* = nullptr>
inline void collect_rgb_statistics
(
    const T* __restrict pSrc,
    const A_long width,
    const A_long height,
    const A_long linePitch,
    const float threshold,
    const eCOLOR_SPACE colorSpace,
    float* u_Avg,
    float* v_Avg
) noexcept
{
    float U_bar = 0.f, V_bar = 0.f, F = 0.f;
    float Y, U, V;
    int32_t totalGray = 0;

    const float* __restrict colorMatrixIn = RGB2YUV[colorSpace];

    __VECTOR_ALIGNED__
    for (A_long j = 0; j < height; j++)
    {
        const A_long l_idx = j * linePitch; /* line IDX */
        for (A_long i = 0; i < width; i++)
        {
            const A_long p_idx = l_idx + i; /* pixel IDX */
                                            /* convert RGB to YUV color space */
            Y = pSrc[p_idx].R * colorMatrixIn[0] + pSrc[p_idx].G * colorMatrixIn[1] + pSrc[p_idx].B * colorMatrixIn[2];
            U = pSrc[p_idx].R * colorMatrixIn[3] + pSrc[p_idx].G * colorMatrixIn[4] + pSrc[p_idx].B * colorMatrixIn[5];
            V = pSrc[p_idx].R * colorMatrixIn[6] + pSrc[p_idx].G * colorMatrixIn[7] + pSrc[p_idx].B * colorMatrixIn[8];

            F = (FastCompute::Abs(U) + FastCompute::Abs(V)) / FastCompute::Max(Y, FLT_EPSILON);
            if (F < threshold)
            {
                totalGray++;
                U_bar += U;
                V_bar += V;
            } /* if (F < T) */

        } /* for (i = 0; i < width; i++) */

    } /* for (j = 0; j < height; j++) */

    if (nullptr != u_Avg)
        *u_Avg = U_bar / static_cast<float>(totalGray);
    if (nullptr != v_Avg)
        *v_Avg = V_bar / static_cast<float>(totalGray);

    return;
}


template <typename T, std::enable_if_t<is_YUV_proc<T>::value>* = nullptr>
inline void collect_yuv_statistics
(
    const T* __restrict pSrc,
    const A_long width,
    const A_long height,
    const A_long linePitch,
    const float threshold,
    const eCOLOR_SPACE colorSpace,
    float* u_Avg,
    float* v_Avg
) noexcept
{
    float U_bar = 0.f, V_bar = 0.f, F = 0.f;
    float Y, U, V;
    int32_t totalGray = 0;

    const float* __restrict colorMatrixIn = RGB2YUV[colorSpace];

    float subtractor = 0.0f;
    if (std::is_same<T, PF_Pixel_VUYA_8u>::value)
        subtractor = 128.0f;

    __VECTOR_ALIGNED__
        for (A_long j = 0; j < height; j++)
        {
            const A_long l_idx = j * linePitch; /* line IDX */
            for (A_long i = 0; i < width; i++)
            {
                const A_long p_idx = l_idx + i; /* pixel IDX */
                                                /* convert RGB to YUV color space */
                Y = pSrc[p_idx].Y;
                U = pSrc[p_idx].U - subtractor;
                V = pSrc[p_idx].V - subtractor;

                F = (FastCompute::Abs(U) + FastCompute::Abs(V)) / FastCompute::Max(Y, FLT_EPSILON);
                if (F < threshold)
                {
                    totalGray++;
                    U_bar += U;
                    V_bar += V;
                } /* if (F < T) */

            } /* for (i = 0; i < width; i++) */

        } /* for (j = 0; j < height; j++) */

    if (nullptr != u_Avg)
        *u_Avg = U_bar / static_cast<float>(totalGray);
    if (nullptr != v_Avg)
        *v_Avg = V_bar / static_cast<float>(totalGray);

    return;
}


template <typename T, std::enable_if_t<is_RGB_proc<T>::value>* = nullptr>
inline void image_rgb_correction
(
    const T* __restrict pSrc,		/* input data  */
    T* __restrict pDst,				/* output data */
    const A_long width,
    const A_long height,
    const A_long srcPitch,
    const A_long dstPitch,
    const float* __restrict correctionMatrix
) noexcept
{
    float newR, newG, newB;

    float whiteValue = static_cast<float>(u8_value_white);
    if (std::is_same<T, PF_Pixel_BGRA_16u>::value || std::is_same<T, PF_Pixel_ARGB_16u>::value)
        whiteValue = static_cast<float>(u16_value_white);
    else if (std::is_same<T, PF_Pixel_BGRA_32f>::value || std::is_same<T, PF_Pixel_ARGB_32f>::value)
        whiteValue = f32_value_white;

    /* in second: perform balance based on computed coefficients */
    for (A_long j = 0; j < height; j++)
    {
        const A_long l_idx_src = j * srcPitch;
        const A_long l_idx_dst = j * dstPitch;

        __VECTOR_ALIGNED__
            for (A_long i = 0; i < width; i++)
            {
                const A_long p_idx_src = l_idx_src + i;
                const A_long p_idx_dst = l_idx_dst + i;

                newR = correctionMatrix[0] * pSrc[p_idx_src].R;
                newG = correctionMatrix[1] * pSrc[p_idx_src].G;
                newB = correctionMatrix[2] * pSrc[p_idx_src].B;

                pDst[p_idx_dst].A = pSrc[p_idx_src].A; /* copy ALPHA channel from source */
                pDst[p_idx_dst].R = static_cast<decltype(pDst[p_idx_dst].R)>(CLAMP_VALUE(newR, f32_value_black, whiteValue));
                pDst[p_idx_dst].G = static_cast<decltype(pDst[p_idx_dst].G)>(CLAMP_VALUE(newG, f32_value_black, whiteValue));
                pDst[p_idx_dst].B = static_cast<decltype(pDst[p_idx_dst].B)>(CLAMP_VALUE(newB, f32_value_black, whiteValue));

            } /* for (i = 0; i < width; i++) */

    } /* for (j = 0; j < height; j++) */

    return;
}


template <typename T, std::enable_if_t<is_YUV_proc<T>::value>* = nullptr>
inline void image_yuv_correction
(
    const T* __restrict pSrc,		/* input data  */
    T* __restrict pDst,				/* output data */
    const A_long width,
    const A_long height,
    const A_long srcPitch,
    const A_long dstPitch,
    const float* __restrict correctionMatrix,
    const bool& isBT709
) noexcept
{
    float R, G, B;
    float newR, newG, newB;
    float newY, newU, newV;

    float whiteValue = static_cast<float>(f32_value_white);
    float subtractor = 0.0f;
    if (std::is_same<T, PF_Pixel_VUYA_8u>::value)
        whiteValue = static_cast<float>(u8_value_white), subtractor = 128.0f;

    const float* __restrict yuv2rgb = ((true == isBT709) ? YUV2RGB[BT709] : YUV2RGB[BT601]);
    const float* __restrict rgb2yuv = ((true == isBT709) ? RGB2YUV[BT709] : RGB2YUV[BT601]);

    /* in second: perform balance based on computed coefficients */
    for (A_long j = 0; j < height; j++)
    {
        const A_long l_idx_src = j * srcPitch;
        const A_long l_idx_dst = j * dstPitch;

        __VECTOR_ALIGNED__
            for (A_long i = 0; i < width; i++)
            {
                const A_long p_idx_src = l_idx_src + i;
                const A_long p_idx_dst = l_idx_dst + i;

                R = pSrc[p_idx_src].Y * yuv2rgb[0] + (pSrc[p_idx_src].U - subtractor) * yuv2rgb[1] + (pSrc[p_idx_src].V - subtractor) * yuv2rgb[2];
                G = pSrc[p_idx_src].Y * yuv2rgb[3] + (pSrc[p_idx_src].U - subtractor) * yuv2rgb[4] + (pSrc[p_idx_src].V - subtractor) * yuv2rgb[5];
                B = pSrc[p_idx_src].Y * yuv2rgb[6] + (pSrc[p_idx_src].U - subtractor) * yuv2rgb[7] + (pSrc[p_idx_src].V - subtractor) * yuv2rgb[8];

                newR = CLAMP_VALUE(correctionMatrix[0] * R, 0.f, whiteValue);
                newG = CLAMP_VALUE(correctionMatrix[1] * G, 0.f, whiteValue);
                newB = CLAMP_VALUE(correctionMatrix[2] * B, 0.f, whiteValue);

                newY = newR * rgb2yuv[0] + newG * rgb2yuv[1] + newB * rgb2yuv[2];
                newU = newR * rgb2yuv[3] + newG * rgb2yuv[4] + newB * rgb2yuv[5];
                newV = newR * rgb2yuv[6] + newG * rgb2yuv[7] + newB * rgb2yuv[8];

                pDst[p_idx_dst].A = pSrc[p_idx_src].A; /* copy ALPHA channel from source */
                pDst[p_idx_dst].Y = static_cast<decltype(pDst[p_idx_dst].Y)>(newY);
                pDst[p_idx_dst].U = static_cast<decltype(pDst[p_idx_dst].U)>(newU + subtractor);
                pDst[p_idx_dst].V = static_cast<decltype(pDst[p_idx_dst].V)>(newV + subtractor);

            } /* for (i = 0; i < width; i++) */

    } /* for (j = 0; j < height; j++) */

    return;
}


inline void collect_rgb_statistics
(
    const PF_Pixel_BGRA_32f* __restrict pSrc,
    const A_long width,
    const A_long height,
    const A_long linePitch,
    const float threshold,
    const eCOLOR_SPACE colorSpace,
    float* u_Avg,
    float* v_Avg
) noexcept
{
    float U_bar = 0.f, V_bar = 0.f, F = 0.f;
    float Y, U, V;
    int32_t totalGray = 0;

    const float* __restrict colorMatrixIn = RGB2YUV[colorSpace];

    __VECTOR_ALIGNED__
        for (A_long j = 0; j < height; j++)
        {
            const A_long l_idx = j * linePitch; /* line IDX */
            for (A_long i = 0; i < width; i++)
            {
                const A_long p_idx = l_idx + i; /* pixel IDX */
                                                /* convert RGB to YUV color space */
                const PF_Pixel_BGRA_32f& inPixel = pSrc[p_idx];
                const float R = inPixel.R * 255.0f;
                const float G = inPixel.G * 255.0f;
                const float B = inPixel.B * 255.0f;

                Y = R * colorMatrixIn[0] + G * colorMatrixIn[1] + B * colorMatrixIn[2];
                U = R * colorMatrixIn[3] + G * colorMatrixIn[4] + B * colorMatrixIn[5];
                V = R * colorMatrixIn[6] + G * colorMatrixIn[7] + B * colorMatrixIn[8];

                F = (FastCompute::Abs(U) + FastCompute::Abs(V)) / FastCompute::Max(Y, FLT_EPSILON);
                if (F < threshold)
                {
                    totalGray++;
                    U_bar += U;
                    V_bar += V;
                } /* if (F < T) */

            } /* for (i = 0; i < width; i++) */

        } /* for (j = 0; j < height; j++) */

    if (nullptr != u_Avg)
        *u_Avg = U_bar / static_cast<float>(totalGray);
    if (nullptr != v_Avg)
        *v_Avg = V_bar / static_cast<float>(totalGray);

    return;
}


inline void collect_yuv_statistics
(
    const PF_Pixel_VUYA_32f* __restrict pSrc,
    const A_long width,
    const A_long height,
    const A_long linePitch,
    const float threshold,
    const eCOLOR_SPACE colorSpace,
    float* u_Avg,
    float* v_Avg
) noexcept
{
    float U_bar = 0.f, V_bar = 0.f, F = 0.f;
    float Y, U, V;
    int32_t totalGray = 0;

    __VECTOR_ALIGNED__
        for (A_long j = 0; j < height; j++)
        {
            const A_long l_idx = j * linePitch; /* line IDX */
            for (A_long i = 0; i < width; i++)
            {
                const A_long p_idx = l_idx + i; /* pixel IDX */

                Y = pSrc[p_idx].Y * 255.0f;
                U = pSrc[p_idx].U * 255.0f;
                V = pSrc[p_idx].V * 255.0f;

                F = (FastCompute::Abs(U) + FastCompute::Abs(V)) / FastCompute::Max(Y, FLT_EPSILON);
                if (F < threshold)
                {
                    totalGray++;
                    U_bar += U;
                    V_bar += V;
                } /* if (F < T) */

            } /* for (i = 0; i < width; i++) */

        } /* for (j = 0; j < height; j++) */

    if (nullptr != u_Avg)
        *u_Avg = U_bar / static_cast<float>(totalGray);
    if (nullptr != v_Avg)
        *v_Avg = V_bar / static_cast<float>(totalGray);

    return;
}


inline void collect_rgb_statistics
(
    const PF_Pixel_BGRA_8u* __restrict pSrc,
    const A_long width,
    const A_long height,
    const A_long linePitch,
    const float threshold,
    const eCOLOR_SPACE colorSpace,
    float* u_Avg,
    float* v_Avg
) noexcept
{
    // Accumulators
    float U_bar = 0.f;
    float V_bar = 0.f;
    int32_t totalGray = 0;

    // Load Color Matrix
    const float* __restrict cm = RGB2YUV[colorSpace];

    // Broadcast Matrix Coefficients to AVX2 registers
    // Y = R*m0 + G*m1 + B*m2
    const __m256 v_m0 = _mm256_set1_ps(cm[0]);
    const __m256 v_m1 = _mm256_set1_ps(cm[1]);
    const __m256 v_m2 = _mm256_set1_ps(cm[2]);
    // U = R*m3 + G*m4 + B*m5
    const __m256 v_m3 = _mm256_set1_ps(cm[3]);
    const __m256 v_m4 = _mm256_set1_ps(cm[4]);
    const __m256 v_m5 = _mm256_set1_ps(cm[5]);
    // V = R*m6 + G*m7 + B*m8
    const __m256 v_m6 = _mm256_set1_ps(cm[6]);
    const __m256 v_m7 = _mm256_set1_ps(cm[7]);
    const __m256 v_m8 = _mm256_set1_ps(cm[8]);

    // Constants
    const __m256 v_threshold = _mm256_set1_ps(threshold);
    const __m256 v_epsilon = _mm256_set1_ps(FLT_EPSILON);
    const __m256 v_zero = _mm256_setzero_ps();
    // Mask to clear sign bit (for Abs)
    const __m256 v_abs_mask = _mm256_castsi256_ps(_mm256_set1_epi32(0x7FFFFFFF));

    // Shuffle masks for de-interleaving BGRA 8-bit to 32-bit Integers
    // We process 8 pixels. We need 3 masks to extract B, G, R into 3 separate registers.
    // -1 in shuffle control zeroes the byte. 
    // We want 32-bit integers: 00 00 00 Byte.
    // Indices for _mm256_shuffle_epi8 work within 128-bit lanes. 
    // Pattern repeats for both lanes.
    const __m256i mask_B = _mm256_setr_epi8(
        0, -1, -1, -1, 4, -1, -1, -1, 8, -1, -1, -1, 12, -1, -1, -1,
        0, -1, -1, -1, 4, -1, -1, -1, 8, -1, -1, -1, 12, -1, -1, -1);

    const __m256i mask_G = _mm256_setr_epi8(
        1, -1, -1, -1, 5, -1, -1, -1, 9, -1, -1, -1, 13, -1, -1, -1,
        1, -1, -1, -1, 5, -1, -1, -1, 9, -1, -1, -1, 13, -1, -1, -1);

    const __m256i mask_R = _mm256_setr_epi8(
        2, -1, -1, -1, 6, -1, -1, -1, 10, -1, -1, -1, 14, -1, -1, -1,
        2, -1, -1, -1, 6, -1, -1, -1, 10, -1, -1, -1, 14, -1, -1, -1);

    // AVX Accumulators
    __m256 v_acc_U = _mm256_setzero_ps();
    __m256 v_acc_V = _mm256_setzero_ps();
    __m256i v_acc_Gray = _mm256_setzero_si256();

    // Loop over height
    for (A_long j = 0; j < height; ++j)
    {
        const uint8_t* __restrict pRow = reinterpret_cast<const uint8_t*>(pSrc + (j * linePitch));
        A_long i = 0;

        // Vectorized Loop (8 pixels per iteration)
        // 8 pixels * 4 bytes = 32 bytes
        const A_long vecWidth = width - 7;

        for (; i < vecWidth; i += 8)
        {
            // 1. Load 32 bytes (8 pixels of BGRA)
            // Use unaligned load as safe default for image buffers
            __m256i v_pixels = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(pRow + i * 4));

            // 2. De-interleave to Planar 32-bit Integers using Shuffle
            // Result: B0 0 0 0, B1 0 0 0... (interpreted as int32)
            __m256i v_B_i = _mm256_shuffle_epi8(v_pixels, mask_B);
            __m256i v_G_i = _mm256_shuffle_epi8(v_pixels, mask_G);
            __m256i v_R_i = _mm256_shuffle_epi8(v_pixels, mask_R);

            // 3. Convert to Float
            __m256 v_B = _mm256_cvtepi32_ps(v_B_i);
            __m256 v_G = _mm256_cvtepi32_ps(v_G_i);
            __m256 v_R = _mm256_cvtepi32_ps(v_R_i);

            // 4. Calculate Y, U, V (Fused Multiply Add)
            // Y = R*m0 + G*m1 + B*m2
            __m256 v_Y = _mm256_fmadd_ps(v_R, v_m0, _mm256_fmadd_ps(v_G, v_m1, _mm256_mul_ps(v_B, v_m2)));
            // U = R*m3 + G*m4 + B*m5
            __m256 v_U = _mm256_fmadd_ps(v_R, v_m3, _mm256_fmadd_ps(v_G, v_m4, _mm256_mul_ps(v_B, v_m5)));
            // V = R*m6 + G*m7 + B*m8
            __m256 v_V = _mm256_fmadd_ps(v_R, v_m6, _mm256_fmadd_ps(v_G, v_m7, _mm256_mul_ps(v_B, v_m8)));

            // 5. Calculate Metric F
            // F = (|U| + |V|) / Max(Y, epsilon)
            // Optimization: Avoid division by checking (|U| + |V|) < Threshold * Max(Y, eps)

            // Abs(U) and Abs(V)
            __m256 v_abs_U = _mm256_and_ps(v_U, v_abs_mask);
            __m256 v_abs_V = _mm256_and_ps(v_V, v_abs_mask);
            __m256 v_sum_abs = _mm256_add_ps(v_abs_U, v_abs_V);

            // Max(Y, epsilon)
            __m256 v_Y_safe = _mm256_max_ps(v_Y, v_epsilon);

            // RHS = Threshold * Y_safe
            __m256 v_rhs = _mm256_mul_ps(v_threshold, v_Y_safe);

            // 6. Compare: F_numerator < RHS
            // mask contains all 1s (NaN) if true, 0 otherwise
            __m256 v_cmp_mask = _mm256_cmp_ps(v_sum_abs, v_rhs, _CMP_LT_OQ);

            // 7. Accumulate
            // Mask out values that didn't pass
            v_acc_U = _mm256_add_ps(v_acc_U, _mm256_and_ps(v_cmp_mask, v_U));
            v_acc_V = _mm256_add_ps(v_acc_V, _mm256_and_ps(v_cmp_mask, v_V));

            // Accumulate Count
            // Cast mask (float) to int. True = -1 (0xFFFFFFFF), False = 0.
            // Subtracting -1 is equivalent to Adding 1.
            __m256i v_cmp_mask_i = _mm256_castps_si256(v_cmp_mask);
            v_acc_Gray = _mm256_sub_epi32(v_acc_Gray, v_cmp_mask_i);
        }

        // Handle Tail (Scalar)
        for (; i < width; ++i)
        {
            // Standard BGRA layout assumption: B=0, G=1, R=2
            const uint8_t* px = pRow + i * 4;
            float b = static_cast<float>(px[0]);
            float g = static_cast<float>(px[1]);
            float r = static_cast<float>(px[2]);

            float Y = r * cm[0] + g * cm[1] + b * cm[2];
            float U = r * cm[3] + g * cm[4] + b * cm[5];
            float V = r * cm[6] + g * cm[7] + b * cm[8];

            float F = (FastCompute::Abs(U) + FastCompute::Abs(V)) / FastCompute::Max(Y, FLT_EPSILON);

            if (F < threshold)
            {
                totalGray++;
                U_bar += U;
                V_bar += V;
            }
        }
    }

    // Horizontal Reduction of AVX Accumulators
    // 1. Sum up the 8 elements in v_acc_U/V/Gray

    // Reduce U
    // [0 1 2 3 4 5 6 7] -> [0+4 1+5 2+6 3+7 ...]
    __m256 v_temp_U = _mm256_add_ps(v_acc_U, _mm256_permute2f128_ps(v_acc_U, v_acc_U, 1));
    // [0+4 1+5 ...] -> [0+4+2+6 ...]
    v_temp_U = _mm256_hadd_ps(v_temp_U, v_temp_U);
    v_temp_U = _mm256_hadd_ps(v_temp_U, v_temp_U);
    // Extract scalar
    U_bar += _mm_cvtss_f32(_mm256_castps256_ps128(v_temp_U));

    // Reduce V
    __m256 v_temp_V = _mm256_add_ps(v_acc_V, _mm256_permute2f128_ps(v_acc_V, v_acc_V, 1));
    v_temp_V = _mm256_hadd_ps(v_temp_V, v_temp_V);
    v_temp_V = _mm256_hadd_ps(v_temp_V, v_temp_V);
    V_bar += _mm_cvtss_f32(_mm256_castps256_ps128(v_temp_V));

    // Reduce Count (Integer)
    // Extract upper 128
    __m128i v_low = _mm256_castsi256_si128(v_acc_Gray);
    __m128i v_high = _mm256_extracti128_si256(v_acc_Gray, 1);
    v_low = _mm_add_epi32(v_low, v_high);
    // Horizontal int add: (x, y, z, w) -> (x+y, z+w, ...)
    v_low = _mm_hadd_epi32(v_low, v_low);
    v_low = _mm_hadd_epi32(v_low, v_low);
    totalGray += _mm_cvtsi128_si32(v_low);

    // Final Calculation
    if (totalGray > 0)
    {
        float invGray = 1.0f / static_cast<float>(totalGray);
        if (nullptr != u_Avg) *u_Avg = U_bar * invGray;
        if (nullptr != v_Avg) *v_Avg = V_bar * invGray;
    }
    else
    {
        if (nullptr != u_Avg) *u_Avg = 0.f;
        if (nullptr != v_Avg) *v_Avg = 0.f;
    }

    return;
}



inline void image_rgb_correction_BGRA_8u
(
    const PF_Pixel_BGRA_8u* __restrict pSrc,
    PF_Pixel_BGRA_8u*       __restrict pDst,
    const A_long width,
    const A_long height,
    const A_long srcPitch,
    const A_long dstPitch,
    const float* __restrict correctionMatrix
) noexcept
{
    // 1. Prepare AVX2 Constants
    // -------------------------------------------------------------------------
    const __m256 vScaleR = _mm256_set1_ps(correctionMatrix[0]);
    const __m256 vScaleG = _mm256_set1_ps(correctionMatrix[1]);
    const __m256 vScaleB = _mm256_set1_ps(correctionMatrix[2]);

    const __m256 vZero = _mm256_setzero_ps();
    const __m256 v255 = _mm256_set1_ps(255.0f);
    const __m256i vAlphaMask = _mm256_set1_epi32(static_cast<int>(0xFF000000));

    // Shuffle Masks (for de-interleaving BGRA -> Planar)
    const __m256i mask_B = _mm256_setr_epi8(
        0, -1, -1, -1, 4, -1, -1, -1, 8, -1, -1, -1, 12, -1, -1, -1,
        0, -1, -1, -1, 4, -1, -1, -1, 8, -1, -1, -1, 12, -1, -1, -1);

    const __m256i mask_G = _mm256_setr_epi8(
        1, -1, -1, -1, 5, -1, -1, -1, 9, -1, -1, -1, 13, -1, -1, -1,
        1, -1, -1, -1, 5, -1, -1, -1, 9, -1, -1, -1, 13, -1, -1, -1);

    const __m256i mask_R = _mm256_setr_epi8(
        2, -1, -1, -1, 6, -1, -1, -1, 10, -1, -1, -1, 14, -1, -1, -1,
        2, -1, -1, -1, 6, -1, -1, -1, 10, -1, -1, -1, 14, -1, -1, -1);

    // 2. Execution Loop
    // -------------------------------------------------------------------------
    for (A_long j = 0; j < height; ++j)
    {
        const uint8_t* __restrict pRowSrc = reinterpret_cast<const uint8_t*>(pSrc) + (j * srcPitch * 4);
        uint8_t*       __restrict pRowDst = reinterpret_cast<uint8_t*>(pDst) + (j * dstPitch * 4);

        A_long i = 0;
        const A_long vecWidth = width - 7;

        // [AVX2 Kernel] Processes 8 pixels at once
        for (; i < vecWidth; i += 8)
        {
            // Load 32 bytes (8 pixels)
            __m256i vPixels = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(pRowSrc + i * 4));

            // Convert to Planar Floats
            __m256 vB_f = _mm256_cvtepi32_ps(_mm256_shuffle_epi8(vPixels, mask_B));
            __m256 vG_f = _mm256_cvtepi32_ps(_mm256_shuffle_epi8(vPixels, mask_G));
            __m256 vR_f = _mm256_cvtepi32_ps(_mm256_shuffle_epi8(vPixels, mask_R));

            // Apply Correction
            vB_f = _mm256_mul_ps(vB_f, vScaleB);
            vG_f = _mm256_mul_ps(vG_f, vScaleG);
            vR_f = _mm256_mul_ps(vR_f, vScaleR);

            // Vector Clamp (Equivalent to CLAMP_VALUE but hardware accelerated)
            vB_f = _mm256_min_ps(v255, _mm256_max_ps(vZero, vB_f));
            vG_f = _mm256_min_ps(v255, _mm256_max_ps(vZero, vG_f));
            vR_f = _mm256_min_ps(v255, _mm256_max_ps(vZero, vR_f));

            // Pack back to BGRA Integers
            __m256i vB_i = _mm256_cvtps_epi32(vB_f);
            __m256i vG_i = _mm256_cvtps_epi32(vG_f);
            __m256i vR_i = _mm256_cvtps_epi32(vR_f);

            // Reconstruct: B | (G << 8) | (R << 16) | Original Alpha
            __m256i vResult = vB_i;
            vResult = _mm256_or_si256(vResult, _mm256_slli_epi32(vG_i, 8));
            vResult = _mm256_or_si256(vResult, _mm256_slli_epi32(vR_i, 16));
            vResult = _mm256_or_si256(vResult, _mm256_and_si256(vPixels, vAlphaMask));

            _mm256_storeu_si256(reinterpret_cast<__m256i*>(pRowDst + i * 4), vResult);
        }

        // [Scalar Tail] Processes remaining pixels (0-7)
        for (; i < width; ++i)
        {
            const PF_Pixel_BGRA_8u* pxSrc = reinterpret_cast<const PF_Pixel_BGRA_8u*>(pRowSrc + i * 4);
            PF_Pixel_BGRA_8u*       pxDst = reinterpret_cast<PF_Pixel_BGRA_8u*>(pRowDst + i * 4);

            float newR = correctionMatrix[0] * static_cast<float>(pxSrc->R);
            float newG = correctionMatrix[1] * static_cast<float>(pxSrc->G);
            float newB = correctionMatrix[2] * static_cast<float>(pxSrc->B);

            // Using your CLAMP_VALUE function here
            pxDst->R = static_cast<uint8_t>(CLAMP_VALUE(newR, 0.0f, 255.0f));
            pxDst->G = static_cast<uint8_t>(CLAMP_VALUE(newG, 0.0f, 255.0f));
            pxDst->B = static_cast<uint8_t>(CLAMP_VALUE(newB, 0.0f, 255.0f));
            pxDst->A = pxSrc->A;
        }
    }

    return;
}

#endif // __IMAGE_LAB_AUTHOMATIC_WB_ALGO_COOMON_FUNCTIONS__