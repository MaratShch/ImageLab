#ifndef __IMAGE_LAB2_POINTILLISM_EFFECT_COLOR_CONVERT_APIS__
#define __IMAGE_LAB2_POINTILLISM_EFFECT_COLOR_CONVERT_APIS__

#define FAST_COMPUTE_EXTRA_PRECISION

#include "Common.hpp"
#include "Param_Utils.h"
#include "CompileTimeUtils.hpp"
#include "CommonPixFormat.hpp"
#include "CommonAuxPixFormat.hpp"
#include "ColorTransformMatrix.hpp"
#include "ColorTransform.hpp"
#include "FastAriphmetics.hpp"

// AVX2 accelerated API's
void ConvertToCIELab_BGRA_8u
(
    const PF_Pixel_BGRA_8u* RESTRICT pRGB,
    fCIELabPix*             RESTRICT pLab,
    const int32_t           sizeX,
    const int32_t           sizeY,
    const int32_t           rgbPitch,
    const int32_t           labPitch
) noexcept;

void ConvertToCIELab_ARGB_8u
(
    const PF_Pixel_ARGB_8u* RESTRICT pRGB,
    fCIELabPix*             RESTRICT pLab,
    const int32_t           sizeX,
    const int32_t           sizeY,
    const int32_t           rgbPitch,
    const int32_t           labPitch
) noexcept;

void ConvertToCIELab_BGRA_32f
(
    const PF_Pixel_BGRA_32f* RESTRICT pRGB,
    fCIELabPix*             RESTRICT pLab,
    const int32_t sizeX,
    const int32_t sizeY,
    const int32_t rgbPitch,
    const int32_t labPitch
) noexcept;

void ConvertToCIELab_ARGB_32f
(
    const PF_Pixel_ARGB_32f* RESTRICT pRGB,
    fCIELabPix*             RESTRICT pLab,
    int32_t sizeX,
    int32_t sizeY,
    int32_t rgbPitch,
    int32_t labPitch
) noexcept;

void ConvertFromCIELab_BGRA_8u
(
    const fCIELabPix*       RESTRICT pLabSrc,
    PF_Pixel_BGRA_8u*       RESTRICT pBGRADestination,
    const int32_t           sizeX,
    const int32_t           sizeY,
    const int32_t           labPitch,
    const int32_t           rgbPitch
) noexcept;


template <typename T, std::enable_if_t<is_RGB_proc<T>::value>* = nullptr>
inline T ClampPixelValue
(
    const T& input,
    const T& black,
    const T& white
) noexcept
{
    T output;
    output.R = CLAMP_VALUE(input.R, black.R, white.R);
    output.G = CLAMP_VALUE(input.G, black.G, white.G);
    output.B = CLAMP_VALUE(input.B, black.B, white.B);
    output.A = input.A; // not touch Alpha Channel 
    return output;
}


inline fXYZPix Rgb2Xyz
(
    const fRGB& in
) noexcept
{
    auto varValue = [&](const float& in) { return ((in > 0.040450f) ? FastCompute::Pow((in + 0.0550f) / 1.0550f, 2.40f) : (in / 12.92f)); };

    const float var_R = varValue(in.R) * 100.f;
    const float var_G = varValue(in.G) * 100.f;
    const float var_B = varValue(in.B) * 100.f;

    fXYZPix out;
    out.X = var_R * 0.4124564f + var_G * 0.3575761f + var_B * 0.1804375f;
    out.Y = var_R * 0.2126729f + var_G * 0.7151522f + var_B * 0.0721750f;
    out.Z = var_R * 0.0193339f + var_G * 0.1191920f + var_B * 0.9503041f;

    return out;
}


inline fCIELabPix Xyz2CieLab
(
    const fXYZPix& in
) noexcept
{
    constexpr float fRef[3] = {
        cCOLOR_ILLUMINANT[CieLabDefaultObserver][CieLabDefaultIlluminant][0],
        cCOLOR_ILLUMINANT[CieLabDefaultObserver][CieLabDefaultIlluminant][1],
        cCOLOR_ILLUMINANT[CieLabDefaultObserver][CieLabDefaultIlluminant][2],
    };

    auto varValue = [&](const float& in) { return ((in > 0.008856f) ? FastCompute::Cbrt(in) : (in * 7.787f + 16.f / 116.f)); };

    const float var_X = varValue(in.X / fRef[0]);
    const float var_Y = varValue(in.Y / fRef[1]);
    const float var_Z = varValue(in.Z / fRef[2]);

    fCIELabPix out;
    out.L = CLAMP_VALUE(116.f * var_Y - 16.f, -100.f, 100.f);       // L
    out.a = CLAMP_VALUE(500.f * (var_X - var_Y), -128.f, 128.f);    // a
    out.b = CLAMP_VALUE(200.f * (var_Y - var_Z), -128.f, 128.f);    // b

    return out;
}


inline fXYZPix CieLab2Xyz
(
    const fCIELabPix& in
) noexcept
{
    constexpr float fRef[3] = {
        cCOLOR_ILLUMINANT[CieLabDefaultObserver][CieLabDefaultIlluminant][0],
        cCOLOR_ILLUMINANT[CieLabDefaultObserver][CieLabDefaultIlluminant][1],
        cCOLOR_ILLUMINANT[CieLabDefaultObserver][CieLabDefaultIlluminant][2],
    };

    const float var_Y = (in.L + 16.f) / 116.f;
    const float var_X = in.a / 500.f + var_Y;
    const float var_Z = var_Y - in.b / 200.f;

    const float y1 = ((var_Y > 0.2068930f) ? (var_Y * var_Y * var_Y) : ((var_Y - 16.f / 116.f) / 7.787f));
    const float x1 = ((var_X > 0.2068930f) ? (var_X * var_X * var_X) : ((var_X - 16.f / 116.f) / 7.787f));
    const float z1 = ((var_Z > 0.2068930f) ? (var_Z * var_Z * var_Z) : ((var_Z - 16.f / 116.f) / 7.787f));

    fXYZPix out;
    out.X = x1 * fRef[0];
    out.Y = y1 * fRef[1];
    out.Z = z1 * fRef[2];

    return out;
}

inline fRGB Xyz2Rgb
(
    const fXYZPix& in
) noexcept
{
    const float var_X = in.X / 100.f;
    const float var_Y = in.Y / 100.f;
    const float var_Z = in.Z / 100.f;

    const float r1 = var_X *  3.2406f + var_Y * -1.5372f + var_Z * -0.4986f;
    const float g1 = var_X * -0.9689f + var_Y *  1.8758f + var_Z *  0.0415f;
    const float b1 = var_X *  0.0557f + var_Y * -0.2040f + var_Z *  1.0570f;

    auto varValue = [&](const float& in) { return ((in > 0.0031308f) ? (1.055f * FastCompute::Pow(in, 1.0f / 2.40f) - 0.055f) : (in * 12.92f)); };

    fRGB out;
    out.R = CLAMP_VALUE(varValue(r1), f32_value_black, f32_value_white);
    out.G = CLAMP_VALUE(varValue(g1), f32_value_black, f32_value_white);
    out.B = CLAMP_VALUE(varValue(b1), f32_value_black, f32_value_white);

    return out;
}

// Convert from RGB to CIELab interleaved
template <typename T, std::enable_if_t<is_RGB_proc<T>::value>* = nullptr>
inline void ConvertToCIELab
(
    const T*    __restrict pRGB,
    fCIELabPix* __restrict pLab,
    const A_long          sizeX,
    const A_long          sizeY,
    const A_long          rgbPitch,
    const A_long          labPitch
) noexcept
{
    float sRgbCoeff = 1.0f / static_cast<float>(u8_value_white);
    if (std::is_same<T, PF_Pixel_BGRA_16u>::value || std::is_same<T, PF_Pixel_ARGB_16u>::value)
        sRgbCoeff = 1.0f / static_cast<float>(u16_value_white);
    else if (std::is_same<T, PF_Pixel_BGRA_32f>::value || std::is_same<T, PF_Pixel_ARGB_32f>::value)
        sRgbCoeff = 1.0f;

    for (A_long j = 0; j < sizeY; j++)
    {
        const T*    __restrict pRgbLine = pRGB + j * rgbPitch;
        fCIELabPix* __restrict pLabLine = pLab + j * labPitch;

        __VECTORIZATION__
        for (A_long i = 0; i < sizeX; i++)
        {
            fRGB inPix;
            inPix.R = pRgbLine[i].R * sRgbCoeff;
            inPix.G = pRgbLine[i].G * sRgbCoeff;
            inPix.B = pRgbLine[i].B * sRgbCoeff;

            pLabLine[i] = Xyz2CieLab(Rgb2Xyz(inPix));
        }
    }

    return;
}


template <typename T, std::enable_if_t<is_YUV_proc<T>::value>* = nullptr>
inline void ConvertToCIELab
(
    const T*    __restrict pYUV,
    fCIELabPix* __restrict pLab,
    const A_long          sizeX,
    const A_long          sizeY,
    const A_long          rgbPitch,
    const A_long          labPitch
) noexcept
{
    float sYuvCoeff = 1.0f / static_cast<float>(u8_value_white);
    float fUVSub = 128.f;
    if (std::is_same<T, PF_Pixel_VUYA_16u>::value)
        sYuvCoeff = 1.0f / static_cast<float>(u16_value_white), fUVSub = 0.f;
    else if (std::is_same<T, PF_Pixel_VUYA_32f>::value)
        sYuvCoeff = 1.0f, fUVSub = 0.f;

    // color space transfer matrix
    CACHE_ALIGN constexpr float ctm[9] =
    {
        YUV2RGB[BT709][0], YUV2RGB[BT709][1], YUV2RGB[BT709][2],
        YUV2RGB[BT709][3], YUV2RGB[BT709][4], YUV2RGB[BT709][5],
        YUV2RGB[BT709][6], YUV2RGB[BT709][7], YUV2RGB[BT709][8]
    };

    for (A_long j = 0; j < sizeY; j++)
    {
        const T*    __restrict pYuvLine = pYUV + j * rgbPitch;
        fCIELabPix* __restrict pLabLine = pLab + j * labPitch;

        __VECTORIZATION__
        for (A_long i = 0; i < sizeX; i++)
        {
            fYUV inYuvPix;
            inYuvPix.Y =  static_cast<float>(pYuvLine[i].Y) * sYuvCoeff;
            inYuvPix.U = (static_cast<float>(pYuvLine[i].U) - fUVSub) * sYuvCoeff;
            inYuvPix.V = (static_cast<float>(pYuvLine[i].V) - fUVSub) * sYuvCoeff;

            fRGB inPix;
            inPix.R = inYuvPix.Y * ctm[0] + inYuvPix.U * ctm[1] + inYuvPix.V * ctm[2];
            inPix.G = inYuvPix.Y * ctm[3] + inYuvPix.U * ctm[4] + inYuvPix.V * ctm[5];
            inPix.B = inYuvPix.Y * ctm[6] + inYuvPix.U * ctm[7] + inYuvPix.V * ctm[8];

            pLabLine[i] = Xyz2CieLab(Rgb2Xyz(inPix));
        }
    }

    return;
}


inline void ConvertToCIELab
(
    const PF_Pixel_RGB_10u* __restrict pRGB,
    fCIELabPix* __restrict pLab,
    const A_long          sizeX,
    const A_long          sizeY,
    const A_long          rgbPitch,
    const A_long          labPitch
) noexcept
{
    constexpr float sRgbCoeff = 1.0f / static_cast<float>(u10_value_white);

    for (A_long j = 0; j < sizeY; j++)
    {
        const PF_Pixel_RGB_10u* __restrict pRgbLine = pRGB + j * rgbPitch;
        fCIELabPix* __restrict pLabLine = pLab + j * labPitch;

        __VECTORIZATION__
        for (A_long i = 0; i < sizeX; i++)
        {
            fRGB inPix;
            inPix.R = pRgbLine[i].R * sRgbCoeff;
            inPix.G = pRgbLine[i].G * sRgbCoeff;
            inPix.B = pRgbLine[i].B * sRgbCoeff;

            pLabLine[i] = Xyz2CieLab(Rgb2Xyz(inPix));
        }
    }

    return;
}


template <typename T, std::enable_if_t<is_RGB_proc<T>::value>* = nullptr>
inline void ConvertFromCIELab
(
    const T*          __restrict pSrc,  // Source (original) image for get Alpha-channel values
    const fCIELabPix* __restrict pLab,  // Processed buffer in LAB format
          T*          __restrict pDst,  // Destination buffer (rendering target)
    const A_long          sizeX,        // Horizontal framne size
    const A_long          sizeY,        // Verticalframe size
    const A_long          srcPitch,     // Source buffer line pitch
    const A_long          labPitch,     // LAB buffer linepitch
    const A_long          dstPitch      // Destination buffer line pitch 
) noexcept
{
    float sRgbCoeff = static_cast<float>(u8_value_white);
    if (std::is_same<T, PF_Pixel_BGRA_16u>::value || std::is_same<T, PF_Pixel_ARGB_16u>::value)
        sRgbCoeff = static_cast<float>(u16_value_white);
    else if (std::is_same<T, PF_Pixel_BGRA_32f>::value || std::is_same<T, PF_Pixel_ARGB_32f>::value)
        sRgbCoeff = 1.0f;

    for (A_long j = 0; j < sizeY; j++)
    {
        const T*          __restrict pSrcLine = pSrc + j * srcPitch;
        const fCIELabPix* __restrict pLabLine = pLab + j * labPitch;
              T*          __restrict pDstLine = pDst + j * dstPitch;

        __VECTORIZATION__
        for (A_long i = 0; i < sizeX; i++)
        {
            const fRGB rgbPix = Xyz2Rgb(CieLab2Xyz(pLabLine[i]));
            pDstLine[i].A = pSrcLine[i].A;
            pDstLine[i].R = static_cast<decltype(pDstLine[i].R)>(CLAMP_VALUE(rgbPix.R * sRgbCoeff, 0.f, static_cast<float>(sRgbCoeff)));
            pDstLine[i].G = static_cast<decltype(pDstLine[i].G)>(CLAMP_VALUE(rgbPix.G * sRgbCoeff, 0.f, static_cast<float>(sRgbCoeff)));
            pDstLine[i].B = static_cast<decltype(pDstLine[i].B)>(CLAMP_VALUE(rgbPix.B * sRgbCoeff, 0.f, static_cast<float>(sRgbCoeff)));
        }
    }

    return;
}


template <typename T, std::enable_if_t<is_YUV_proc<T>::value>* = nullptr>
inline void ConvertFromCIELab
(
    const T*          __restrict pSrc,  // Source (original) image for get Alpha-channel values
    const fCIELabPix* __restrict pLab,  // Processed buffer in LAB format
          T*          __restrict pDst,  // Destination buffer (rendering target)
    const A_long          sizeX,        // Horizontal framne size
    const A_long          sizeY,        // Verticalframe size
    const A_long          srcPitch,     // Source buffer line pitch
    const A_long          labPitch,     // LAB buffer linepitch
    const A_long          dstPitch      // Destination buffer line pitch 
) noexcept
{
    float sYuvCoeff = static_cast<float>(u8_value_white);
    float fUVAdd = 128.f;
    if (std::is_same<T, PF_Pixel_VUYA_16u>::value)
        sYuvCoeff = static_cast<float>(u16_value_white), fUVAdd = 0.f;
    else if (std::is_same<T, PF_Pixel_VUYA_32f>::value)
        sYuvCoeff = 1.0f, fUVAdd = 0.f;

    // color space transfer matrix
    CACHE_ALIGN constexpr float ctm[9] =
    {
        RGB2YUV[BT709][0], RGB2YUV[BT709][1], RGB2YUV[BT709][2],
        RGB2YUV[BT709][3], RGB2YUV[BT709][4], RGB2YUV[BT709][5],
        RGB2YUV[BT709][6], RGB2YUV[BT709][7], RGB2YUV[BT709][8]
    };

    for (A_long j = 0; j < sizeY; j++)
    {
        const T*          __restrict pSrcLine = pSrc + j * srcPitch;
        const fCIELabPix* __restrict pLabLine = pLab + j * labPitch;
              T*          __restrict pDstLine = pDst + j * dstPitch;

        __VECTORIZATION__
        for (A_long i = 0; i < sizeX; i++)
        {
            const fRGB rgbPix = Xyz2Rgb(CieLab2Xyz(pLabLine[i]));

            const float Y = sYuvCoeff * (rgbPix.R * ctm[0] + rgbPix.G * ctm[1] + rgbPix.B * ctm[2]);
            const float U = sYuvCoeff * (rgbPix.R * ctm[3] + rgbPix.G * ctm[4] + rgbPix.B * ctm[5]) + fUVAdd;
            const float V = sYuvCoeff * (rgbPix.R * ctm[6] + rgbPix.G * ctm[7] + rgbPix.B * ctm[8]) + fUVAdd;

            pDstLine[i].A = pSrcLine[i].A;
            pDstLine[i].Y = static_cast<decltype(pDstLine[i].Y)>(CLAMP_VALUE(Y, 0.f, static_cast<float>(sYuvCoeff)));
            pDstLine[i].U = static_cast<decltype(pDstLine[i].U)>(CLAMP_VALUE(U, 0.f, static_cast<float>(sYuvCoeff)));
            pDstLine[i].V = static_cast<decltype(pDstLine[i].V)>(CLAMP_VALUE(V, 0.f, static_cast<float>(sYuvCoeff)));
        }
    }

    return;
}

inline void ConvertFromCIELab
(
    fCIELabPix*       __restrict pLab,
    PF_Pixel_RGB_10u* __restrict pDst,
    const A_long          sizeX,
    const A_long          sizeY,
    const A_long          labPitch,
    const A_long          dstPitch
) noexcept
{
    constexpr float sRgbCoeff = static_cast<float>(u10_value_white);

    for (A_long j = 0; j < sizeY; j++)
    {
        const fCIELabPix*       __restrict pLabLine = pLab + j * labPitch;
              PF_Pixel_RGB_10u* __restrict pDstLine = pDst + j * dstPitch;

        __VECTORIZATION__
        for (A_long i = 0; i < sizeX; i++)
        {
            const fRGB rgbPix = Xyz2Rgb(CieLab2Xyz(pLabLine[i]));
            pDstLine[i].R = static_cast<decltype(pDstLine[i].R)>(CLAMP_VALUE(rgbPix.R * sRgbCoeff, static_cast<float>(u10_value_black), static_cast<float>(u10_value_white)));
            pDstLine[i].G = static_cast<decltype(pDstLine[i].G)>(CLAMP_VALUE(rgbPix.G * sRgbCoeff, static_cast<float>(u10_value_black), static_cast<float>(u10_value_white)));
            pDstLine[i].B = static_cast<decltype(pDstLine[i].B)>(CLAMP_VALUE(rgbPix.B * sRgbCoeff, static_cast<float>(u10_value_black), static_cast<float>(u10_value_white)));
        }
    }

    return;
}

#if 0
// ============================================================================
// SIMD HELPER FUNCTIONS
// ============================================================================


// ============================================================================
// MAIN CONVERSION KERNEL
// ============================================================================

// Converts BGRA (8u) -> CIELab (32f Interleaved)
// src: Pointer to BGRA data (uint8)
// dst: Pointer to float buffer. Size must be width * height * 3
// num_pixels: width * height
void Convert_BGRA_to_CIELab_AVX2 (const uint8_t* src, float* dst, int num_pixels) noexcept
{
    // Constants for XYZ -> LAB
    const __m256 D65_Xn_Inv = _mm256_set1_ps(1.0f / 0.95047f);
    const __m256 D65_Yn_Inv = _mm256_set1_ps(1.0f / 1.00000f);
    const __m256 D65_Zn_Inv = _mm256_set1_ps(1.0f / 1.08883f);

    const __m256 delta = _mm256_set1_ps(6.0f / 29.0f);
    const __m256 epsilon = _mm256_set1_ps(0.008856f); // (6/29)^3
    const __m256 kappa = _mm256_set1_ps(7.787f);      // (1/3) * (29/6)^2
    const __m256 sixteen_116 = _mm256_set1_ps(16.0f / 116.0f);

    // Process 8 pixels at a time
    int i = 0;
    for (; i <= num_pixels - 8; i += 8) {

        // 1. Load 8 pixels (BGRA BGRA ...) -> 32 bytes
        // We use _mm256_loadu_si256 assuming unaligned input
        __m256i bgra_pixels = _mm256_loadu_si256((const __m256i*)(src + i * 4));

        // 2. De-interleave and Convert to 32-bit Integers
        // We need separate registers for R, G, B indices to use Gather.
        // Input layout in 32-bit chunks: [B G R A]

        // Create Shuffle Mask to extract components to lower 32-bits
        // 0xFF means zero out
        // Indices: B=0, G=1, R=2
        const __m256i mask_B = _mm256_set_epi8(
            -1, -1, -1, 12, -1, -1, -1, 8, -1, -1, -1, 4, -1, -1, -1, 0,
            -1, -1, -1, 12, -1, -1, -1, 8, -1, -1, -1, 4, -1, -1, -1, 0
        ); // Pattern repeats for high lane if loading 256 bits directly
           // Note: _mm256_shuffle_epi8 works within 128-bit lanes.
           // A simple shuffle won't work across lanes directly for simple deinterleaving 
           // without permutes, but Gather takes indices.

           // Let's use a simpler extraction strategy:
           // Extract B, G, R indices directly into 3 separate Int vectors.

           // Expand u8 to u32. 
           // Lane 0 (Pixels 0-3), Lane 1 (Pixels 4-7)
        __m256i p_lo = _mm256_cvtepu8_epi32(_mm_loadu_si128((__m128i*)(src + i * 4)));
        __m256i p_hi = _mm256_cvtepu8_epi32(_mm_loadu_si128((__m128i*)(src + i * 4 + 16)));

        // Now we have standard [B G R A] integers in vectors.
        // We need to shift/mask to isolate indices.
        // Actually, since we already unpacked to 32-bit integers:
        // p_lo contains: B0, G0, R0, A0, B1, G1...
        // This layout is NOT ideal for Gather. Gather wants: R0 R1 R2...

        // Optimized Gather Approach:
        // Load indices: 0, 4, 8... for B
        // Load indices: 1, 5, 9... for G
        // Load indices: 2, 6, 10... for R

        // Setup Gather Indices
        // Offsets in bytes relative to src+i*4
        __m256i idx_B = _mm256_setr_epi32(0, 4, 8, 12, 16, 20, 24, 28);
        __m256i idx_G = _mm256_add_epi32(idx_B, _mm256_set1_epi32(1));
        __m256i idx_R = _mm256_add_epi32(idx_B, _mm256_set1_epi32(2));

        // 3. GATHER & LINEARIZE (sRGB -> Linear via LUT)
        // Uses src pointer as base.
        // WARNING: i32gather uses scale factor.
        // Since input is u8, we gather directly from the LUT using the pixel values as indices.

        // We need the values (0-255) to serve as INDICES into the LUT.
        // Re-using the unpacked integers 'p_lo' and 'p_hi' is messy.
        // Let's gather the uint8 values properly.

        // Actually, we can use _mm256_i32gather_ps on the LUT, using the integer Pixel Values as offsets.
        // To do that, we need the pixel values (0-255) in integer vectors.

        // Get B, G, R integers (0-255) into planar vectors
        // We utilize the shuffle mask idea again, properly.
        // BGRA BGRA ... 
        // We want 8 integers of B in one register.
        // We use bit masking and shuffling.

        // Or/And/Shift is faster than Gather from RAM for extraction usually.
        __m256i v_src = _mm256_loadu_si256((const __m256i*)(src + i * 4));

        // Permute to group by 32-bits to align for pack
        const __m256i K_SHUFF_B = _mm256_set_epi8(-1, -1, -1, 12, -1, -1, -1, 8, -1, -1, -1, 4, -1, -1, -1, 0, -1, -1, -1, 12, -1, -1, -1, 8, -1, -1, -1, 4, -1, -1, -1, 0);
        // This shuffle is lane-specific.
        // A standard SOA-AOS unzipping is complex. 

        // Let's use the Gather-from-Memory approach for the pixels themselves? 
        // No, that's slow. Load full line, then shuffle.

        // Re-approach: Extract using bitmasks on the 32-bit expanded data
        // It's instruction heavy but predictable.
        // B = val & 0xFF
        // G = (val >> 8) & 0xFF
        // R = (val >> 16) & 0xFF

        __m256i b_int = _mm256_and_si256(p_lo, _mm256_set1_epi32(0xFF));
        __m256i g_int = _mm256_and_si256(_mm256_srli_epi32(p_lo, 8), _mm256_set1_epi32(0xFF));
        __m256i r_int = _mm256_and_si256(_mm256_srli_epi32(p_lo, 16), _mm256_set1_epi32(0xFF));

        // BUT p_lo only has 4 pixels. Need p_hi too.
        // Let's combine them into 8-wide vectors.
        // Pack 32-bit integers back? No.

        // We have 8 pixels.
        // Use _mm256_i32gather_epi32? No.
        // Let's just use scalar lookup for the *indices* if shuffle is too hard? NO. PROLEVEL.

        // Correct way to extract 8 bytes from BGRA stream to 8 integers:
        // Use _mm256_shuffle_epi8 with a mask that puts the bytes into the lower positions of 32-bit words
        // Mask for B: [0, X, X, X, 4, X, X, X ...]
        // Note: shuffle_epi8 does NOT cross 128-bit lanes.
        // We loaded 32 bytes (2 lanes). Lane 0 has px 0-3. Lane 1 has px 4-7.
        // Perfect.

        __m256i mask_extract = _mm256_setr_epi8(
            0, -1, -1, -1, 4, -1, -1, -1, 8, -1, -1, -1, 12, -1, -1, -1,
            0, -1, -1, -1, 4, -1, -1, -1, 8, -1, -1, -1, 12, -1, -1, -1
        );

        __m256i v_B = _mm256_shuffle_epi8(bgra_pixels, mask_extract);
        __m256i v_G = _mm256_shuffle_epi8(_mm256_srli_epi16(bgra_pixels, 8), mask_extract); // Shift right 8 bits (G is now at 0)
                                                                                            // For R (offset 2), shift right 16 is tricky on vector.
                                                                                            // Shift global 16 bits? _mm256_srli_epi32(bgra, 16)
        __m256i v_R = _mm256_shuffle_epi8(_mm256_srli_epi32(bgra_pixels, 16), mask_extract);

        // Now v_B, v_G, v_R hold 32-bit integers (0-255). 
        // We can use them as indices for the LUT gather.

        __m256 r_lin = _mm256_i32gather_ps(SRGB_TO_LINEAR_LUT, v_R, 4);
        __m256 g_lin = _mm256_i32gather_ps(SRGB_TO_LINEAR_LUT, v_G, 4);
        __m256 b_lin = _mm256_i32gather_ps(SRGB_TO_LINEAR_LUT, v_B, 4);

        // 4. Linear RGB -> XYZ (Matrix Multiply)
        // X = 0.4124*R + 0.3576*G + 0.1805*B
        __m256 X = _mm256_fmadd_ps(_mm256_set1_ps(0.4124564f), r_lin,
            _mm256_fmadd_ps(_mm256_set1_ps(0.3575761f), g_lin,
                _mm256_mul_ps(_mm256_set1_ps(0.1804375f), b_lin)));

        __m256 Y = _mm256_fmadd_ps(_mm256_set1_ps(0.2126729f), r_lin,
            _mm256_fmadd_ps(_mm256_set1_ps(0.7151522f), g_lin,
                _mm256_mul_ps(_mm256_set1_ps(0.0721750f), b_lin)));

        __m256 Z = _mm256_fmadd_ps(_mm256_set1_ps(0.0193339f), r_lin,
            _mm256_fmadd_ps(_mm256_set1_ps(0.1191920f), g_lin,
                _mm256_mul_ps(_mm256_set1_ps(0.9503041f), b_lin)));

        // 5. XYZ -> LAB (Non-linear transform)
        // Normalize
        __m256 xr = _mm256_mul_ps(X, D65_Xn_Inv);
        __m256 yr = _mm256_mul_ps(Y, D65_Yn_Inv);
        __m256 zr = _mm256_mul_ps(Z, D65_Zn_Inv);

        // Compute f(t)
        // Condition: t > epsilon
        __m256 mask_x = _mm256_cmp_ps(xr, epsilon, _CMP_GT_OQ);
        __m256 mask_y = _mm256_cmp_ps(yr, epsilon, _CMP_GT_OQ);
        __m256 mask_z = _mm256_cmp_ps(zr, epsilon, _CMP_GT_OQ);

        // Branch 1: t^(1/3) (approximated)
        __m256 fx_cbrt = mm256_cbrt_ps_fast(xr);
        __m256 fy_cbrt = mm256_cbrt_ps_fast(yr);
        __m256 fz_cbrt = mm256_cbrt_ps_fast(zr);

        // Branch 2: 7.787*t + 16/116
        __m256 fx_lin = _mm256_fmadd_ps(kappa, xr, sixteen_116);
        __m256 fy_lin = _mm256_fmadd_ps(kappa, yr, sixteen_116);
        __m256 fz_lin = _mm256_fmadd_ps(kappa, zr, sixteen_116);

        // Blend results based on mask
        __m256 fx = _mm256_blendv_ps(fx_lin, fx_cbrt, mask_x);
        __m256 fy = _mm256_blendv_ps(fy_lin, fy_cbrt, mask_y);
        __m256 fz = _mm256_blendv_ps(fz_lin, fz_cbrt, mask_z);

        // Calculate L, a, b
        // L = 116 * fy - 16
        __m256 L = _mm256_fmadd_ps(_mm256_set1_ps(116.0f), fy, _mm256_set1_ps(-16.0f));
        // a = 500 * (fx - fy)
        __m256 a = _mm256_mul_ps(_mm256_set1_ps(500.0f), _mm256_sub_ps(fx, fy));
        // b = 200 * (fy - fz)
        __m256 b = _mm256_mul_ps(_mm256_set1_ps(200.0f), _mm256_sub_ps(fy, fz));

        // 6. Pack Planar (L, a, b) to Interleaved (LabLab...)
        // We have 8 values of L, 8 of a, 8 of b.
        // We need to write 24 floats.
        // Efficient Store Strategy: Write to stack buffer then memcpy or store.
        // Direct intrinsics for 3-way shuffle is complex.
        // Using a temporary align buffer on stack is optimal for L1 cache.

        float temp_buf[24];
        // We can't vector store directly easily.
        // Let's use scalar extract for simplicity and reliability, 
        // or 3 _mm256_storeu_ps calls if we shuffle first.

        // Manual shuffle to temp buffer (Compiler vectorizes this usually)
        _mm256_storeu_ps(temp_buf, L);      // L0..L7
        _mm256_storeu_ps(temp_buf + 8, a);    // a0..a7
        _mm256_storeu_ps(temp_buf + 16, b);   // b0..b7

                                              // Interleave manually (Unrolled loop)
                                              // This is the only scalar part, but it's pure L1 cache movement.
        float* d = dst + i * 3;
        for (int k = 0; k<8; ++k) {
            d[k * 3 + 0] = temp_buf[k];
            d[k * 3 + 1] = temp_buf[k + 8];
            d[k * 3 + 2] = temp_buf[k + 16];
        }
    }

    // Handle Remaining Pixels (Scalar Fallback)
    for (; i < num_pixels; i++) {
        // ... standard scalar implementation for last <8 pixels ...
        // (omitted for brevity, essentially the same math)
    }
}

#endif

#endif // __IMAGE_LAB2_POINTILLISM_EFFECT_COLOR_CONVERT_APIS__