// ============================================================================
//  AwbHostConvert.cpp  --  host BGRA_8u  <->  planar scene-linear RGB_32f
//
//  Bridges the host's interleaved 8-bit BGRA frames to the algorithm's planar
//  float planes (mem.input / mem.output), applying the sRGB EOTF/OETF so the
//  planar data is scene-linear in [0,1] as the PCA core expects.
//
//  Pitches are in PIXELS and signed; planar stride may be padded (padW).
// ============================================================================
#include <cmath>
#include <cstdint>
#include <cstddef>

#include "Common.hpp"
#include "FastAriphmetics.hpp"
#include "AwbHostConvert.hpp"

#ifndef RESTRICT
#define RESTRICT __restrict
#endif

// Set to 0 to bridge 0..255 <-> 0..1 with NO gamma curve (old non-linear path).
#ifndef AWB_HOST_APPLY_GAMMA
#define AWB_HOST_APPLY_GAMMA 1
#endif

namespace
{
    // 256-entry sRGB EOTF LUT (8-bit code -> scene-linear [0,1]). Exact, fast.
    struct SrgbDecodeLUT
    {
        CACHE_ALIGN float v[256];
        SrgbDecodeLUT() noexcept
        {
            for (int i = 0; i < 256; ++i)
            {
                const float s = static_cast<float>(i) * (1.0f / 255.0f);
#if AWB_HOST_APPLY_GAMMA
                v[i] = (s <= 0.04045f) ? (s * (1.0f / 12.92f))
                                       : FastCompute::Pow((s + 0.055f) * (1.0f / 1.055f), 2.4f);
#else
                v[i] = s;
#endif
            }
        }
    };
    const SrgbDecodeLUT gDecode;

    // sRGB OETF (scene-linear [0,1] -> [0,1] encoded).
    inline float linear_to_srgb(float v) noexcept
    {
#if AWB_HOST_APPLY_GAMMA
        if (v <= 0.0031308f) return 12.92f * v;
        return 1.055f * FastCompute::Pow(v, 1.0f / 2.4f) - 0.055f;
#else
        return v;
#endif
    }

    // encoded float -> clamped, rounded 8-bit code
    inline A_u_char to_u8(float lin) noexcept
    {
        if (lin <= 0.0f) lin = 0.0f;          // floor (apply already floored; round-off safe)
        else if (lin >= 1.0f) lin = 1.0f;     // clamp highlights before encode
        const int c = static_cast<int>(linear_to_srgb(lin) * 255.0f + 0.5f);
        return static_cast<A_u_char>(c < 0 ? 0 : (c > 255 ? 255 : c));
    }
}

// ----------------------------------------------------------------------------
void Convert_BGRA8u_to_LinearPlanar
(
    MemHandler&                      mem,
    const PF_Pixel_BGRA_8u* RESTRICT inBGRA,
    const A_long                     sizeX,
    const A_long                     sizeY,
    const A_long                     inPitch,
    const A_long                     planarPitch
) noexcept
{
    if (nullptr == inBGRA || sizeX <= 0 || sizeY <= 0 ||
        nullptr == mem.input.R || nullptr == mem.input.G || nullptr == mem.input.B)
        return;

    float* RESTRICT R = mem.input.R;
    float* RESTRICT G = mem.input.G;
    float* RESTRICT B = mem.input.B;

    for (A_long y = 0; y < sizeY; ++y)
    {
        const PF_Pixel_BGRA_8u* RESTRICT in = inBGRA + static_cast<ptrdiff_t>(y) * inPitch;
        const ptrdiff_t                  o  = static_cast<ptrdiff_t>(y) * planarPitch;
        float* RESTRICT rowR = R + o;
        float* RESTRICT rowG = G + o;
        float* RESTRICT rowB = B + o;

        for (A_long x = 0; x < sizeX; ++x)
        {
            const PF_Pixel_BGRA_8u px = in[x];   // .B .G .R .A  (alpha not stored; carried at egress)
            rowR[x] = gDecode.v[px.R];
            rowG[x] = gDecode.v[px.G];
            rowB[x] = gDecode.v[px.B];
        }
    }
}

// ----------------------------------------------------------------------------
void Convert_LinearPlanar_to_BGRA8u
(
    const MemHandler&                mem,
    const PF_Pixel_BGRA_8u* RESTRICT inBGRA,
    PF_Pixel_BGRA_8u*       RESTRICT outBGRA,
    const A_long                     sizeX,
    const A_long                     sizeY,
    const A_long                     inPitch,
    const A_long                     outPitch,
    const A_long                     planarPitch
) noexcept
{
    if (nullptr == outBGRA || sizeX <= 0 || sizeY <= 0 ||
        nullptr == mem.output.R || nullptr == mem.output.G || nullptr == mem.output.B)
        return;

    const float* RESTRICT R = mem.output.R;
    const float* RESTRICT G = mem.output.G;
    const float* RESTRICT B = mem.output.B;

    for (A_long y = 0; y < sizeY; ++y)
    {
        const ptrdiff_t                  o   = static_cast<ptrdiff_t>(y) * planarPitch;
        const float* RESTRICT            rowR = R + o;
        const float* RESTRICT            rowG = G + o;
        const float* RESTRICT            rowB = B + o;
        const PF_Pixel_BGRA_8u* RESTRICT inR  = inBGRA ? (inBGRA + static_cast<ptrdiff_t>(y) * inPitch) : nullptr;
        PF_Pixel_BGRA_8u*       RESTRICT outR =           outBGRA + static_cast<ptrdiff_t>(y) * outPitch;

        for (A_long x = 0; x < sizeX; ++x)
        {
            PF_Pixel_BGRA_8u px;
            px.B = to_u8(rowB[x]);
            px.G = to_u8(rowG[x]);
            px.R = to_u8(rowR[x]);
            px.A = inR ? inR[x].A : static_cast<A_u_char>(255);   // alpha copied from input
            outR[x] = px;
        }
    }
}
