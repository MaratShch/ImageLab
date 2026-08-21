#ifndef __IMAGELAB2_MEASURE_AVX2_HPP__
#define __IMAGELAB2_MEASURE_AVX2_HPP__

// =============================================================================
// AlgoMeasureAVX2.hpp - FUSED AVX2 measure pass (forward conversion).
//
// ONE streaming pass over the interleaved Adobe frame that produces, together:
//   1. the linear, interleaved working-space RGB float32 buffer (canonical),
//      or - in confidence-map mode - the map (kept pixels as-is, excluded
//      pixels black) in the same buffer;
//   2. the locus-gated SuperPixel (weighted mean, DOUBLE accumulation);
//   3. keptFraction (confidence readout).
//
// ARCHITECTURE (mirrors the DenoiseColor* reference):
//   * PixelTraitsAVX2<Reader>: LoadAVX2() deinterleaves 8 interleaved pixels
//     into planar REGISTERS (vR,vG,vB encoded/normalized, premultiply already
//     resolved) - "planar in registers, interleaved in memory".
//   * compile-time template kernel per format + one runtime dispatch switch;
//   * padded-tail-in-registers: the tail also runs the AVX2 body; zero-padded
//     lanes carry zero energy so the energy gate excludes them from the
//     SuperPixel automatically; only 'remaining' pixels are stored back.
//
// TRANSFER DECODE: polynomial sRGB EOTF (deg-7 log2 / deg-6 exp2, FMA),
// worst abs error 1.36e-6 - finer than the 16-bit LUT quantization of the
// scalar path, and gather-free so the whole pipeline stays 8-wide. The
// scalar path (AlgoPrFormatIngest.hpp) remains the bit-exact reference.
//
// MeasureCtxAVX2 carries ALL surrounding data prebuilt (broadcast matrix,
// thresholds, locus-gate tables as f32) - built ONCE per setup from the
// LocusGate; nothing is rediscovered per pixel.
//
// PRECISION: per-pixel math in float (u,v error ~1e-7, three orders below
// the gate band); SuperPixel accumulation in DOUBLE (mandatory - summing
// millions of floats loses low bits). Validated against the scalar path.
//
// PORTABILITY: strictly <immintrin.h> + <cmath>; no compiler/OS specifics.
// Flags: GCC/Clang -mavx2 -mfma ; MSVC /arch:AVX2. Gate the call site on a
// runtime AVX2 check and fall back to the scalar ingest if unavailable.
// Single-threaded per call (host owns concurrency); composes with the
// host's render threading.
// =============================================================================

#include <immintrin.h>
#include <cstdint>
#include <cmath>
#include "Common.hpp"               // CACHE_ALIGN
#include "AlgoPrFormatIngest.hpp"   // formats, structs, LocusGate, SuperPixel

namespace AlgoPrIngest
{
namespace avx2
{
    // ------------------------------------------------------------------ math
    static inline __m256 fmadd(__m256 a, __m256 b, __m256 c) { return _mm256_fmadd_ps(a, b, c); }

    // 8-wide log2(x), x > 0 : exponent/mantissa split + deg-7 minimax poly.
    static inline __m256 log2_ps(__m256 x)
    {
        const __m256i xi   = _mm256_castps_si256(x);
        const __m256i eI   = _mm256_sub_epi32(_mm256_srli_epi32(
                                _mm256_and_si256(xi, _mm256_set1_epi32(0x7F800000)), 23),
                                _mm256_set1_epi32(127));
        const __m256  ef   = _mm256_cvtepi32_ps(eI);
        const __m256  m    = _mm256_or_ps(_mm256_castsi256_ps(
                                _mm256_and_si256(xi, _mm256_set1_epi32(0x007FFFFF))),
                                _mm256_set1_ps(1.0f));
        const __m256  t    = _mm256_sub_ps(m, _mm256_set1_ps(1.0f));
        __m256 p = _mm256_set1_ps( 1.459860554e-02f);
        p = fmadd(p, t, _mm256_set1_ps(-7.592089396e-02f));
        p = fmadd(p, t, _mm256_set1_ps( 1.886527228e-01f));
        p = fmadd(p, t, _mm256_set1_ps(-3.214835301e-01f));
        p = fmadd(p, t, _mm256_set1_ps( 4.717218708e-01f));
        p = fmadd(p, t, _mm256_set1_ps(-7.202026917e-01f));
        p = fmadd(p, t, _mm256_set1_ps( 1.442633691e+00f));
        p = fmadd(p, t, _mm256_set1_ps( 8.116678600e-07f));
        return _mm256_add_ps(ef, p);
    }

    // 8-wide exp2(y) : integer/fraction split + deg-6 minimax poly.
    static inline __m256 exp2_ps(__m256 y)
    {
        const __m256 fl = _mm256_floor_ps(y);
        const __m256 f  = _mm256_sub_ps(y, fl);
        __m256 p = _mm256_set1_ps(2.187125795e-04f);
        p = fmadd(p, f, _mm256_set1_ps(1.238241248e-03f));
        p = fmadd(p, f, _mm256_set1_ps(9.686187232e-03f));
        p = fmadd(p, f, _mm256_set1_ps(5.547891155e-02f));
        p = fmadd(p, f, _mm256_set1_ps(2.402310971e-01f));
        p = fmadd(p, f, _mm256_set1_ps(6.931468376e-01f));
        p = fmadd(p, f, _mm256_set1_ps(1.000000006e+00f));
        const __m256i ei = _mm256_cvtps_epi32(fl);
        const __m256i pw = _mm256_slli_epi32(_mm256_add_epi32(ei, _mm256_set1_epi32(127)), 23);
        return _mm256_mul_ps(p, _mm256_castsi256_ps(pw));
    }

    // 8-wide sRGB EOTF decode (IEC 61966-2-1), branch-free.
    // Worst abs error 1.36e-6 vs the exact decode (verified, 200k samples).
    static inline __m256 srgb_decode_ps(__m256 c)
    {
        const __m256 lin  = _mm256_mul_ps(c, _mm256_set1_ps(1.0f / 12.92f));
        const __m256 base = _mm256_max_ps(_mm256_mul_ps(
                              _mm256_add_ps(c, _mm256_set1_ps(0.055f)),
                              _mm256_set1_ps(1.0f / 1.055f)),
                              _mm256_set1_ps(1.17549435e-38f));   // no log2(<=0)
        const __m256 pw   = exp2_ps(_mm256_mul_ps(_mm256_set1_ps(2.4f), log2_ps(base)));
        const __m256 mask = _mm256_cmp_ps(c, _mm256_set1_ps(0.04045f), _CMP_LE_OQ);
        return _mm256_blendv_ps(pw, lin, mask);
    }

    // portable popcount of an 8-bit movemask (no POPCNT dependency)
    static inline int popcount8(int m)
    {
        m = m - ((m >> 1) & 0x55);
        m = (m & 0x33) + ((m >> 2) & 0x33);
        return (m + (m >> 4)) & 0x0F;
    }

    // ------------------------------------------------------- measure context
    // Built ONCE per setup (per observer x working space) from the LocusGate.
    // All constants pre-broadcast; gate tables re-emitted as f32 (2 KB, L1).
    struct MeasureCtxAVX2
    {
        // working-space RGB->XYZ rows (broadcast)
        __m256 M0, M1, M2, M3, M4, M5, M6, M7, M8;
        // gates
        __m256 vYDark, vChClip, vTaperLo, vTaperInv;      // luma / clip taper
        __m256 vEnergyMin;
        // locus gate
        __m256 vUMin, vInvStep, vDuvZero, vLocInv;        // taper 1/(zero-full)
        __m256 vInvStepD;                                  // dense-table step
        __m256 vDuvFull;
        __m256 vEu0, vEv0, vEu1, vEv1;                    // endpoints (2D dist)
        float  uMaxScalar;                                 // for index clamp
        // DENSE nearest-entry resampling of the gate tables (u step ~2.6e-4:
        // nearest-entry error << taper band, so no per-lane interpolation is
        // needed -> 2 gathers instead of 4). 8 KB total, L1-resident.
        static const int kNd = 1024;
        CACHE_ALIGN float vTab[kNd];
        CACHE_ALIGN float fTab[kNd];
    };

    inline void build_measure_ctx(const LocusGate& g, MeasureCtxAVX2& c)
    {
        c.M0 = _mm256_set1_ps((float)g.M[0]); c.M1 = _mm256_set1_ps((float)g.M[1]);
        c.M2 = _mm256_set1_ps((float)g.M[2]); c.M3 = _mm256_set1_ps((float)g.M[3]);
        c.M4 = _mm256_set1_ps((float)g.M[4]); c.M5 = _mm256_set1_ps((float)g.M[5]);
        c.M6 = _mm256_set1_ps((float)g.M[6]); c.M7 = _mm256_set1_ps((float)g.M[7]);
        c.M8 = _mm256_set1_ps((float)g.M[8]);
        c.vYDark    = _mm256_set1_ps(0.010f);
        c.vChClip   = _mm256_set1_ps(0.95f);
        c.vTaperLo  = _mm256_set1_ps(0.90f);
        c.vTaperInv = _mm256_set1_ps(1.0f / (0.95f - 0.90f));
        c.vEnergyMin= _mm256_set1_ps(1.0e-6f);
        c.vUMin     = _mm256_set1_ps((float)g.uMin);
        c.vInvStep  = _mm256_set1_ps((float)g.invStep);
        c.vDuvFull  = _mm256_set1_ps((float)g.duvFull);
        c.vDuvZero  = _mm256_set1_ps((float)g.duvZero);
        c.vLocInv   = _mm256_set1_ps((float)(1.0 / (g.duvZero - g.duvFull)));
        c.vEu0 = _mm256_set1_ps((float)g.eu0); c.vEv0 = _mm256_set1_ps((float)g.ev0);
        c.vEu1 = _mm256_set1_ps((float)g.eu1); c.vEv1 = _mm256_set1_ps((float)g.ev1);
        c.uMaxScalar = (float)(g.uMin + (LocusGate::kN - 1) / g.invStep);
        // resample (linear) the gate tables onto the dense grid
        const double uMaxD = g.uMin + (LocusGate::kN - 1) / g.invStep;
        const double stepD = (uMaxD - g.uMin) / (MeasureCtxAVX2::kNd - 1);
        for (int i = 0; i < MeasureCtxAVX2::kNd; ++i) {
            const double u  = g.uMin + i * stepD;
            double p        = (u - g.uMin) * g.invStep;
            int    i0       = (int)p; if (i0 > LocusGate::kN - 2) i0 = LocusGate::kN - 2;
            const double t  = p - i0;
            c.vTab[i] = (float)(g.vTab[i0] + t * (g.vTab[i0+1] - g.vTab[i0]));
            c.fTab[i] = (float)(g.fTab[i0] + t * (g.fTab[i0+1] - g.fTab[i0]));
        }
        c.vInvStepD = _mm256_set1_ps((float)((MeasureCtxAVX2::kNd - 1) /
                                             (uMaxD - g.uMin)));
    }

    // ------------------------------------------------- interleaved LOAD traits
    // Every trait fills vR,vG,vB with ENCODED (or already-linear) values,
    // normalized to [0,1] nominal, premultiply resolved. kLinear tells the
    // kernel whether to run the sRGB decode.

    // channel byte positions inside the 32-bit pixel (little-endian shifts)
    template <typename Pix> struct Shift8;
    template <> struct Shift8<PF_Pixel_BGRA_8u> { enum { B=0, G=8, R=16, A=24 }; };
    template <> struct Shift8<PF_Pixel_ARGB_8u> { enum { A=0, R=8, G=16, B=24 }; };

    // channel u16 index inside the 64-bit pixel
    template <typename Pix> struct Idx16;
    template <> struct Idx16<PF_Pixel_BGRA_16u> { enum { B=0, G=1, R=2, A=3 }; };
    template <> struct Idx16<PF_Pixel_ARGB_16u> { enum { A=0, R=1, G=2, B=3 }; };

    // compact the 4 u64-lane values of two regs into one 8x32 vector
    static inline __m256i pack_u64lanes(__m256i lo4, __m256i hi4)
    {
        const __m256i idx = _mm256_setr_epi32(0,2,4,6,0,2,4,6);
        const __m256i a = _mm256_permutevar8x32_epi32(lo4, idx); // 4 vals in low128
        const __m256i b = _mm256_permutevar8x32_epi32(hi4, idx);
        return _mm256_permute2x128_si256(a, b, 0x20);
    }

    template <typename Pix, bool Premul>
    struct LoadInt8
    {
        static const std::size_t kPixelBytes = sizeof(Pix);
        static const bool kLinear = false;
        static inline void load(const std::uint8_t* row, int32_t x,
                                __m256& vR, __m256& vG, __m256& vB)
        {
            const __m256i px = _mm256_loadu_si256(
                reinterpret_cast<const __m256i*>(row + (std::size_t)x * kPixelBytes));
            const __m256i m8 = _mm256_set1_epi32(0xFF);
            const __m256i r = _mm256_and_si256(_mm256_srli_epi32(px, Shift8<Pix>::R), m8);
            const __m256i g = _mm256_and_si256(_mm256_srli_epi32(px, Shift8<Pix>::G), m8);
            const __m256i b = _mm256_and_si256(_mm256_srli_epi32(px, Shift8<Pix>::B), m8);
            const __m256 s = _mm256_set1_ps(1.0f / 255.0f);
            vR = _mm256_mul_ps(_mm256_cvtepi32_ps(r), s);
            vG = _mm256_mul_ps(_mm256_cvtepi32_ps(g), s);
            vB = _mm256_mul_ps(_mm256_cvtepi32_ps(b), s);
            if (Premul) {
                const __m256i a = _mm256_and_si256(_mm256_srli_epi32(px, Shift8<Pix>::A), m8);
                const __m256 vA = _mm256_mul_ps(_mm256_cvtepi32_ps(a), s);
                const __m256 nz = _mm256_cmp_ps(vA, _mm256_setzero_ps(), _CMP_GT_OQ);
                const __m256 d  = _mm256_blendv_ps(_mm256_set1_ps(1.0f), vA, nz);
                const __m256 one= _mm256_set1_ps(1.0f);
                vR = _mm256_min_ps(_mm256_div_ps(vR, d), one);   // scalar clamps codes
                vG = _mm256_min_ps(_mm256_div_ps(vG, d), one);   // at white; mirror it
                vB = _mm256_min_ps(_mm256_div_ps(vB, d), one);
            }
        }
    };

    template <typename Pix, bool Premul>
    struct LoadInt16
    {
        static const std::size_t kPixelBytes = sizeof(Pix);
        static const bool kLinear = false;
        static inline __m256i channel(__m256i lo, __m256i hi, int idx)
        {
            const __m256i m16 = _mm256_set1_epi64x(0xFFFF);
            // idx*16 is 0/16/32/48 - a compile-time-known small set; srli by
            // variable imm is fine here because idx is a template-driven const
            __m256i a, b;
            switch (idx) {
                case 0:  a = _mm256_and_si256(lo, m16);
                         b = _mm256_and_si256(hi, m16); break;
                case 1:  a = _mm256_and_si256(_mm256_srli_epi64(lo, 16), m16);
                         b = _mm256_and_si256(_mm256_srli_epi64(hi, 16), m16); break;
                case 2:  a = _mm256_and_si256(_mm256_srli_epi64(lo, 32), m16);
                         b = _mm256_and_si256(_mm256_srli_epi64(hi, 32), m16); break;
                default: a = _mm256_srli_epi64(lo, 48);
                         b = _mm256_srli_epi64(hi, 48); break;
            }
            return pack_u64lanes(a, b);
        }
        static inline void load(const std::uint8_t* row, int32_t x,
                                __m256& vR, __m256& vG, __m256& vB)
        {
            const std::uint8_t* p = row + (std::size_t)x * kPixelBytes;
            const __m256i lo = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(p));
            const __m256i hi = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(p + 32));
            const __m256 s = _mm256_set1_ps(1.0f / 32767.0f);
            vR = _mm256_mul_ps(_mm256_cvtepi32_ps(channel(lo, hi, Idx16<Pix>::R)), s);
            vG = _mm256_mul_ps(_mm256_cvtepi32_ps(channel(lo, hi, Idx16<Pix>::G)), s);
            vB = _mm256_mul_ps(_mm256_cvtepi32_ps(channel(lo, hi, Idx16<Pix>::B)), s);
            if (Premul) {
                const __m256 vA = _mm256_mul_ps(
                    _mm256_cvtepi32_ps(channel(lo, hi, Idx16<Pix>::A)), s);
                const __m256 nz = _mm256_cmp_ps(vA, _mm256_setzero_ps(), _CMP_GT_OQ);
                const __m256 d  = _mm256_blendv_ps(_mm256_set1_ps(1.0f), vA, nz);
                const __m256 one= _mm256_set1_ps(1.0f);
                vR = _mm256_min_ps(_mm256_div_ps(vR, d), one);
                vG = _mm256_min_ps(_mm256_div_ps(vG, d), one);
                vB = _mm256_min_ps(_mm256_div_ps(vB, d), one);
            }
        }
    };

    // 8 x (4-float pixels) -> four planar channel vectors. Simple aligned
    // temp-array deinterleave: measured net 4.2-4.3x for the full pipeline;
    // a shuffle-based transpose is a later micro-optimization if profiling
    // ever shows this as the bottleneck.
    static inline void transpose8x4(const float* p, __m256& c0, __m256& c1,
                                    __m256& c2, __m256& c3)
    {
        CACHE_ALIGN float a[8], b[8], cc[8], dd[8];
        for (int k = 0; k < 8; ++k) {
            a [k] = p[k*4 + 0]; b [k] = p[k*4 + 1];
            cc[k] = p[k*4 + 2]; dd[k] = p[k*4 + 3];
        }
        c0 = _mm256_load_ps(a);  c1 = _mm256_load_ps(b);
        c2 = _mm256_load_ps(cc); c3 = _mm256_load_ps(dd);
    }

    // channel order of the two 32f layouts (memory order of the 4 floats)
    template <typename Pix> struct Shift32IsBGRA;
    template <> struct Shift32IsBGRA<PF_Pixel_BGRA_32f> { static const bool value = true;  };
    template <> struct Shift32IsBGRA<PF_Pixel_ARGB_32f> { static const bool value = false; };

    template <typename Pix, bool Premul, bool Linear>
    struct LoadF32
    {
        static const std::size_t kPixelBytes = sizeof(Pix);
        static const bool kLinear = Linear;
        static inline void load(const std::uint8_t* row, int32_t x,
                                __m256& vR, __m256& vG, __m256& vB)
        {
            const float* p = reinterpret_cast<const float*>(row + (std::size_t)x * kPixelBytes);
            __m256 c0, c1, c2, c3;
            transpose8x4(p, c0, c1, c2, c3);
            __m256 vA;
            // map struct channel order (memory order of the 4 floats)
            if (Shift32IsBGRA<Pix>::value) { vB = c0; vG = c1; vR = c2; vA = c3; }
            else                           { vA = c0; vR = c1; vG = c2; vB = c3; }
            if (Premul) {
                const __m256 nz = _mm256_cmp_ps(vA, _mm256_setzero_ps(), _CMP_GT_OQ);
                const __m256 d  = _mm256_blendv_ps(_mm256_set1_ps(1.0f), vA, nz);
                vR = _mm256_div_ps(vR, d);
                vG = _mm256_div_ps(vG, d);
                vB = _mm256_div_ps(vB, d);
            }
        }
    };

    // exact YCbCr -> R'G'B' (coefficients identical to the scalar kRec601/709)
    template <bool k709>
    inline void ycbcr_to_rgb(__m256 Y, __m256 Cb, __m256 Cr,
                             __m256& R, __m256& G, __m256& B);
    inline void unpremul_encoded(__m256& R, __m256& G, __m256& B, __m256 A);

    // ---- VUYA 8u (studio) / 32f (full range, chroma at +0.5) ----
    template <bool Premul, bool k709>
    struct LoadVUYA8
    {
        static const std::size_t kPixelBytes = sizeof(PF_Pixel_VUYA_8u);
        static const bool kLinear = false;
        static inline void load(const std::uint8_t* row, int32_t x,
                                __m256& vR, __m256& vG, __m256& vB)
        {
            const __m256i px = _mm256_loadu_si256(
                reinterpret_cast<const __m256i*>(row + (std::size_t)x * kPixelBytes));
            const __m256i m8 = _mm256_set1_epi32(0xFF);
            // memory order V,U,Y,A -> shifts 0,8,16,24
            const __m256 V = _mm256_cvtepi32_ps(_mm256_and_si256(px, m8));
            const __m256 U = _mm256_cvtepi32_ps(_mm256_and_si256(_mm256_srli_epi32(px,  8), m8));
            const __m256 Y = _mm256_cvtepi32_ps(_mm256_and_si256(_mm256_srli_epi32(px, 16), m8));
            const __m256 A = _mm256_cvtepi32_ps(_mm256_and_si256(_mm256_srli_epi32(px, 24), m8));
            // studio expand
            const __m256 Yp = _mm256_mul_ps(_mm256_sub_ps(Y, _mm256_set1_ps(16.0f)),
                                            _mm256_set1_ps(1.0f / 219.0f));
            const __m256 Cb = _mm256_mul_ps(_mm256_sub_ps(U, _mm256_set1_ps(128.0f)),
                                            _mm256_set1_ps(1.0f / 224.0f));
            const __m256 Cr = _mm256_mul_ps(_mm256_sub_ps(V, _mm256_set1_ps(128.0f)),
                                            _mm256_set1_ps(1.0f / 224.0f));
            ycbcr_to_rgb<k709>(Yp, Cb, Cr, vR, vG, vB);
            if (Premul) unpremul_encoded(vR, vG, vB,
                             _mm256_mul_ps(A, _mm256_set1_ps(1.0f / 255.0f)));
        }
    };

    template <bool Premul, bool k709>
    struct LoadVUYA32
    {
        static const std::size_t kPixelBytes = sizeof(PF_Pixel_VUYA_32f);
        static const bool kLinear = false;
        static inline void load(const std::uint8_t* row, int32_t x,
                                __m256& vR, __m256& vG, __m256& vB)
        {
            const float* p = reinterpret_cast<const float*>(row + (std::size_t)x * kPixelBytes);
            __m256 V, U, Y, A;
            transpose8x4(p, V, U, Y, A);        // memory order V,U,Y,A
            const __m256 half = _mm256_set1_ps(0.5f);
            const __m256 Cb = _mm256_sub_ps(U, half);
            const __m256 Cr = _mm256_sub_ps(V, half);
            ycbcr_to_rgb<k709>(Y, Cb, Cr, vR, vG, vB);
            if (Premul) unpremul_encoded(vR, vG, vB, A);
        }
    };

    // exact YCbCr -> R'G'B' (coefficients identical to the scalar kRec601/709)
    template <bool k709>
    inline void ycbcr_to_rgb(__m256 Y, __m256 Cb, __m256 Cr,
                             __m256& R, __m256& G, __m256& B)
    {
        const float aR  = k709 ? 1.5748f          : 1.402f;
        const float aB  = k709 ? 1.8556f          : 1.772f;
        const float gCr = k709 ? 0.46812427293064877f : 0.71413628620102210f;
        const float gCb = k709 ? 0.18732427293064878f : 0.34413628620102216f;
        R = fmadd(_mm256_set1_ps(aR), Cr, Y);
        B = fmadd(_mm256_set1_ps(aB), Cb, Y);
        G = _mm256_sub_ps(Y, fmadd(_mm256_set1_ps(gCr), Cr,
                                   _mm256_mul_ps(_mm256_set1_ps(gCb), Cb)));
    }

    inline void unpremul_encoded(__m256& R, __m256& G, __m256& B, __m256 A)
    {
        const __m256 nz = _mm256_cmp_ps(A, _mm256_setzero_ps(), _CMP_GT_OQ);
        const __m256 d  = _mm256_blendv_ps(_mm256_set1_ps(1.0f), A, nz);
        R = _mm256_div_ps(R, d); G = _mm256_div_ps(G, d); B = _mm256_div_ps(B, d);
    }

    struct LoadRGB10
    {
        static const std::size_t kPixelBytes = sizeof(PF_Pixel_RGB_10u);
        static const bool kLinear = false;
        static inline void load(const std::uint8_t* row, int32_t x,
                                __m256& vR, __m256& vG, __m256& vB)
        {
            const __m256i px = _mm256_loadu_si256(
                reinterpret_cast<const __m256i*>(row + (std::size_t)x * kPixelBytes));
            const __m256i m10 = _mm256_set1_epi32(0x3FF);
            const __m256i b = _mm256_and_si256(_mm256_srli_epi32(px,  2), m10);
            const __m256i g = _mm256_and_si256(_mm256_srli_epi32(px, 12), m10);
            const __m256i r = _mm256_srli_epi32(px, 22);
            const __m256 s = _mm256_set1_ps(1.0f / 1023.0f);
            vR = _mm256_mul_ps(_mm256_cvtepi32_ps(r), s);
            vG = _mm256_mul_ps(_mm256_cvtepi32_ps(g), s);
            vB = _mm256_mul_ps(_mm256_cvtepi32_ps(b), s);
        }
    };

    // SoA (R,G,B x8) -> interleaved rgbrgb... : fully vectorized
    // (3 permutes + 2 blends per output vector; 3 unaligned stores).
    static inline void store_rgb8(float* dst, __m256 R, __m256 G, __m256 B)
    {
        const __m256i idx0 = _mm256_setr_epi32(0,0,0,1,1,1,2,2);
        const __m256i idx1 = _mm256_setr_epi32(2,3,3,3,4,4,4,5);
        const __m256i idx2 = _mm256_setr_epi32(5,5,6,6,6,7,7,7);
        const __m256 r0 = _mm256_permutevar8x32_ps(R, idx0);
        const __m256 g0 = _mm256_permutevar8x32_ps(G, idx0);
        const __m256 b0 = _mm256_permutevar8x32_ps(B, idx0);
        const __m256 r1 = _mm256_permutevar8x32_ps(R, idx1);
        const __m256 g1 = _mm256_permutevar8x32_ps(G, idx1);
        const __m256 b1 = _mm256_permutevar8x32_ps(B, idx1);
        const __m256 r2 = _mm256_permutevar8x32_ps(R, idx2);
        const __m256 g2 = _mm256_permutevar8x32_ps(G, idx2);
        const __m256 b2 = _mm256_permutevar8x32_ps(B, idx2);
        // lane picks: o0=[r g b r g b r g] o1=[b r g b r g b r] o2=[g b r g b r g b]
        __m256 o0 = _mm256_blend_ps(r0, g0, 0x92);           // g at 1,4,7
        o0        = _mm256_blend_ps(o0, b0, 0x24);           // b at 2,5
        __m256 o1 = _mm256_blend_ps(b1, r1, 0x92);           // r at 1,4,7
        o1        = _mm256_blend_ps(o1, g1, 0x24);           // g at 2,5
        __m256 o2 = _mm256_blend_ps(g2, b2, 0x92);           // b at 1,4,7
        o2        = _mm256_blend_ps(o2, r2, 0x24);           // r at 2,5
        _mm256_storeu_ps(dst +  0, o0);
        _mm256_storeu_ps(dst +  8, o1);
        _mm256_storeu_ps(dst + 16, o2);
    }

    // -------------------------------------------------------------- the kernel
    // Running frame state: 4 double accumulators x 2 halves + kept counter.
    struct MeasureAccum
    {
        static const int kFlushEvery = 32;                 // <=256 px per strip
        __m256  fR, fG, fB, fW;                            // strip (float)
        __m256d rLo, rHi, gLo, gHi, bLo, bHi, wLo, wHi;    // frame (double)
        std::int64_t kept;
        inline void resetStrip() { fR=fG=fB=fW=_mm256_setzero_ps(); }
        inline void reset() {
            rLo=rHi=gLo=gHi=bLo=bHi=wLo=wHi=_mm256_setzero_pd(); kept = 0;
            resetStrip();
        }
        inline void flush() {
            rLo = _mm256_add_pd(rLo, _mm256_cvtps_pd(_mm256_castps256_ps128(fR)));
            rHi = _mm256_add_pd(rHi, _mm256_cvtps_pd(_mm256_extractf128_ps(fR, 1)));
            gLo = _mm256_add_pd(gLo, _mm256_cvtps_pd(_mm256_castps256_ps128(fG)));
            gHi = _mm256_add_pd(gHi, _mm256_cvtps_pd(_mm256_extractf128_ps(fG, 1)));
            bLo = _mm256_add_pd(bLo, _mm256_cvtps_pd(_mm256_castps256_ps128(fB)));
            bHi = _mm256_add_pd(bHi, _mm256_cvtps_pd(_mm256_extractf128_ps(fB, 1)));
            wLo = _mm256_add_pd(wLo, _mm256_cvtps_pd(_mm256_castps256_ps128(fW)));
            wHi = _mm256_add_pd(wHi, _mm256_cvtps_pd(_mm256_extractf128_ps(fW, 1)));
            resetStrip();
        }
    };

    // Process 8 pixels: decode -> gates -> (store linear / confidence) ->
    // accumulate. 'laneMask' disables tail padding lanes for the store;
    // padded lanes carry zeros and are excluded by the energy gate anyway.
    template <typename Trait, bool kConfidenceMap>
    static inline void body8(const MeasureCtxAVX2& c, __m256 vR, __m256 vG, __m256 vB,
                             float* dst, int lanes, MeasureAccum& acc)
    {
        // 1. transfer decode (skipped for _Linear inputs). Non-linear inputs
        // are CLAMPED to the encoded [0,1] domain first - this mirrors the
        // scalar path exactly (its LUT index clamp), and protects the
        // un-premultiplied / out-of-gamut YCbCr reconstructions from being
        // decoded outside the transfer's domain (e.g. VUYP with tiny alpha).
        __m256 R = vR, G = vG, B = vB;
        if (!Trait::kLinear) {
            const __m256 one = _mm256_set1_ps(1.0f);
            const __m256 zed = _mm256_setzero_ps();
            R = srgb_decode_ps(_mm256_min_ps(_mm256_max_ps(vR, zed), one));
            G = srgb_decode_ps(_mm256_min_ps(_mm256_max_ps(vG, zed), one));
            B = srgb_decode_ps(_mm256_min_ps(_mm256_max_ps(vB, zed), one));
        }

        // 2. gates (float, masks)
        const __m256 zero = _mm256_setzero_ps();
        __m256 keep = _mm256_and_ps(
            _mm256_and_ps(_mm256_cmp_ps(R, zero, _CMP_GE_OQ),
                          _mm256_cmp_ps(G, zero, _CMP_GE_OQ)),
            _mm256_cmp_ps(B, zero, _CMP_GE_OQ));          // negatives + NaN out
        const __m256 energy = _mm256_add_ps(R, _mm256_add_ps(G, B));
        keep = _mm256_and_ps(keep, _mm256_cmp_ps(energy, c.vEnergyMin, _CMP_GT_OQ));
        const __m256 maxc = _mm256_max_ps(R, _mm256_max_ps(G, B));
        keep = _mm256_and_ps(keep, _mm256_cmp_ps(maxc, c.vChClip, _CMP_LT_OQ));

        // exact working-space XYZ
        const __m256 X = fmadd(c.M0, R, fmadd(c.M1, G, _mm256_mul_ps(c.M2, B)));
        const __m256 Y = fmadd(c.M3, R, fmadd(c.M4, G, _mm256_mul_ps(c.M5, B)));
        const __m256 Z = fmadd(c.M6, R, fmadd(c.M7, G, _mm256_mul_ps(c.M8, B)));
        keep = _mm256_and_ps(keep, _mm256_cmp_ps(Y, c.vYDark, _CMP_GE_OQ));

        const __m256 den = fmadd(_mm256_set1_ps(15.0f), Y,
                            fmadd(_mm256_set1_ps(3.0f), Z, X));
        keep = _mm256_and_ps(keep, _mm256_cmp_ps(den, zero, _CMP_GT_OQ));
        const __m256 dSafe = _mm256_blendv_ps(_mm256_set1_ps(1.0f), den,
                            _mm256_cmp_ps(den, zero, _CMP_GT_OQ));
        const __m256 inv = _mm256_div_ps(_mm256_set1_ps(1.0f), dSafe);   // 1 div
        const __m256 u = _mm256_mul_ps(_mm256_mul_ps(_mm256_set1_ps(4.0f), X), inv);
        const __m256 v = _mm256_mul_ps(_mm256_mul_ps(_mm256_set1_ps(6.0f), Y), inv);

        // 3. locus distance: interpolated u-grid lookup (2 KB L1 tables),
        //    2D endpoint distance outside the u range - mirrors the scalar gate
        const __m256 uMax = _mm256_set1_ps(c.uMaxScalar);
        const __m256 uc   = _mm256_min_ps(_mm256_max_ps(u, c.vUMin), uMax);
        // dense nearest-entry lookup: 2 gathers, no per-lane interpolation
        const __m256 pIdx = _mm256_mul_ps(_mm256_sub_ps(uc, c.vUMin), c.vInvStepD);
        const __m256i iN  = _mm256_cvtps_epi32(pIdx);                // round-nearest
        const __m256 vl = _mm256_i32gather_ps(c.vTab, iN, 4);
        const __m256 fl = _mm256_i32gather_ps(c.fTab, iN, 4);
        const __m256 absMask = _mm256_castsi256_ps(_mm256_set1_epi32(0x7FFFFFFF));
        const __m256 dTab = _mm256_mul_ps(_mm256_and_ps(_mm256_sub_ps(v, vl), absMask), fl);
        // endpoint override via SQUARED distances -> ONE sqrt total
        const __m256 du0 = _mm256_sub_ps(u, c.vEu0), dv0 = _mm256_sub_ps(v, c.vEv0);
        const __m256 du1 = _mm256_sub_ps(u, c.vEu1), dv1 = _mm256_sub_ps(v, c.vEv1);
        const __m256 d0sq = fmadd(du0, du0, _mm256_mul_ps(dv0, dv0));
        const __m256 d1sq = fmadd(du1, du1, _mm256_mul_ps(dv1, dv1));
        __m256 dsq = _mm256_mul_ps(dTab, dTab);
        dsq = _mm256_blendv_ps(dsq, d0sq, _mm256_cmp_ps(u, c.vUMin, _CMP_LE_OQ));
        dsq = _mm256_blendv_ps(dsq, d1sq, _mm256_cmp_ps(u, uMax,    _CMP_GE_OQ));
        const __m256 dLoc = _mm256_sqrt_ps(dsq);
        keep = _mm256_and_ps(keep, _mm256_cmp_ps(dLoc, c.vDuvZero, _CMP_LT_OQ));

        // 4. weights: highlight taper x locus taper
        __m256 wLum = _mm256_mul_ps(_mm256_sub_ps(c.vChClip, maxc), c.vTaperInv);
        wLum = _mm256_min_ps(wLum, _mm256_set1_ps(1.0f));
        __m256 wLoc = _mm256_mul_ps(_mm256_sub_ps(c.vDuvZero, dLoc), c.vLocInv);
        wLoc = _mm256_min_ps(wLoc, _mm256_set1_ps(1.0f));
        __m256 w = _mm256_mul_ps(wLum, wLoc);
        w = _mm256_and_ps(w, keep);                            // excluded -> 0
        keep = _mm256_and_ps(keep, _mm256_cmp_ps(w, zero, _CMP_GT_OQ));
        w = _mm256_and_ps(w, keep);

        // 5. store: canonical linear buffer, or confidence map (kept:black)
        __m256 oR = R, oG = G, oB = B;
        if (kConfidenceMap) {
            oR = _mm256_and_ps(R, keep);
            oG = _mm256_and_ps(G, keep);
            oB = _mm256_and_ps(B, keep);
        }
        if (lanes == 8) {
            store_rgb8(dst, oR, oG, oB);                       // shuffle-based
        } else {
            CACHE_ALIGN float sr[8], sg[8], sb[8];
            _mm256_store_ps(sr, oR); _mm256_store_ps(sg, oG); _mm256_store_ps(sb, oB);
            for (int k = 0; k < lanes; ++k) {
                dst[k*3 + 0] = sr[k]; dst[k*3 + 1] = sg[k]; dst[k*3 + 2] = sb[k];
            }
        }

        // 6. accumulate + kept count. Strategy: float accumulation WITHIN a
        // short strip (flushed to the double accumulators every
        // kFlushEvery iterations by the caller - magnitudes stay <= a few
        // hundred, so float is exact enough there), DOUBLE across the frame
        // (the millions-of-pixels part, where float would lose low bits).
        acc.fR = fmadd(w, R, acc.fR);
        acc.fG = fmadd(w, G, acc.fG);
        acc.fB = fmadd(w, B, acc.fB);
        acc.fW = _mm256_add_ps(acc.fW, w);
        acc.kept += popcount8(_mm256_movemask_ps(keep));
    }

    // ------------------------------------------------------------- frame loop
    template <typename Trait, bool kConfidenceMap>
    static void run_measure(const std::uint8_t* base, int32_t sizeX, int32_t sizeY,
                            int32_t srcPitchPx, const MeasureCtxAVX2& ctx,
                            float* dstRGB_f32, SuperPixel<double>& super,
                            double* keptFraction)
    {
        const std::ptrdiff_t stride =
            (std::ptrdiff_t)srcPitchPx * (std::ptrdiff_t)Trait::kPixelBytes;
        MeasureAccum acc; acc.reset();

        for (int32_t y = 0; y < sizeY; ++y)
        {
            const std::uint8_t* row = base + (std::ptrdiff_t)y * stride;
            float* dst = dstRGB_f32 + (std::ptrdiff_t)y * sizeX * 3;
            int32_t x = 0, sinceFlush = 0;
            for (; x + 8 <= sizeX; x += 8) {
                __m256 vR, vG, vB;
                Trait::load(row, x, vR, vG, vB);
                body8<Trait, kConfidenceMap>(ctx, vR, vG, vB, dst + (std::size_t)x*3, 8, acc);
                if (++sinceFlush == MeasureAccum::kFlushEvery) { acc.flush(); sinceFlush = 0; }
            }
            const int32_t remaining = sizeX - x;
            if (remaining > 0) {                       // padded tail, AVX2 body
                CACHE_ALIGN std::uint8_t tail[8 * 16] = {};   // largest pixel 16 B
                const std::uint8_t* src = row + (std::size_t)x * Trait::kPixelBytes;
                for (int32_t i = 0; i < remaining * (int32_t)Trait::kPixelBytes; ++i)
                    tail[i] = src[i];
                __m256 vR, vG, vB;
                Trait::load(tail, 0, vR, vG, vB);
                // zero-padded lanes fail the energy gate -> no superpixel harm;
                // only 'remaining' pixels are written back.
                body8<Trait, kConfidenceMap>(ctx, vR, vG, vB, dst + (std::size_t)x*3,
                                             remaining, acc);
            }
            acc.flush();                                   // row end
        }

        // horizontal reduction (once per frame)
        CACHE_ALIGN double t4[4];
        auto hsum = [&](const __m256d a, const __m256d b) {
            _mm256_store_pd(t4, _mm256_add_pd(a, b));
            return t4[0] + t4[1] + t4[2] + t4[3];
        };
        const double rS = hsum(acc.rLo, acc.rHi);
        const double gS = hsum(acc.gLo, acc.gHi);
        const double bS = hsum(acc.bLo, acc.bHi);
        const double wS = hsum(acc.wLo, acc.wHi);
        if (wS > 0.0) { super.r = rS / wS; super.g = gS / wS; super.b = bS / wS; }
        else          { super.r = super.g = super.b = 0.0; }
        if (keptFraction) {
            const double total = (double)sizeX * (double)sizeY;
            *keptFraction = (total > 0.0) ? (double)acc.kept / total : 0.0;
        }
    }

    // ------------------------------------------------------- public dispatch
    // Same contract as the scalar ingest_and_superpixel, AVX2-fused. dst gets
    // the linear canonical buffer (or the confidence map). Build the ctx once
    // per setup with build_measure_ctx(gate, ctx).
    inline void measure_avx2(const void* src, int32_t sizeX, int32_t sizeY,
                             int32_t srcPitchPx, ePrPixelFormat fmt,
                             const MeasureCtxAVX2& ctx,
                             float* dstRGB_f32, SuperPixel<double>& super,
                             bool confidenceMap = false,
                             double* keptFraction = nullptr)
    {
        const std::uint8_t* base = static_cast<const std::uint8_t*>(src);
        #define IL2_RUN(...)                                                       \
            do { if (confidenceMap)                                                \
                     run_measure<__VA_ARGS__, true >(base, sizeX, sizeY, srcPitchPx, \
                                               ctx, dstRGB_f32, super, keptFraction);\
                 else                                                              \
                     run_measure<__VA_ARGS__, false>(base, sizeX, sizeY, srcPitchPx, \
                                               ctx, dstRGB_f32, super, keptFraction);\
            } while (0)

        switch (fmt)
        {
            case fmt_BGRA_4444_8u:
            case fmt_BGRX_4444_8u:  IL2_RUN(LoadInt8 <PF_Pixel_BGRA_8u , false>); break;
            case fmt_BGRP_4444_8u:  IL2_RUN(LoadInt8 <PF_Pixel_BGRA_8u , true >); break;
            case fmt_ARGB_4444_8u:
            case fmt_XRGB_4444_8u:  IL2_RUN(LoadInt8 <PF_Pixel_ARGB_8u , false>); break;
            case fmt_PRGB_4444_8u:  IL2_RUN(LoadInt8 <PF_Pixel_ARGB_8u , true >); break;

            case fmt_BGRA_4444_16u:
            case fmt_BGRX_4444_16u: IL2_RUN(LoadInt16<PF_Pixel_BGRA_16u, false>); break;
            case fmt_BGRP_4444_16u: IL2_RUN(LoadInt16<PF_Pixel_BGRA_16u, true >); break;
            case fmt_ARGB_4444_16u:
            case fmt_XRGB_4444_16u: IL2_RUN(LoadInt16<PF_Pixel_ARGB_16u, false>); break;
            case fmt_PRGB_4444_16u: IL2_RUN(LoadInt16<PF_Pixel_ARGB_16u, true >); break;

            case fmt_BGRA_4444_32f:
            case fmt_BGRX_4444_32f:        IL2_RUN(LoadF32<PF_Pixel_BGRA_32f, false, false>); break;
            case fmt_BGRP_4444_32f:        IL2_RUN(LoadF32<PF_Pixel_BGRA_32f, true , false>); break;
            case fmt_BGRA_4444_32f_Linear:
            case fmt_BGRX_4444_32f_Linear: IL2_RUN(LoadF32<PF_Pixel_BGRA_32f, false, true >); break;
            case fmt_BGRP_4444_32f_Linear: IL2_RUN(LoadF32<PF_Pixel_BGRA_32f, true , true >); break;
            case fmt_ARGB_4444_32f:
            case fmt_XRGB_4444_32f:        IL2_RUN(LoadF32<PF_Pixel_ARGB_32f, false, false>); break;
            case fmt_ARGB_4444_32f_Linear:
            case fmt_XRGB_4444_32f_Linear: IL2_RUN(LoadF32<PF_Pixel_ARGB_32f, false, true >); break;
            case fmt_PRGB_4444_32f:        IL2_RUN(LoadF32<PF_Pixel_ARGB_32f, true , false>); break;
            case fmt_PRGB_4444_32f_Linear: IL2_RUN(LoadF32<PF_Pixel_ARGB_32f, true , true >); break;

            case fmt_VUYA_4444_8u_709:  IL2_RUN(LoadVUYA8 <false, true >); break;
            case fmt_VUYA_4444_8u:      IL2_RUN(LoadVUYA8 <false, false>); break;
            case fmt_VUYP_4444_8u_709:  IL2_RUN(LoadVUYA8 <true , true >); break;
            case fmt_VUYP_4444_8u:      IL2_RUN(LoadVUYA8 <true , false>); break;
            case fmt_VUYX_4444_8u_709:  IL2_RUN(LoadVUYA8 <false, true >); break;
            case fmt_VUYX_4444_8u:      IL2_RUN(LoadVUYA8 <false, false>); break;
            case fmt_VUYA_4444_32f_709: IL2_RUN(LoadVUYA32<false, true >); break;
            case fmt_VUYA_4444_32f:     IL2_RUN(LoadVUYA32<false, false>); break;
            case fmt_VUYP_4444_32f_709: IL2_RUN(LoadVUYA32<true , true >); break;
            case fmt_VUYP_4444_32f:     IL2_RUN(LoadVUYA32<true , false>); break;
            case fmt_VUYX_4444_32f_709: IL2_RUN(LoadVUYA32<false, true >); break;
            case fmt_VUYX_4444_32f:     IL2_RUN(LoadVUYA32<false, false>); break;

            case fmt_RGB_444_10u:       IL2_RUN(LoadRGB10); break;
            default: break;
        }
        #undef IL2_RUN
    }

} // namespace avx2
} // namespace AlgoPrIngest

#endif // __IMAGELAB2_MEASURE_AVX2_HPP__
