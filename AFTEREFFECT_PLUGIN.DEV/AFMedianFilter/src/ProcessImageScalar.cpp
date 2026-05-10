#include <cstdint>
#include <cfloat>
#include <algorithm>
#include <immintrin.h>
#include "Common.hpp"
#include "AlgoControls.hpp"


// ============================================================================
// Constants
// ============================================================================
constexpr int32_t MAX_WINDOW_AREA = (kernelRadiusMax * 2 + 1) *
                                    (kernelRadiusMax * 2 + 1);

constexpr int32_t MAX_PADDED_SIZE = 512;


// ============================================================================
// AVX2 helpers (file-local) -- min/max sweep utilities
// ============================================================================

static inline __m256i avx2_mask_for_count (const int32_t count) noexcept
{
    static CACHE_ALIGN constexpr int32_t kMaskTable[9][8] = {
        {  0,  0,  0,  0,  0,  0,  0,  0 },
        { -1,  0,  0,  0,  0,  0,  0,  0 },
        { -1, -1,  0,  0,  0,  0,  0,  0 },
        { -1, -1, -1,  0,  0,  0,  0,  0 },
        { -1, -1, -1, -1,  0,  0,  0,  0 },
        { -1, -1, -1, -1, -1,  0,  0,  0 },
        { -1, -1, -1, -1, -1, -1,  0,  0 },
        { -1, -1, -1, -1, -1, -1, -1,  0 },
        { -1, -1, -1, -1, -1, -1, -1, -1 },
    };
    return _mm256_load_si256(reinterpret_cast<const __m256i*>(kMaskTable[count]));
}

static inline float avx2_hmin_ps (const __m256 v) noexcept
{
    __m128 lo = _mm256_castps256_ps128(v);
    __m128 hi = _mm256_extractf128_ps(v, 1);
    __m128 r  = _mm_min_ps(lo, hi);
    r = _mm_min_ps(r, _mm_movehl_ps(r, r));
    r = _mm_min_ss(r, _mm_shuffle_ps(r, r, _MM_SHUFFLE(0, 0, 0, 1)));
    return _mm_cvtss_f32(r);
}

static inline float avx2_hmax_ps (const __m256 v) noexcept
{
    __m128 lo = _mm256_castps256_ps128(v);
    __m128 hi = _mm256_extractf128_ps(v, 1);
    __m128 r  = _mm_max_ps(lo, hi);
    r = _mm_max_ps(r, _mm_movehl_ps(r, r));
    r = _mm_max_ss(r, _mm_shuffle_ps(r, r, _MM_SHUFFLE(0, 0, 0, 1)));
    return _mm_cvtss_f32(r);
}


// ============================================================================
// Fast 3x3 min/max  --  the workhorse of the early-exit fast path.
//
// For 9 pixels, 128-bit SSE is a better fit than 256-bit AVX. We issue three
// 4-wide unaligned loads (one per row), perform vertical reductions across
// the rows, then a 4 -> 2 -> 1 horizontal reduction.
//
// Each row is read as 4 floats with the 4th lane being an overread of one
// pixel past the 3x3 window. The padding allocation (RADIUS_MAX = 8 pixels
// on every side, reserved by the allocator) guarantees the overread lands
// in valid memory. The blend at the end masks lane 3 with +/-FLT_MAX so
// whatever happens to live at that overread address cannot affect the
// reduced min/max.
// ============================================================================
static inline void avx2_minmax_3x3
(
    const float* RESTRICT centerPtr,
    const int32_t         strideElements,
    float&                outMin,
    float&                outMax
) noexcept
{
    // Three 4-wide loads -- one per row of the 3x3 window.
    const __m128 vRow0 = _mm_loadu_ps(centerPtr - strideElements - 1);
    const __m128 vRow1 = _mm_loadu_ps(centerPtr                  - 1);
    const __m128 vRow2 = _mm_loadu_ps(centerPtr + strideElements - 1);

    // Vertical reduce across the three rows.
    __m128 vColMin = _mm_min_ps(_mm_min_ps(vRow0, vRow1), vRow2);
    __m128 vColMax = _mm_max_ps(_mm_max_ps(vRow0, vRow1), vRow2);

    // Lane 3 holds the overread column; neutralise it before horizontal reduce.
    const __m128 kPosInf = _mm_set1_ps(FLT_MAX);
    const __m128 kNegInf = _mm_set1_ps(-FLT_MAX);
    vColMin = _mm_blend_ps(vColMin, kPosInf, 0x8);  // lane 3 -> +inf
    vColMax = _mm_blend_ps(vColMax, kNegInf, 0x8);  // lane 3 -> -inf

    // Horizontal min: 4 -> 2 -> 1
    {
        __m128 t = _mm_movehl_ps(vColMin, vColMin);
        __m128 r = _mm_min_ps(vColMin, t);
        r = _mm_min_ss(r, _mm_shuffle_ps(r, r, _MM_SHUFFLE(0, 0, 0, 1)));
        outMin = _mm_cvtss_f32(r);
    }

    // Horizontal max: 4 -> 2 -> 1
    {
        __m128 t = _mm_movehl_ps(vColMax, vColMax);
        __m128 r = _mm_max_ps(vColMax, t);
        r = _mm_max_ss(r, _mm_shuffle_ps(r, r, _MM_SHUFFLE(0, 0, 0, 1)));
        outMax = _mm_cvtss_f32(r);
    }
}


// ============================================================================
// AVX2 bitonic sort building blocks  (unchanged from previous step)
// ============================================================================

static inline __m256 avx2_reverse_8 (__m256 v) noexcept
{
    __m256 t = _mm256_permute2f128_ps(v, v, 0x01);
    return _mm256_shuffle_ps(t, t, _MM_SHUFFLE(0, 1, 2, 3));
}

static inline __m256 avx2_bsort_8 (__m256 v) noexcept
{
    __m256 t;

    t = _mm256_shuffle_ps(v, v, _MM_SHUFFLE(2, 3, 0, 1));
    v = _mm256_blend_ps(_mm256_min_ps(v, t), _mm256_max_ps(v, t), 0x66);

    t = _mm256_shuffle_ps(v, v, _MM_SHUFFLE(1, 0, 3, 2));
    v = _mm256_blend_ps(_mm256_min_ps(v, t), _mm256_max_ps(v, t), 0x3C);

    t = _mm256_shuffle_ps(v, v, _MM_SHUFFLE(2, 3, 0, 1));
    v = _mm256_blend_ps(_mm256_min_ps(v, t), _mm256_max_ps(v, t), 0x5A);

    t = _mm256_permute2f128_ps(v, v, 0x01);
    v = _mm256_blend_ps(_mm256_min_ps(v, t), _mm256_max_ps(v, t), 0xF0);

    t = _mm256_shuffle_ps(v, v, _MM_SHUFFLE(1, 0, 3, 2));
    v = _mm256_blend_ps(_mm256_min_ps(v, t), _mm256_max_ps(v, t), 0xCC);

    t = _mm256_shuffle_ps(v, v, _MM_SHUFFLE(2, 3, 0, 1));
    v = _mm256_blend_ps(_mm256_min_ps(v, t), _mm256_max_ps(v, t), 0xAA);

    return v;
}

static inline void avx2_reverse_half (__m256* vecs, const int32_t M) noexcept
{
    for (int32_t i = 0; i < M / 2; ++i)
    {
        const __m256 a = avx2_reverse_8(vecs[i]);
        const __m256 b = avx2_reverse_8(vecs[M - 1 - i]);
        vecs[i]         = b;
        vecs[M - 1 - i] = a;
    }
    if (M & 1)
    {
        vecs[M / 2] = avx2_reverse_8(vecs[M / 2]);
    }
}

static inline void avx2_bstage_inter (__m256* vecs, const int32_t VC, const int32_t D) noexcept
{
    const int32_t VD = D >> 3;
    for (int32_t blockStart = 0; blockStart < VC; blockStart += 2 * VD)
    {
        for (int32_t k = 0; k < VD; ++k)
        {
            const int32_t loIdx = blockStart + k;
            const int32_t hiIdx = blockStart + k + VD;
            const __m256 lo = vecs[loIdx];
            const __m256 hi = vecs[hiIdx];
            vecs[loIdx] = _mm256_min_ps(lo, hi);
            vecs[hiIdx] = _mm256_max_ps(lo, hi);
        }
    }
}

static inline void avx2_bstage_intra_4 (__m256* vecs, const int32_t VC) noexcept
{
    for (int32_t i = 0; i < VC; ++i)
    {
        const __m256 v = vecs[i];
        const __m256 t = _mm256_permute2f128_ps(v, v, 0x01);
        vecs[i] = _mm256_blend_ps(_mm256_min_ps(v, t), _mm256_max_ps(v, t), 0xF0);
    }
}

static inline void avx2_bstage_intra_2 (__m256* vecs, const int32_t VC) noexcept
{
    for (int32_t i = 0; i < VC; ++i)
    {
        const __m256 v = vecs[i];
        const __m256 t = _mm256_shuffle_ps(v, v, _MM_SHUFFLE(1, 0, 3, 2));
        vecs[i] = _mm256_blend_ps(_mm256_min_ps(v, t), _mm256_max_ps(v, t), 0xCC);
    }
}

static inline void avx2_bstage_intra_1 (__m256* vecs, const int32_t VC) noexcept
{
    for (int32_t i = 0; i < VC; ++i)
    {
        const __m256 v = vecs[i];
        const __m256 t = _mm256_shuffle_ps(v, v, _MM_SHUFFLE(2, 3, 0, 1));
        vecs[i] = _mm256_blend_ps(_mm256_min_ps(v, t), _mm256_max_ps(v, t), 0xAA);
    }
}

static inline void avx2_bmerge (__m256* vecs, const int32_t VC) noexcept
{
    const int32_t N = VC * 8;
    for (int32_t D = N / 2; D >= 8; D >>= 1)
    {
        avx2_bstage_inter(vecs, VC, D);
    }
    avx2_bstage_intra_4(vecs, VC);
    avx2_bstage_intra_2(vecs, VC);
    avx2_bstage_intra_1(vecs, VC);
}

static inline void avx2_bsort (__m256* vecs, const int32_t VC) noexcept
{
    for (int32_t i = 0; i < VC; ++i)
    {
        vecs[i] = avx2_bsort_8(vecs[i]);
    }

    for (int32_t blockVC = 2; blockVC <= VC; blockVC <<= 1)
    {
        for (int32_t start = 0; start < VC; start += blockVC)
        {
            const int32_t half = blockVC / 2;
            avx2_reverse_half(vecs + start + half, half);
            avx2_bmerge(vecs + start, blockVC);
        }
    }
}


// ============================================================================
// AnalyzeAdaptiveWindow  --  AVX2-accelerated (unchanged)
// ============================================================================
inline void AnalyzeAdaptiveWindow
(
    const float* RESTRICT centerPtr,
    const int32_t         strideElements,
    const int32_t         maxRadius,
    int32_t&              outBestRadius,
    float&                outMin,
    float&                outMax
) noexcept
{
    CACHE_ALIGN float window[MAX_WINDOW_AREA];

    const __m256 kPosInf = _mm256_set1_ps(FLT_MAX);
    const __m256 kNegInf = _mm256_set1_ps(-FLT_MAX);

    for (int32_t r = 1; r <= maxRadius; ++r)
    {
        const int32_t width      = 2 * r + 1;
        const int32_t totalCount = width * width;

        __m256  vMin     = kPosInf;
        __m256  vMax     = kNegInf;
        int32_t writeIdx = 0;

        for (int32_t wy = -r; wy <= r; ++wy)
        {
            const float* RESTRICT rowStart = centerPtr + wy * strideElements - r;

            int32_t off       = 0;
            int32_t remaining = width;

            while (remaining >= 8)
            {
                const __m256 vData = _mm256_loadu_ps(rowStart + off);
                _mm256_storeu_ps(window + writeIdx, vData);
                vMin = _mm256_min_ps(vMin, vData);
                vMax = _mm256_max_ps(vMax, vData);
                off       += 8;
                writeIdx  += 8;
                remaining -= 8;
            }

            if (remaining > 0)
            {
                const __m256i vMask  = avx2_mask_for_count(remaining);
                const __m256  vMaskF = _mm256_castsi256_ps(vMask);
                const __m256  vData  = _mm256_maskload_ps(rowStart + off, vMask);
                _mm256_maskstore_ps(window + writeIdx, vMask, vData);

                const __m256 vForMin = _mm256_blendv_ps(kPosInf, vData, vMaskF);
                const __m256 vForMax = _mm256_blendv_ps(kNegInf, vData, vMaskF);
                vMin = _mm256_min_ps(vMin, vForMin);
                vMax = _mm256_max_ps(vMax, vForMax);

                writeIdx += remaining;
            }
        }

        outMin = avx2_hmin_ps(vMin);
        outMax = avx2_hmax_ps(vMax);

        const int32_t halfIdx = totalCount / 2;
        std::nth_element(window, window + halfIdx, window + totalCount);
        const float currentMedian = window[halfIdx];

        if (currentMedian > outMin && currentMedian < outMax)
        {
            outBestRadius = r;
            return;
        }
    }

    outBestRadius = maxRadius;
}


// ============================================================================
// FrequencyMedian  --  AVX2 bitonic sort + scalar dedup (unchanged)
// ============================================================================
inline float FrequencyMedian
(
    const float* RESTRICT centerPtr,
    const int32_t         strideElements,
    const int32_t         radius
) noexcept
{
    static constexpr CACHE_ALIGN int32_t kPaddedSizeByRadius[9] =
    {
        /* r=0 (unused) */  16,
        /* r=1:   9 -> */  16,
        /* r=2:  25 -> */  32,
        /* r=3:  49 -> */  64,
        /* r=4:  81 -> */ 128,
        /* r=5: 121 -> */ 128,
        /* r=6: 169 -> */ 256,
        /* r=7: 225 -> */ 256,
        /* r=8: 289 -> */ 512,
    };

    const int32_t width      = 2 * radius + 1;
    const int32_t totalCount = width * width;
    const int32_t paddedSize = kPaddedSizeByRadius[radius];

    CACHE_ALIGN float window[MAX_PADDED_SIZE];

    const __m256 vMax = _mm256_set1_ps(FLT_MAX);
    for (int32_t i = 0; i < paddedSize; i += 8)
    {
        _mm256_store_ps(window + i, vMax);
    }

    int32_t idx = 0;
    for (int32_t wy = -radius; wy <= radius; ++wy)
    {
        const float* RESTRICT rowPtr = centerPtr + (wy * strideElements);
        for (int32_t wx = -radius; wx <= radius; ++wx)
        {
            window[idx++] = rowPtr[wx];
        }
    }

    const int32_t VC = paddedSize >> 3;
    avx2_bsort(reinterpret_cast<__m256*>(window), VC);

    CACHE_ALIGN float   uniqueValues[MAX_PADDED_SIZE];
    CACHE_ALIGN int32_t frequencies [MAX_PADDED_SIZE];

    uniqueValues[0] = window[0];
    frequencies[0]  = 1;
    int32_t uniqueCount = 1;

    for (int32_t i = 1; i < totalCount; ++i)
    {
        if (window[i] == uniqueValues[uniqueCount - 1])
        {
            frequencies[uniqueCount - 1]++;
        }
        else
        {
            uniqueValues[uniqueCount] = window[i];
            frequencies[uniqueCount]  = 1;
            uniqueCount++;
        }
    }

    if (uniqueCount % 2 != 0)
    {
        return uniqueValues[uniqueCount / 2];
    }

    const int32_t midL = (uniqueCount / 2) - 1;
    const int32_t midR =  uniqueCount / 2;

    return (frequencies[midL] <= frequencies[midR])
           ? uniqueValues[midR]
           : uniqueValues[midL];
}


// ============================================================================
// ProcessImage_Scalar  --  fast-path early-exit on r=1 min/max
//
// Per-pixel structure:
//   1. FAST PATH: compute min/max of the 3x3 window only (no median, no
//      adaptive expansion). If the centre pixel is strictly inside
//      (min1 + tol, max1 - tol), monotonicity guarantees it cannot be
//      classified as "corrupted" at any larger radius either, so we can
//      output it untouched and skip everything else.
//
//   2. SLOW PATH: only reached if the r=1 min/max bracketed the centre. We
//      then run the full adaptive analysis (which may grow the window past
//      r=1 if the median there is an impulse) and re-check corruption with
//      the resulting localMin/localMax. Only genuinely-corrupted pixels
//      reach the FrequencyMedian call.
//
// Most pixels at typical noise densities take the fast path -- and the
// fast path skips std::nth_element entirely, which is the largest single
// piece of unoptimised work in the previous revision.
// ============================================================================
void ProcessImage_Scalar
(
    const float* RESTRICT inY,
    float*       RESTRICT outY,
    const int32_t         sizeX,
    const int32_t         sizeY,
    const int32_t         strideY_Elements,
    const AlgoControls&   ctrl
)
{
    auto isPixelCorrupted = [tol = ctrl.tolerance]
        (const float center, const float lMin, const float lMax) -> bool
    {
        return (center <= (lMin + tol)) || (center >= (lMax - tol));
    };

    for (int32_t y = 0; y < sizeY; ++y)
    {
        const float* RESTRICT inRow  = inY  + (y * strideY_Elements);
        float*       RESTRICT outRow = outY + (y * strideY_Elements);

        for (int32_t x = 0; x < sizeX; ++x)
        {
            const float  centerPixel = inRow[x];
            const float* centerPtr   = &inRow[x];

            // ============ FAST PATH ============
            // Compute r=1 min/max only. No median, no window storage,
            // no adaptive loop.
            float min1, max1;
            avx2_minmax_3x3(centerPtr, strideY_Elements, min1, max1);

            // Monotonic invariant: if the centre is strictly inside the
            // r=1 bracket, it cannot be at the extremes for any larger r.
            if (!isPixelCorrupted(centerPixel, min1, max1))
            {
                outRow[x] = centerPixel;
                continue;
            }

            // ============ SLOW PATH ============
            // The centre touched a local extreme at r=1 -- it might be noise.
            // Run the full adaptive analysis (which finds the smallest r
            // whose median is non-impulse).
            int32_t bestRadius = 1;
            float   localMin   = 0.0f;
            float   localMax   = 0.0f;

            AnalyzeAdaptiveWindow
            (
                centerPtr, strideY_Elements, ctrl.radius,
                bestRadius, localMin, localMax
            );

            // Re-check corruption using the (potentially wider) window's
            // localMin / localMax. The window may have grown enough that
            // the centre is no longer at the extremes.
            if (!isPixelCorrupted(centerPixel, localMin, localMax))
            {
                outRow[x] = centerPixel;
                continue;
            }

            // Genuinely corrupted -- replace via frequency median.
            outRow[x] = FrequencyMedian
            (
                centerPtr,
                strideY_Elements,
                bestRadius
            );
        }
    }
}