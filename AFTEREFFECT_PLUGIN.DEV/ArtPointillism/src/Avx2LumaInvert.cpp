#include <immintrin.h>
#include "AlgoLumaManipulation.hpp"

/**
 * AVX2 KERNEL: Luma Invert
 * 
 * Operation: pDst[i] = 1.0f - (pSrc[i] * 0.01f)
 * Optimized using Fused Negative Multiply-Add (FNMADD).
 * Processes 32 pixels per loop iteration.
 */
void CIELab_LumaInvert
(
    const float* RESTRICT pSrc, 
    float*       RESTRICT pDst, 
    int32_t      sizeX,
    int32_t      sizeY 
) noexcept
{
    const int32_t pixelCount = sizeX * sizeY;

    // Constants
    const __m256 c_scale = _mm256_set1_ps(0.01f); // Scaling factor (1/100)
    const __m256 c_one   = _mm256_set1_ps(1.0f);  // Inversion base

    int32_t i = 0;

    // --- MAIN LOOP (Unrolled 4x = 32 pixels) ---
    // Unrolling helps hide memory latency and maximizes throughput.
    for (; i <= pixelCount - 32; i += 32)
	{
        // 1. Load 32 float values
        __m256 l0 = _mm256_loadu_ps(pSrc + i);
        __m256 l1 = _mm256_loadu_ps(pSrc + i + 8);
        __m256 l2 = _mm256_loadu_ps(pSrc + i + 16);
        __m256 l3 = _mm256_loadu_ps(pSrc + i + 24);

        // 2. Compute: 1.0 - (L * 0.01)
        // We use _mm256_fnmadd_ps(a, b, c) -> -(a * b) + c
        // This is faster and more precise than separate MUL and SUB.
        l0 = _mm256_fnmadd_ps(l0, c_scale, c_one);
        l1 = _mm256_fnmadd_ps(l1, c_scale, c_one);
        l2 = _mm256_fnmadd_ps(l2, c_scale, c_one);
        l3 = _mm256_fnmadd_ps(l3, c_scale, c_one);

        // 3. Store results
        _mm256_storeu_ps(pDst + i, l0);
        _mm256_storeu_ps(pDst + i + 8, l1);
        _mm256_storeu_ps(pDst + i + 16, l2);
        _mm256_storeu_ps(pDst + i + 24, l3);
    }

    // --- TAIL LOOP (AVX2 - 8 pixels) ---
    for (; i <= pixelCount - 8; i += 8)
	{
        __m256 l = _mm256_loadu_ps(pSrc + i);
        l = _mm256_fnmadd_ps(l, c_scale, c_one);
        _mm256_storeu_ps(pDst + i, l);
    }

    // --- SCALAR FALLBACK (Remaining pixels) ---
    for (; i < pixelCount; ++i)
	{
        pDst[i] = 1.0f - (pSrc[i] * 0.01f);
    }
	
	return;
}