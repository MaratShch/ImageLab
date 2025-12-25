#include <immintrin.h>
#include <cstdint>

#include "Common.hpp"
#include "CommonPixFormat.hpp"
#include "CommonAuxPixFormat.hpp"
#include "FastAriphmetics.hpp"


FORCE_INLINE void StorePackedLAB_Fast(float* RESTRICT dst, __m256 L, __m256 a, __m256 b)
{
    // Dump registers to aligned stack buffer
    CACHE_ALIGN float buf[24];
    _mm256_store_ps(&buf[0], L);
    _mm256_store_ps(&buf[8], a);
    _mm256_store_ps(&buf[16], b);

    // Unrolled assignment to allow Store Combining
    dst[0] = buf[0]; dst[1] = buf[8]; dst[2] = buf[16];
    dst[3] = buf[1]; dst[4] = buf[9]; dst[5] = buf[17];
    dst[6] = buf[2]; dst[7] = buf[10]; dst[8] = buf[18];
    dst[9] = buf[3]; dst[10] = buf[11]; dst[11] = buf[19];
    dst[12] = buf[4]; dst[13] = buf[12]; dst[14] = buf[20];
    dst[15] = buf[5]; dst[16] = buf[13]; dst[17] = buf[21];
    dst[18] = buf[6]; dst[19] = buf[14]; dst[20] = buf[22];
    dst[21] = buf[7]; dst[22] = buf[15]; dst[23] = buf[23];
}

// -----------------------------------------------------------------------------------------
// MAIN FUNCTION: Planar L + Interleaved AB  -->  Packed Lab
// -----------------------------------------------------------------------------------------
void CIELabPlanar2Interleaved
(
    const float*      RESTRICT pSrcL,
    const float*      RESTRICT pSrcAB,
    fCIELabPix*       RESTRICT pDstLab,
    const int32_t     sizeX,
    const int32_t     sizeY,
    const int32_t     lPitch,    // Bytes
    const int32_t     abPitch,   // Bytes
    const int32_t     dstPitch   // Bytes
) noexcept
{
    const uint8_t* rowL  = (const uint8_t*)pSrcL;
    const uint8_t* rowAB = (const uint8_t*)pSrcAB;
    uint8_t*       rowDst= (uint8_t*)pDstLab;

    for (int y = 0; y < sizeY; ++y)
    {
        const float* S_L  = (const float*)rowL;
        const float* S_AB = (const float*)rowAB;
        float*       D    = (float*)rowDst; // Cast struct to float array for indexing

        int x = 0;
        
        // Vector Loop (8 pixels)
        for (; x <= sizeX - 8; x += 8)
        {
            // 1. Load L (Planar) -> [L0 L1 L2 L3 L4 L5 L6 L7]
            __m256 L = _mm256_loadu_ps(S_L + x);

            // 2. Load AB (Interleaved) -> 2 Vectors (16 floats total)
            // v0 = [a0 b0 a1 b1 | a2 b2 a3 b3]
            // v1 = [a4 b4 a5 b5 | a6 b6 a7 b7]
            __m256 v0 = _mm256_loadu_ps(S_AB + 2*x);
            __m256 v1 = _mm256_loadu_ps(S_AB + 2*x + 8);

            // 3. De-interleave AB (CORRECTED LOGIC)
            
            // Step A: Group Low 128-bits and High 128-bits
            // pA = [v0_lo | v1_lo] = [a0 b0 a1 b1 | a4 b4 a5 b5]
            __m256 pA = _mm256_permute2f128_ps(v0, v1, 0x20); 
            // pB = [v0_hi | v1_hi] = [a2 b2 a3 b3 | a6 b6 a7 b7]
            __m256 pB = _mm256_permute2f128_ps(v0, v1, 0x31);

            // Step B: Shuffle to separate a (even indices) and b (odd indices)
            // Mask 0x88 (2,0,2,0) selects indices 0 and 2 from both lanes
            // A = [a0 a1 a4 a5] (from pA) mixed with [a2 a3 a6 a7] (from pB)
            // Wait, standard shuffle_ps takes src1, src2.
            // Lane 0 Output: src1[0], src1[2], src2[0], src2[2]
            // Lane 0: pA[0](a0), pA[2](a1), pB[0](a2), pB[2](a3) -> a0 a1 a2 a3 (CORRECT)
            // Lane 1 Output: src1[4+0], src1[4+2], src2[4+0], src2[4+2]
            // Lane 1: pA[4](a4), pA[6](a5), pB[4](a6), pB[6](a7) -> a4 a5 a6 a7 (CORRECT)
            
            __m256 A = _mm256_shuffle_ps(pA, pB, _MM_SHUFFLE(2, 0, 2, 0));
            
            // Mask 0xDD (3,1,3,1) selects indices 1 and 3
            // Lane 0: pA[1](b0), pA[3](b1), pB[1](b2), pB[3](b3) -> b0 b1 b2 b3 (CORRECT)
            // Lane 1: pA[5](b4), pA[7](b5), pB[5](b6), pB[7](b7) -> b4 b5 b6 b7 (CORRECT)
            
            __m256 B = _mm256_shuffle_ps(pA, pB, _MM_SHUFFLE(3, 1, 3, 1));

            // 4. Store using optimized 3-channel packer
            StorePackedLAB_Fast(D + 3*x, L, A, B);
        }

        // Scalar Tail
        for (; x < sizeX; ++x)
        {
            D[3*x + 0] = S_L[x];
            D[3*x + 1] = S_AB[2*x + 0];
            D[3*x + 2] = S_AB[2*x + 1];
        }

        rowL   += lPitch * sizeof(float);
        rowAB  += abPitch * 2 * sizeof(float);
        rowDst += dstPitch * sizeof(fCIELabPix);
    }
}