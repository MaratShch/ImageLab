#pragma once

#include <immintrin.h>

namespace AVX2
{
	namespace MiscUtils
	{
		inline void cum_sum_uint32 (const uint32_t* __restrict src, uint32_t* __restrict dst, const uint32_t& size) noexcept
		{
			constexpr int32_t permuteMask1 = (2 << 6) | (1 << 4) | 3;
			constexpr int32_t permuteMask2 = (1 << 6) | (3 << 2) | 2;
			__m256i offset = _mm256_setzero_si256();
			for (int32_t i = 0; i < size; i += 8)
			{
				__m256i srcVec = _mm256_loadu_si256(reinterpret_cast<const __m256i* __restrict>(&src[i]));
				__m256i t0 = _mm256_shuffle_epi32(srcVec, permuteMask1);
				__m256i t1 = _mm256_permute2f128_si256(t0, t0, 41); 
				__m256i sample = _mm256_add_epi32(srcVec, _mm256_blend_epi32(t0, t1, 0x11)); 
				t0 = _mm256_shuffle_epi32(sample, permuteMask2); 
				t1 = _mm256_permute2f128_si256(t0, t0, 41); 
				sample = _mm256_add_epi32(sample, _mm256_blend_epi32(t0, t1, 0x33));
				__m256i outVec = _mm256_add_epi32(sample, _mm256_permute2f128_si256(sample, sample, 41)); 
				outVec = _mm256_add_epi32(outVec, offset);
				_mm256_storeu_si256(reinterpret_cast<__m256i*>(&dst[i]), outVec);
				__m256i t2 = _mm256_permute2f128_si256(outVec, outVec, 0x11);
				offset = _mm256_shuffle_epi32(t2, 0xff);
			}

			return;
		}

		inline void generate_lut_uint32 (const uint32_t* __restrict pCumSum, uint32_t* __restrict pLut, const float coeff, int32_t size) noexcept
		{
			const __m256 fMult = _mm256_set1_ps(coeff);
			for (int32_t i = 0; i < size; i += 8)
			{
				__m256 srcVec = _mm256_cvtepi32_ps(_mm256_loadu_si256(reinterpret_cast<const __m256i* __restrict>(&pCumSum[i])));
				__m256 mult = _mm256_mul_ps(srcVec, fMult);
				__m256i outVec = _mm256_cvtps_epi32(mult);
				_mm256_storeu_si256(reinterpret_cast<__m256i*>(&pLut[i]), outVec);
			}
			return;
		}

	}
}