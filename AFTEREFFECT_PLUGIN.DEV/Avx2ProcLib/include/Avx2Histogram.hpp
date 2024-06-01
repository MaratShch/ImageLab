#pragma once

#include <climits>
#include <immintrin.h>
#include "Avx2Log.hpp"
#include "Common.hpp"
#include "CommonPixFormat.hpp"


namespace AVX2
{
	namespace Histogram
	{
		typedef uint32_t HistBin;

		static constexpr size_t HistSizeImg8u  = (UCHAR_MAX + 1) * sizeof(int); /* valid pixels values in range: 0...255	*/
		static constexpr size_t HistSizeImg10u = (1023 + 1) * sizeof(int);      /* valid pixels values in range: 0...1023  */
		static constexpr size_t HistSizeImg16u = (SHRT_MAX + 1) * sizeof(int); /* valid pixels values in range: 0...32767 */
		static constexpr size_t Avx2BitsSize   = 256;
		static constexpr size_t Avx2BytesSize  = Avx2BitsSize / 8;

		constexpr float HistFloatPointNorm = static_cast<float>(u16_value_white);

		inline void clean_hist_buffer(void* __restrict pBuffer, const size_t bytesSize) noexcept
		{
			const __m256d  zVal{ 0 };
			__m256d* p = reinterpret_cast<__m256d*>(pBuffer);
			const size_t loopCnt = bytesSize / Avx2BytesSize;

			for (size_t i = 0; i < loopCnt; i++)
			{
				*p++ = zVal;
			}

			return;
		}


		void make_histogram_BGRA4444_8u
		(
			const PF_Pixel_BGRA_8u* __restrict pImage,
			HistBin*  __restrict  pFinalHistogramR,
			HistBin*  __restrict  pFinalHistogramG,
			HistBin*  __restrict  pFinalHistogramB,
			A_long histBufSizeBytes,
			A_long sizeX,
			A_long sizeY,
			A_long linePitch
		) noexcept;

		void make_histogram_BGRA4444_16u
		(
			const PF_Pixel_BGRA_16u* __restrict pImage,
			HistBin*  __restrict  pFinalHistogramR,
			HistBin*  __restrict  pFinalHistogramG,
			HistBin*  __restrict  pFinalHistogramB,
			A_long histBufSizeBytes,
			A_long sizeX,
			A_long sizeY,
			A_long linePitch
		) noexcept;

		void make_luma_histogram_VUYA4444_8u
		(
			const PF_Pixel_VUYA_8u* __restrict pImage,
			HistBin*  __restrict  pFinalHistogramY,
			A_long histBufSizeBytes,
			A_long sizeX,
			A_long sizeY,
			A_long linePitch
		) noexcept;

		void make_luma_histogram_VUYA4444_32f
		(
			const PF_Pixel_VUYA_32f* __restrict pImage,
			HistBin*  __restrict  pFinalHistogramY,
			A_long histBufSizeBytes,
			A_long sizeX,
			A_long sizeY,
			A_long linePitch
		) noexcept;

		inline void make_histogram_binarization
		(
			const HistBin* __restrict pHistogram,
			HistBin* __restrict pBinHistogram,
			A_long                    histElemSize,
			A_long                    noiseLevel
		) noexcept
		{
			const A_long noise = (noiseLevel > 0 ? noiseLevel : 1);
			constexpr A_long elemSize = static_cast<A_long>(sizeof(pHistogram[0]));
			constexpr A_long loadElems = Avx2BytesSize / elemSize;
			const A_long loopCnt = histElemSize / (loadElems * 2);
			const A_long loopCntFrac = histElemSize - loopCnt * loadElems * 2;

			const __m256i* __restrict pInVector = reinterpret_cast<const __m256i* __restrict>(pHistogram);
			__m256i* __restrict pOutVector = reinterpret_cast<__m256i* __restrict>(pBinHistogram);
			const __m256i noisePacket = _mm256_setr_epi32(noise, noise, noise, noise, noise, noise, noise, noise);

			for (A_long x = 0; x < loopCnt; x++)
			{
				/* load 2 vectors 8 x 32 bits*/
				__m256i packetVal1 = _mm256_loadu_si256(pInVector++);
				__m256i packetVal2 = _mm256_loadu_si256(pInVector++);

				/* compare each vector with zero */
				__m256i cmpVal1 = _mm256_cmpgt_epi32(packetVal1, noisePacket);
				__m256i cmpVal2 = _mm256_cmpgt_epi32(packetVal2, noisePacket);

				/* shift each result rigth on 31 bits for make binarization */
				__m256i binVal1 = _mm256_srli_epi32(cmpVal1, 31);
				__m256i binVal2 = _mm256_srli_epi32(cmpVal2, 31);

				/* store binarized histogram */
				_mm256_storeu_si256(pOutVector++, binVal1);
				_mm256_storeu_si256(pOutVector++, binVal2);
			}
			const HistBin* __restrict p1 = reinterpret_cast<const HistBin* __restrict>(pInVector);
			      HistBin* __restrict p2 = reinterpret_cast<      HistBin* __restrict>(pOutVector);
			for (A_long x = 0; x < loopCntFrac; x++)
				p2[x] = p1[x] > noise ? 1 : 0;

			return;
		}


	}
}