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


		inline void clean_hist_buffer(void* __restrict pBuffer, const size_t bytesSize) noexcept
		{
			__m256d  zVal{ 0 };
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


		void make_histogram_binarization
		(
			const HistBin* __restrict pHistogram,
			      HistBin* __restrict pBinHistogram,
			      A_long              histElemSize,
			      A_long              noiseLevel = 0
		) noexcept;

		void make_histogram_bin_cumulative_sum
		(
			const HistBin* __restrict pHistogram,
			      HistBin* __restrict pBinHistogram,
			      HistBin* __restrict pCumSumHistogram,
			      A_long              histElemSize,
			      A_long              noiseLevel
		) noexcept;

	}
}