#include "Common.hpp"
#include "CommonPixFormat.hpp"
#include "Avx2ColorConvert.hpp"

/* AVX2 optimizations */
void AVX2::ColorConvert::BGRA8u_to_VUYA8u
(
	const PF_Pixel_BGRA_8u* __restrict pSrcImage,
	      PF_Pixel_VUYA_8u* __restrict pDstImage,
	A_long sizeX,
	A_long sizeY,
	A_long linePitch
) noexcept
{
	constexpr A_long pixSize = static_cast<A_long>(PF_Pixel_BGRA_8u_size);
	constexpr A_long loadElems = Avx2BytesSize / pixSize;
	const A_long loopCnt = sizeX / loadElems;
	const A_long loopFract = sizeX - loadElems * loopCnt;

	const __m256i& errCorr = _mm256_setr_epi32(
		InternalColorConvert::Mult,
		InternalColorConvert::Mult,
		InternalColorConvert::Mult,
		InternalColorConvert::Mult,
		InternalColorConvert::Mult,
		InternalColorConvert::Mult,
		InternalColorConvert::Mult,
		InternalColorConvert::Mult);

	/* Coefficients for compute Y (mask ALPHA channel by zero) */
	const __m256i& coeffY = _mm256_setr_epi16
	(
		InternalColorConvert::yB, InternalColorConvert::yG, InternalColorConvert::yR, 0,
		InternalColorConvert::yB, InternalColorConvert::yG, InternalColorConvert::yR, 0,
		InternalColorConvert::yB, InternalColorConvert::yG, InternalColorConvert::yR, 0,
		InternalColorConvert::yB, InternalColorConvert::yG, InternalColorConvert::yR, 0
	);

	/* Coefficients for compute U (mask ALPHA channel by zero) */
	const __m256i& coeffU = _mm256_setr_epi16
	(
		InternalColorConvert::uB, InternalColorConvert::uG, InternalColorConvert::uR, 0,
		InternalColorConvert::uB, InternalColorConvert::uG, InternalColorConvert::uR, 0,
		InternalColorConvert::uB, InternalColorConvert::uG, InternalColorConvert::uR, 0,
		InternalColorConvert::uB, InternalColorConvert::uG, InternalColorConvert::uR, 0
	);

	/* Coefficients for compute U (mask ALPHA channel by zero) */
	const __m256i& coeffV = _mm256_setr_epi16
	(
		InternalColorConvert::vB, InternalColorConvert::vG, InternalColorConvert::vR, 0,
		InternalColorConvert::vB, InternalColorConvert::vG, InternalColorConvert::vR, 0,
		InternalColorConvert::vB, InternalColorConvert::vG, InternalColorConvert::vR, 0,
		InternalColorConvert::vB, InternalColorConvert::vG, InternalColorConvert::vR, 0
	);

	for (A_long y = 0; y < sizeY; y++)
	{
		const __m256i* __restrict pSrcBufVector = reinterpret_cast<const __m256i* __restrict>(pSrcImage + y * linePitch);
		      __m256i* __restrict pDstBufVector = reinterpret_cast<      __m256i* __restrict>(pDstImage + y * linePitch);

		/* AVX2 vector part */
		for (A_long x = 0; x < loopCnt; x++)
		{
			/* non-aligned load 8 packet pixels at once */
			const __m256i packetSrcPix = _mm256_loadu_si256 (pSrcBufVector++);
            __m256i valYuv = Convert_bgra2vuya_8u(packetSrcPix);// , errCorr, coeffY, coeffU, coeffV);
			_mm256_storeu_si256(pDstBufVector++, valYuv);
		} /* for (x = 0; x < loopCnt; x++) */

		/* scalar process for fraction pixels */
		const PF_Pixel_BGRA_8u* pSrcScalar = reinterpret_cast<const PF_Pixel_BGRA_8u*>(pSrcBufVector);
		      PF_Pixel_VUYA_8u* pDstScalar = reinterpret_cast<      PF_Pixel_VUYA_8u*>(pDstBufVector);
		for (A_long x = 0; x < loopFract; x++)
		{
			pDstScalar[x].A = pSrcScalar[x].A;

			pDstScalar[x].Y = 
			(
				static_cast<int32_t>(pSrcScalar[x].R) * InternalColorConvert::yR +
				static_cast<int32_t>(pSrcScalar[x].G) * InternalColorConvert::yG +
				static_cast<int32_t>(pSrcScalar[x].R) * InternalColorConvert::yB
			) >> InternalColorConvert::Shift;

			pDstScalar[x].U =
			(
				static_cast<int32_t>(pSrcScalar[x].R) * InternalColorConvert::uR +
				static_cast<int32_t>(pSrcScalar[x].G) * InternalColorConvert::uG +
				static_cast<int32_t>(pSrcScalar[x].R) * InternalColorConvert::uB
			) >> InternalColorConvert::Shift;

			pDstScalar[x].U =
			(
				static_cast<int32_t>(pSrcScalar[x].R) * InternalColorConvert::vR +
				static_cast<int32_t>(pSrcScalar[x].G) * InternalColorConvert::vG +
				static_cast<int32_t>(pSrcScalar[x].R) * InternalColorConvert::vB
			) >> InternalColorConvert::Shift;

		} /* for (A_long x = 0; x < loopFract; x++) */
	}/* for (y = 0; y < sizeY; y++) */

	return;
}


/* AVX2 optimizations */
void AVX2::ColorConvert::VUYA8u_to_BGRA8u
(
	const PF_Pixel_VUYA_8u* __restrict pSrcImage,
	      PF_Pixel_BGRA_8u* __restrict pDstImage,
	A_long sizeX,
	A_long sizeY,
	A_long linePitch
) noexcept
{

	return;
}