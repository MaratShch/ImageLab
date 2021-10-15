#include "Avx2ColorConvert.hpp"

constexpr int32_t Shift = 14;
constexpr int32_t Mult = 1 << Shift;

static_assert(Shift <= 14 && Shift > 0, "Invalid shift value");

/* matrix coefficients for convert RGB to YUV */
constexpr int32_t yR = 0.299000f * Mult;
constexpr int32_t yG = 0.587000f * Mult;
constexpr int32_t yB = 0.114000f * Mult;

constexpr int32_t uR = -0.168736f * Mult;
constexpr int32_t uG = -0.331264f * Mult;
constexpr int32_t uB = 0.500000f  * Mult;

constexpr int32_t vR = 0.500000f  * Mult;
constexpr int32_t vG = -0.418688f * Mult;
constexpr int32_t vB = -0.081312f * Mult;

/* Matrix coefficients for convert YUV to RGB */

/*

bgraX8 contains packet 8u BGRA pixels in folowing format:

+----+----+----+----+----+----+----+----+-----+----+----+----+----+
| b0 | g0 | r0 | a0 | b1 | g1 | r1 | a1 | ... | b7 | g7 | r7 | a7 |
+----+----+----+----+----+----+----+----+-----+----+----+----+----+

*/
__m256i AVX2::ColorConvert::Convert_bgra2vuya_8u(const __m256i& bgraX8) noexcept
{
	const __m256i& errCorr = _mm256_setr_epi32(Mult, Mult, Mult, Mult, Mult, Mult, Mult, Mult);

	/* Coefficients for compute Y (mask ALPHA channel by zero) */
	const __m256i& coeffY = _mm256_setr_epi16
	(
		yB, yG, yR, 0,
		yB, yG, yR, 0,
		yB, yG, yR, 0,
		yB, yG, yR, 0
	);

	/* Coefficients for compute U (mask ALPHA channel by zero) */
	const __m256i& coeffU = _mm256_setr_epi16
	(
		uB, uG, uR, 0,
		uB, uG, uR, 0,
		uB, uG, uR, 0,
		uB, uG, uR, 0
	);

	/* Coefficients for compute U (mask ALPHA channel by zero) */
	const __m256i& coeffV = _mm256_setr_epi16
	(
		vB, vG, vR, 0,
		vB, vG, vR, 0,
		vB, vG, vR, 0,
		vB, vG, vR, 0
	);

	/* extract 4 low pixels from uint8_t to int16_t */
	__m256i bgra03 = _mm256_cvtepu8_epi16(_mm256_extracti128_si256(bgraX8, 0));
	/* extract 4 high pixels from uint8_t to int16_t */
	__m256i bgra47 = _mm256_cvtepu8_epi16(_mm256_extracti128_si256(bgraX8, 1));

	/* COMPUTE Y COMPONENT */
	/* DOT operation for pixels 0 - 3:  B * coeff[0] + G * coeff[1]; R * coeff[2] + A * coeff[3] */
	__m256i dotY1 = _mm256_madd_epi16(bgra03, coeffY);
	/* DOT operation for pixels 4 - 7:  B * coeff[0] + G * coeff[1]; R * coeff[2] + A * coeff[3] */
	__m256i dotY2 = _mm256_madd_epi16(bgra47, coeffY);

	/* SUM between DOT elements */
	__m256i addDotsY1 = _mm256_add_epi32(dotY1, _mm256_slli_epi64(dotY1, 32));
	__m256i addDotsY2 = _mm256_add_epi32(dotY2, _mm256_slli_epi64(dotY2, 32));

	/* final shift for get Y value: normalize ariphmetic result */
	__m256i Y03 = _mm256_srai_epi32(addDotsY1, Shift);
	__m256i Y47 = _mm256_srai_epi32(addDotsY2, Shift);

	/* COMPUTE U COMPONENT */
	/* DOT operation for pixels 0 - 3:  B * coeff[0] + G * coeff[1]; R * coeff[2] + A * coeff[3] */
	__m256i dotU1 = _mm256_madd_epi16(bgra03, coeffU);
	/* DOT operation for pixels 4 - 7:  B * coeff[0] + G * coeff[1]; R * coeff[2] + A * coeff[3] */
	__m256i dotU2 = _mm256_madd_epi16(bgra47, coeffU);

	/* SUM between DOT elements */
	__m256i addDotsU1 = _mm256_add_epi32(dotU1, _mm256_slli_epi64(dotU1, 32));
	__m256i addDotsU2 = _mm256_add_epi32(dotU2, _mm256_slli_epi64(dotU2, 32));

	/* final shift for get Y value: normalize ariphmetic result */
	__m256i U03 = _mm256_srai_epi32(addDotsU1, Shift);
	__m256i U47 = _mm256_srai_epi32(addDotsU2, Shift);

	/* COMPUTE V COMPONENT */
	__m256i dotV1 = _mm256_madd_epi16(bgra03, coeffV);
	/* DOT operation for pixels 4 - 7:  B * coeff[0] + G * coeff[1]; R * coeff[2] + A * coeff[3] */
	__m256i dotV2 = _mm256_madd_epi16(bgra47, coeffV);

	/* SUM between DOT elements */
	__m256i addDotsV1 = _mm256_add_epi32(dotV1, _mm256_slli_epi64(dotV1, 32));
	__m256i addDotsV2 = _mm256_add_epi32(dotV2, _mm256_slli_epi64(dotV2, 32));

	/* final shift for get Y value: normalize ariphmetic result */
	__m256i V03 = _mm256_srai_epi32(addDotsV1, Shift);
	__m256i V47 = _mm256_srai_epi32(addDotsV2, Shift);

	/* final PACK to VUYA vector */
	/* compine all Y values to single vector 32 bits */
	const __m256i permute_idx = _mm256_setr_epi32(1, 3, 5, 7, 0, 2, 4, 6);
	const __m256i addUV = _mm256_setr_epi32(128, 128, 128, 128, 128, 128, 128, 128);
	__m256i Y = _mm256_permute2x128_si256(_mm256_permutevar8x32_epi32(Y03, permute_idx), _mm256_permutevar8x32_epi32(Y47, permute_idx), 0x20);
	__m256i U = _mm256_add_epi32(addUV, _mm256_permute2x128_si256(_mm256_permutevar8x32_epi32(U03, permute_idx), _mm256_permutevar8x32_epi32(U47, permute_idx), 0x20));
	__m256i V = _mm256_add_epi32(addUV, _mm256_permute2x128_si256(_mm256_permutevar8x32_epi32(V03, permute_idx), _mm256_permutevar8x32_epi32(V47, permute_idx), 0x20));

	/* prepare Y [set Y value on cpecific byte positions] */
	Y = _mm256_shuffle_epi8(Y,
		_mm256_setr_epi8(
			128, 128, 0, 128,
			128, 128, 4, 128,
			128, 128, 8, 128,
			128, 128, 12, 128,
			128, 128, 16, 128,
			128, 128, 20, 128,
			128, 128, 24, 128,
			128, 126, 28, 128)
	);

	/* prepare U [set U value on cpecific byte positions] */
	U = _mm256_shuffle_epi8(U,
		_mm256_setr_epi8(
			128, 0, 128, 128,
			128, 4, 128, 128,
			128, 8, 128, 128,
			128, 12, 128, 128,
			128, 16, 128, 128,
			128, 20, 128, 128,
			128, 24, 128, 128,
			128, 28, 128, 128)
	);

	/* compbine V and U component */
	__m256i VU = _mm256_or_si256(V, U);

	/* combine VU and Y component */
	__m256i VUY = _mm256_or_si256(VU, Y);

	/* combine VUY with ALPHA channel from source pixel */
	__m256i VUYA = _mm256_blendv_epi8(bgraX8, VUY,
		_mm256_setr_epi32(0xFFFFFF, 0xFFFFFF, 0xFFFFFF, 0xFFFFFF, 0xFFFFFF, 0xFFFFFF, 0xFFFFFF, 0xFFFFFF)
	);

	return VUYA;
}

__m256i AVX2::ColorConvert::Convert_vuya2bgra_8u (const __m256i& bgraX8) noexcept
{
	return{ 0 };
}