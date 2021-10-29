#include "Avx2ColorConvert.hpp"

namespace InternalColorConvert
{
	static constexpr int32_t Shift = 13;
	static constexpr int32_t Mult = 1 << Shift;

	static_assert(Shift <= 14 && Shift > 0, "Invalid shift value");

	/* matrix coefficients for convert RGB to YUV */
	static constexpr int32_t yR =  0.299000f * Mult;
	static constexpr int32_t yG =  0.587000f * Mult;
	static constexpr int32_t yB =  0.114000f * Mult;
	static constexpr int32_t uR = -0.168736f * Mult;
	static constexpr int32_t uG = -0.331264f * Mult;
	static constexpr int32_t uB =  0.500000f * Mult;
	static constexpr int32_t vR =  0.500000f * Mult;
	static constexpr int32_t vG = -0.418688f * Mult;
	static constexpr int32_t vB = -0.081312f * Mult;

	static constexpr int32_t rY = 1.0      * Mult;
	static constexpr int32_t rU = 0.0      * Mult;
	static constexpr int32_t rV = 1.4075   * Mult;
	static constexpr int32_t gY = 1.0      * Mult;
	static constexpr int32_t gU = -0.34550 * Mult;
	static constexpr int32_t gV = -0.71690 * Mult;
	static constexpr int32_t bY = 1.0      * Mult;
	static constexpr int32_t bU = 1.7790   * Mult;
	static constexpr int32_t bV = 0.0      * Mult;

}; /* InternalColorConvert */

/* Matrix coefficients for convert YUV to RGB */

/*

bgraX8 contains packet 8u BGRA pixels in folowing format:

+----+----+----+----+----+----+----+----+-----+----+----+----+----+
| b0 | g0 | r0 | a0 | b1 | g1 | r1 | a1 | ... | b7 | g7 | r7 | a7 |
+----+----+----+----+----+----+----+----+-----+----+----+----+----+

*/
__m256i AVX2::ColorConvert::Convert_bgra2vuya_8u(const __m256i& bgraX8) noexcept
{
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
	__m256i Y03 = _mm256_srai_epi32(addDotsY1, InternalColorConvert::Shift);
	__m256i Y47 = _mm256_srai_epi32(addDotsY2, InternalColorConvert::Shift);

	/* COMPUTE U COMPONENT */
	/* DOT operation for pixels 0 - 3:  B * coeff[0] + G * coeff[1]; R * coeff[2] + A * coeff[3] */
	__m256i dotU1 = _mm256_madd_epi16(bgra03, coeffU);
	/* DOT operation for pixels 4 - 7:  B * coeff[0] + G * coeff[1]; R * coeff[2] + A * coeff[3] */
	__m256i dotU2 = _mm256_madd_epi16(bgra47, coeffU);

	/* SUM between DOT elements */
	__m256i addDotsU1 = _mm256_add_epi32(dotU1, _mm256_slli_epi64(dotU1, 32));
	__m256i addDotsU2 = _mm256_add_epi32(dotU2, _mm256_slli_epi64(dotU2, 32));

	/* final shift for get Y value: normalize ariphmetic result */
	__m256i U03 = _mm256_srai_epi32(addDotsU1, InternalColorConvert::Shift);
	__m256i U47 = _mm256_srai_epi32(addDotsU2, InternalColorConvert::Shift);

	/* COMPUTE V COMPONENT */
	__m256i dotV1 = _mm256_madd_epi16(bgra03, coeffV);
	/* DOT operation for pixels 4 - 7:  B * coeff[0] + G * coeff[1]; R * coeff[2] + A * coeff[3] */
	__m256i dotV2 = _mm256_madd_epi16(bgra47, coeffV);

	/* SUM between DOT elements */
	__m256i addDotsV1 = _mm256_add_epi32(dotV1, _mm256_slli_epi64(dotV1, 32));
	__m256i addDotsV2 = _mm256_add_epi32(dotV2, _mm256_slli_epi64(dotV2, 32));

	/* final shift for get Y value: normalize ariphmetic result */
	__m256i V03 = _mm256_srai_epi32(addDotsV1, InternalColorConvert::Shift);
	__m256i V47 = _mm256_srai_epi32(addDotsV2, InternalColorConvert::Shift);

	/* final PACK to VUYA vector */
	/* compine all Y values to single vector 32 bits */
	const __m256i& permute_idx = _mm256_setr_epi32(1, 3, 5, 7, 0, 2, 4, 6);
	const __m256i& addUV = _mm256_setr_epi32(128, 128, 128, 128, 128, 128, 128, 128);
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

__m256i AVX2::ColorConvert::Convert_vuya2bgra_8u (const __m256i& vuyaX8) noexcept
{
	const __m256i& subVal = _mm256_setr_epi16(
		128, 128, 0, 0,
		128, 128, 0, 0,
		128, 128, 0, 0,
		128, 128, 0, 0
	);

	/* Coefficients for compute R (mask ALPHA channel by zero) */
	const __m256i& coeffR = _mm256_setr_epi16
	(
		InternalColorConvert::rV, InternalColorConvert::rU, InternalColorConvert::rY, 0,
		InternalColorConvert::rV, InternalColorConvert::rU, InternalColorConvert::rY, 0,
		InternalColorConvert::rV, InternalColorConvert::rU, InternalColorConvert::rY, 0,
		InternalColorConvert::rV, InternalColorConvert::rU, InternalColorConvert::rY, 0
	);

	/* Coefficients for compute G (mask ALPHA channel by zero) */
	const __m256i& coeffG = _mm256_setr_epi16
	(
		InternalColorConvert::gV, InternalColorConvert::gU, InternalColorConvert::gY, 0,
		InternalColorConvert::gV, InternalColorConvert::gU, InternalColorConvert::gY, 0,
		InternalColorConvert::gV, InternalColorConvert::gU, InternalColorConvert::gY, 0,
		InternalColorConvert::gV, InternalColorConvert::gU, InternalColorConvert::gY, 0
	);

	/* Coefficients for compute B (mask ALPHA channel by zero) */
	const __m256i& coeffB = _mm256_setr_epi16
	(
		InternalColorConvert::bV, InternalColorConvert::bU, InternalColorConvert::bY, 0,
		InternalColorConvert::bV, InternalColorConvert::bU, InternalColorConvert::bY, 0,
		InternalColorConvert::bV, InternalColorConvert::bU, InternalColorConvert::bY, 0,
		InternalColorConvert::bV, InternalColorConvert::bU, InternalColorConvert::bY, 0
	);

	const __m256i& maxVal = _mm256_setr_epi16(
		0, 0, 0, 0,
		0, 0, 0, 0,
		0, 0, 0, 0,
		0, 0, 0, 0
	);

	const __m256i& minVal = _mm256_setr_epi16(
		255, 255, 255, 255,
		255, 255, 255, 255,
		255, 255, 255, 255,
		255, 255, 255, 255
	);

	/* extract 4 low pixels from uint8_t to int16_t and subtract 128 from V and U channel  */
	__m256i vuya03 = _mm256_subs_epi16(_mm256_cvtepu8_epi16(_mm256_extracti128_si256(vuyaX8, 0)), subVal);
	/* extract 4 high pixels from uint8_t to int16_t and subtract 128 from V and U channel */
	__m256i vuya47 = _mm256_subs_epi16(_mm256_cvtepu8_epi16(_mm256_extracti128_si256(vuyaX8, 1)), subVal);

	/* DOT operation for pixels 0 - 3:  V * coeff[0] + U * coeff[1]; Y * coeff[2] + A * coeff[3] */
	__m256i dotR03 = _mm256_madd_epi16(vuya03, coeffR);
	__m256i dotG03 = _mm256_madd_epi16(vuya03, coeffG);
	__m256i dotB03 = _mm256_madd_epi16(vuya03, coeffB);
	__m256i dorR47 = _mm256_madd_epi16(vuya47, coeffR);
	__m256i dorG47 = _mm256_madd_epi16(vuya47, coeffG);
	__m256i dorB47 = _mm256_madd_epi16(vuya47, coeffB);

	/* SUM between DOT elements */
	__m256i addDotsR1 = _mm256_add_epi32(dotR03, _mm256_slli_epi64(dotR03, 32));
	__m256i addDotsG1 = _mm256_add_epi32(dotG03, _mm256_slli_epi64(dotG03, 32));
	__m256i addDotsB1 = _mm256_add_epi32(dotB03, _mm256_slli_epi64(dotB03, 32));
	__m256i addDotsR2 = _mm256_add_epi32(dorR47, _mm256_slli_epi64(dorR47, 32));
	__m256i addDotsG2 = _mm256_add_epi32(dorG47, _mm256_slli_epi64(dorG47, 32));
	__m256i addDotsB2 = _mm256_add_epi32(dorB47, _mm256_slli_epi64(dorB47, 32));

	/* final shift for get R value: normalize ariphmetic result */
	__m256i R03 = _mm256_srai_epi32(addDotsR1, InternalColorConvert::Shift);
	__m256i R47 = _mm256_srai_epi32(addDotsR2, InternalColorConvert::Shift);
	__m256i G03 = _mm256_srai_epi32(addDotsG1, InternalColorConvert::Shift);
	__m256i G47 = _mm256_srai_epi32(addDotsG2, InternalColorConvert::Shift);
	__m256i B03 = _mm256_srai_epi32(addDotsB1, InternalColorConvert::Shift);
	__m256i B47 = _mm256_srai_epi32(addDotsB2, InternalColorConvert::Shift);

	R03 = _mm256_min_epi16(minVal, _mm256_max_epi16(R03, maxVal));
	G03 = _mm256_min_epi16(minVal, _mm256_max_epi16(G03, maxVal));
	B03 = _mm256_min_epi16(minVal, _mm256_max_epi16(B03, maxVal));
	R47 = _mm256_min_epi16(minVal, _mm256_max_epi16(R47, maxVal));
	G47 = _mm256_min_epi16(minVal, _mm256_max_epi16(G47, maxVal));
	B47 = _mm256_min_epi16(minVal, _mm256_max_epi16(B47, maxVal));

	const __m256i& permute_idx = _mm256_setr_epi32(1, 3, 5, 7, 0, 2, 4, 6);
	__m256i B = _mm256_permute2x128_si256(_mm256_permutevar8x32_epi32(B03, permute_idx), _mm256_permutevar8x32_epi32(B47, permute_idx), 0x20);
	__m256i G = _mm256_permute2x128_si256(_mm256_permutevar8x32_epi32(G03, permute_idx), _mm256_permutevar8x32_epi32(G47, permute_idx), 0x20);
	__m256i R = _mm256_permute2x128_si256(_mm256_permutevar8x32_epi32(R03, permute_idx), _mm256_permutevar8x32_epi32(R47, permute_idx), 0x20);

	R = _mm256_shuffle_epi8(R,
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

	G = _mm256_shuffle_epi8(G,
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

	__m256i BG = _mm256_or_si256(B, G);
	__m256i BGR = _mm256_or_si256(BG, R);
	__m256i BGRA = _mm256_blendv_epi8(vuyaX8, BGR,
		_mm256_setr_epi32(0xFFFFFF, 0xFFFFFF, 0xFFFFFF, 0xFFFFFF, 0xFFFFFF, 0xFFFFFF, 0xFFFFFF, 0xFFFFFF)
	);

	return BGRA;
}