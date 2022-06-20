#pragma once
/*
	https://stackoverflow.com/questions/51778721/how-to-convert-32-bit-float-to-8-bit-signed-char-41-packing-of-int32-to-int8
	https://stackoverflow.com/questions/30853773/sse-avx-conversion-from-double-to-char
*/
#include <immintrin.h>

// loads 128 bytes = 32 floats
// converts and packs with signed saturation to 32 int8_t
__m256i pack_float_int8(const float*p) {
	__m256i a = _mm256_cvtps_epi32(_mm256_loadu_ps(p));
	__m256i b = _mm256_cvtps_epi32(_mm256_loadu_ps(p + 8));
	__m256i c = _mm256_cvtps_epi32(_mm256_loadu_ps(p + 16));
	__m256i d = _mm256_cvtps_epi32(_mm256_loadu_ps(p + 24));
	__m256i ab = _mm256_packs_epi32(a, b);        // 16x int16_t
	__m256i cd = _mm256_packs_epi32(c, d);
	__m256i abcd = _mm256_packs_epi16(ab, cd);   // 32x int8_t
												 // packed to one vector, but in [ a_lo, b_lo, c_lo, d_lo | a_hi, b_hi, c_hi, d_hi ] order
												 // if you can deal with that in-memory format (e.g. for later in-lane unpack), great, you're done

												 // but if you need sequential order, then vpermd:
	__m256i lanefix = _mm256_permutevar8x32_epi32(abcd, _mm256_setr_epi32(0, 4, 1, 5, 2, 6, 3, 7));
	return lanefix;
}

/*
If you want to convert e.g. 16 doubles to 16 chars per iteration using AVX/SSE then here is some code that works:
*/
__m128i proc(const __m256d in0, const __m256d in1, const __m256d in2, const __m256d in3)
{
	__m128i v0 = _mm256_cvtpd_epi32(in0);
	__m128i v1 = _mm256_cvtpd_epi32(in1);
	__m128i v2 = _mm256_cvtpd_epi32(in2);
	__m128i v3 = _mm256_cvtpd_epi32(in3);
	__m128i v01 = _mm_packs_epi32(v0, v1);
	__m128i v23 = _mm_packs_epi32(v2, v3);
	return _mm_packs_epi16(v01, v23);
}
