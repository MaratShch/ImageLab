#include <cstdint>
#include <immintrin.h>
#include "CommonPixFormat.hpp"

namespace AVX2
{
	namespace ColorConvert
	{
		void BGRA8u_to_VUYA8u
		(
			const PF_Pixel_BGRA_8u* __restrict pSrcImage,
			PF_Pixel_VUYA_8u* __restrict pDstImage,
			A_long sizeX,
			A_long sizeY,
			A_long linePitch
		) noexcept;

		void VUYA8u_to_BGRA8u
		(
			const PF_Pixel_VUYA_8u* __restrict pSrcImage,
			PF_Pixel_BGRA_8u* __restrict pDstImage,
			A_long sizeX,
			A_long sizeY,
			A_long linePitch
		) noexcept;

		static constexpr size_t Avx2BitsSize = 256;
		static constexpr size_t Avx2BytesSize = Avx2BitsSize / 8;

		namespace InternalColorConvert
		{
			static constexpr int32_t Shift = 13;
			static constexpr int32_t Mult = 1 << Shift;

			static_assert(Shift <= 14 && Shift > 0, "Invalid shift value");

			/* matrix coefficients for convert RGB to YUV */
			static constexpr int32_t yR = 0.299000f * Mult;
			static constexpr int32_t yG = 0.587000f * Mult;
			static constexpr int32_t yB = 0.114000f * Mult;
			static constexpr int32_t uR = -0.168736f * Mult;
			static constexpr int32_t uG = -0.331264f * Mult;
			static constexpr int32_t uB = 0.500000f * Mult;
			static constexpr int32_t vR = 0.500000f * Mult;
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

        static constexpr int32_t Shift = 13;
        const __m256i permute_idx_32 = _mm256_setr_epi32(1, 3, 5, 7, 0, 2, 4, 6);
        const __m256i rounding_bias = _mm256_set1_epi32(1 << 12);
        const __m256i uv_offset_128 = _mm256_set1_epi16(128);
        const __m256i alpha_blend_mask = _mm256_set1_epi32(static_cast<int>(0xFF000000));

        // VUYA Reconstruction: V=Byte0, U=Byte1, Y=Byte2
        const __m256i mask_reconstruct_VUYA = _mm256_setr_epi8(
            0, -1, -1, -1, 4, -1, -1, -1, 8, -1, -1, -1, 12, -1, -1, -1,
            0, -1, -1, -1, 4, -1, -1, -1, 8, -1, -1, -1, 12, -1, -1, -1
        );

        // BGRA Reconstruction: B=Byte0, G=Byte1, R=Byte2 (Derived from Packus output)
        const __m256i mask_reconstruct_BGRA = _mm256_setr_epi8(
            0, 12, 8, 4, 1, 13, 9, 5, 2, 14, 10, 6, 3, 15, 11, 7,
            0, 12, 8, 4, 1, 13, 9, 5, 2, 14, 10, 6, 3, 15, 11, 7
        );

        // BGRA Input -> Coefficients layout: [B, G, R, 0]
        // Y: B*0.114, G*0.587, R*0.299
        static const __m256i coeffY = _mm256_setr_epi16(934, 4809, 2449, 0, 934, 4809, 2449, 0, 934, 4809, 2449, 0, 934, 4809, 2449, 0);

        // U: B*0.500, G*-0.331, R*-0.169
        static const __m256i coeffU = _mm256_setr_epi16(4096, -2714, -1382, 0, 4096, -2714, -1382, 0, 4096, -2714, -1382, 0, 4096, -2714, -1382, 0);

        // V: B*-0.081, G*-0.419, R*0.500
        static const __m256i coeffV = _mm256_setr_epi16(-666, -3430, 4096, 0, -666, -3430, 4096, 0, -666, -3430, 4096, 0, -666, -3430, 4096, 0);

        // VUYA Input -> Coefficients layout: [V, U, Y, 0]
        // R: V*1.4075 + Y*1.0
        static const __m256i coeffR = _mm256_setr_epi16(11530, 0, 8192, 0, 11530, 0, 8192, 0, 11530, 0, 8192, 0, 11530, 0, 8192, 0);

        // G: V*-0.7169 + U*-0.3455 + Y*1.0
        static const __m256i coeffG = _mm256_setr_epi16(-5873, -2830, 8192, 0, -5873, -2830, 8192, 0, -5873, -2830, 8192, 0, -5873, -2830, 8192, 0);

        // B: U*1.7790 + Y*1.0
        static const __m256i coeffB = _mm256_setr_epi16(0, 14574, 8192, 0, 0, 14574, 8192, 0, 0, 14574, 8192, 0, 0, 14574, 8192, 0);

/*
        bgraX8 contains packet 8u BGRA pixels in folowing format :

            +---- + ---- + ---- + ---- + ---- + ---- + ---- + ---- + ---- - +---- + ---- + ---- + ---- +
            | b0 | g0 | r0 | a0 | b1 | g1 | r1 | a1 | ... | b7 | g7 | r7 | a7 |
            +---- + ---- + ---- + ---- + ---- + ---- + ---- + ---- + ---- - +---- + ---- + ---- + ---- +
*/
        inline __m256i Convert_bgra2vuya_8u(const __m256i& bgraX8) noexcept
        {
            // 1. Expand uint8 -> int16
            __m256i bgra03 = _mm256_cvtepu8_epi16(_mm256_castsi256_si128(bgraX8));
            __m256i bgra47 = _mm256_cvtepu8_epi16(_mm256_extracti128_si256(bgraX8, 1));

            // 2. Dot Product
            __m256i y1 = _mm256_madd_epi16(bgra03, coeffY);
            __m256i y2 = _mm256_madd_epi16(bgra47, coeffY);

            __m256i u1 = _mm256_madd_epi16(bgra03, coeffU);
            __m256i u2 = _mm256_madd_epi16(bgra47, coeffU);

            __m256i v1 = _mm256_madd_epi16(bgra03, coeffV);
            __m256i v2 = _mm256_madd_epi16(bgra47, coeffV);

            // 3. Horizontal Sum + Rounding Bias
            __m256i y1_sum = _mm256_add_epi32(_mm256_add_epi32(y1, _mm256_slli_epi64(y1, 32)), rounding_bias);
            __m256i y2_sum = _mm256_add_epi32(_mm256_add_epi32(y2, _mm256_slli_epi64(y2, 32)), rounding_bias);

            __m256i u1_sum = _mm256_add_epi32(_mm256_add_epi32(u1, _mm256_slli_epi64(u1, 32)), rounding_bias);
            __m256i u2_sum = _mm256_add_epi32(_mm256_add_epi32(u2, _mm256_slli_epi64(u2, 32)), rounding_bias);

            __m256i v1_sum = _mm256_add_epi32(_mm256_add_epi32(v1, _mm256_slli_epi64(v1, 32)), rounding_bias);
            __m256i v2_sum = _mm256_add_epi32(_mm256_add_epi32(v2, _mm256_slli_epi64(v2, 32)), rounding_bias);

            // 4. Shift and Permute to Indexes
            __m256i Y = _mm256_permute2x128_si256(
                _mm256_permutevar8x32_epi32(_mm256_srai_epi32(y1_sum, Shift), permute_idx_32),
                _mm256_permutevar8x32_epi32(_mm256_srai_epi32(y2_sum, Shift), permute_idx_32), 0x20);

            __m256i U = _mm256_permute2x128_si256(
                _mm256_permutevar8x32_epi32(_mm256_srai_epi32(u1_sum, Shift), permute_idx_32),
                _mm256_permutevar8x32_epi32(_mm256_srai_epi32(u2_sum, Shift), permute_idx_32), 0x20);

            __m256i V = _mm256_permute2x128_si256(
                _mm256_permutevar8x32_epi32(_mm256_srai_epi32(v1_sum, Shift), permute_idx_32),
                _mm256_permutevar8x32_epi32(_mm256_srai_epi32(v2_sum, Shift), permute_idx_32), 0x20);

            // 5. Add 128 to U/V
            U = _mm256_add_epi32(U, _mm256_set1_epi32(128));
            V = _mm256_add_epi32(V, _mm256_set1_epi32(128));

            // 6. Pack to Byte Slots
            __m256i v_pos = _mm256_shuffle_epi8(V, mask_reconstruct_VUYA);
            __m256i u_pos = _mm256_slli_epi32(_mm256_shuffle_epi8(U, mask_reconstruct_VUYA), 8);
            __m256i y_pos = _mm256_slli_epi32(_mm256_shuffle_epi8(Y, mask_reconstruct_VUYA), 16);

            __m256i result = _mm256_or_si256(v_pos, _mm256_or_si256(u_pos, y_pos));

            // 7. Blend Alpha
            __m256i alpha = _mm256_and_si256(bgraX8, alpha_blend_mask);
            result = _mm256_or_si256(result, alpha);

            return result;
        }

        // =========================================================================
        // VUYA -> BGRA
        // =========================================================================
        inline __m256i Convert_vuya2bgra_8u(const __m256i& vuyaX8) noexcept
        {
            // 1. Expand and Subtract 128 from U/V (bytes 0 and 1)
            __m256i vuya03 = _mm256_subs_epi16(_mm256_cvtepu8_epi16(_mm256_castsi256_si128(vuyaX8)), uv_offset_128);
            __m256i vuya47 = _mm256_subs_epi16(_mm256_cvtepu8_epi16(_mm256_extracti128_si256(vuyaX8, 1)), uv_offset_128);

            // 2. Dot Product
            __m256i r03 = _mm256_madd_epi16(vuya03, coeffR);
            __m256i g03 = _mm256_madd_epi16(vuya03, coeffG);
            __m256i b03 = _mm256_madd_epi16(vuya03, coeffB);

            __m256i r47 = _mm256_madd_epi16(vuya47, coeffR);
            __m256i g47 = _mm256_madd_epi16(vuya47, coeffG);
            __m256i b47 = _mm256_madd_epi16(vuya47, coeffB);

            // 3. Sum + Bias + Shift
            __m256i r03_sum = _mm256_srai_epi32(_mm256_add_epi32(_mm256_add_epi32(r03, _mm256_slli_epi64(r03, 32)), rounding_bias), Shift);
            __m256i g03_sum = _mm256_srai_epi32(_mm256_add_epi32(_mm256_add_epi32(g03, _mm256_slli_epi64(g03, 32)), rounding_bias), Shift);
            __m256i b03_sum = _mm256_srai_epi32(_mm256_add_epi32(_mm256_add_epi32(b03, _mm256_slli_epi64(b03, 32)), rounding_bias), Shift);

            __m256i r47_sum = _mm256_srai_epi32(_mm256_add_epi32(_mm256_add_epi32(r47, _mm256_slli_epi64(r47, 32)), rounding_bias), Shift);
            __m256i g47_sum = _mm256_srai_epi32(_mm256_add_epi32(_mm256_add_epi32(g47, _mm256_slli_epi64(g47, 32)), rounding_bias), Shift);
            __m256i b47_sum = _mm256_srai_epi32(_mm256_add_epi32(_mm256_add_epi32(b47, _mm256_slli_epi64(b47, 32)), rounding_bias), Shift);

            // 4. Permute to Linear
            __m256i R = _mm256_permute2x128_si256(
                _mm256_permutevar8x32_epi32(r03_sum, permute_idx_32),
                _mm256_permutevar8x32_epi32(r47_sum, permute_idx_32), 0x20);

            __m256i G = _mm256_permute2x128_si256(
                _mm256_permutevar8x32_epi32(g03_sum, permute_idx_32),
                _mm256_permutevar8x32_epi32(g47_sum, permute_idx_32), 0x20);

            __m256i B = _mm256_permute2x128_si256(
                _mm256_permutevar8x32_epi32(b03_sum, permute_idx_32),
                _mm256_permutevar8x32_epi32(b47_sum, permute_idx_32), 0x20);

            // 5. Pack (with Saturation)
            __m256i R_G_16 = _mm256_packus_epi32(R, G);

            __m256i A = _mm256_and_si256(vuyaX8, alpha_blend_mask);
            A = _mm256_srli_epi32(A, 24);

            __m256i B_A_16 = _mm256_packus_epi32(B, A);

            // 6. Final Pack to 8-bit [B0..B3 A0..A3 | R0..R3 G0..G3]
            __m256i packed = _mm256_packus_epi16(B_A_16, R_G_16);

            // 7. Shuffle to BGRA
            return _mm256_shuffle_epi8(packed, mask_reconstruct_BGRA);
        }

	}; /* namespace ColorConvert */

}; /* namespace AVX2 */
