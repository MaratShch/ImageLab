#include "MosaicColorConvert.hpp"


void rgb2planar
(
	const PF_Pixel_BGRA_8u* RESTRICT pSrc,
	const MemHandler& memHndl,
	A_long sizeX,
	A_long sizeY,
	A_long linePitch
)
{
	float* RESTRICT pR = memHndl.R_planar;
	float* RESTRICT pG = memHndl.G_planar;
	float* RESTRICT pB = memHndl.B_planar;
 
	const A_long spanX8 = sizeX & ~7; // Snap to nearest multiple of 8
	const __m256i mask_FF = _mm256_set1_epi32(0xFF);

	for (A_long j = 0; j < sizeY; j++)
	{
		const PF_Pixel_BGRA_8u* pLine = pSrc + j * linePitch;
		A_long i = 0;
		
		for (; i < spanX8; i += 8)
		{
			// Load 8 pixels (32 bytes) directly into an integer register
			__m256i v_bgra = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(&pLine[i]));
			
			// Extract channels assuming little-endian physical memory layout: 
			// Byte 0 = B, Byte 1 = G, Byte 2 = R, Byte 3 = A.
			// (If your struct layout differs physically, just swap these shift values: 0, 8, 16)
			__m256i v_b_int = _mm256_and_si256(v_bgra, mask_FF);
			__m256i v_g_int = _mm256_and_si256(_mm256_srli_epi32(v_bgra, 8), mask_FF);
			__m256i v_r_int = _mm256_and_si256(_mm256_srli_epi32(v_bgra, 16), mask_FF);
			
			// Convert to 32-bit floats
			__m256 v_b_f = _mm256_cvtepi32_ps(v_b_int);
			__m256 v_g_f = _mm256_cvtepi32_ps(v_g_int);
			__m256 v_r_f = _mm256_cvtepi32_ps(v_r_int);
			
			// Calculate linear memory index and store to planar buffers
			const A_long idx = j * sizeX + i;
			_mm256_storeu_ps(&pR[idx], v_r_f);
			_mm256_storeu_ps(&pG[idx], v_g_f);
			_mm256_storeu_ps(&pB[idx], v_b_f);
		}
		
		for (; i < sizeX; i++)
		{
			const A_long idx = j * sizeX + i;
			pR[idx] = static_cast<float>(pLine[i].R);
			pG[idx] = static_cast<float>(pLine[i].G);
			pB[idx] = static_cast<float>(pLine[i].B);
		}
	}
    return;
}


void planar2rgb
(
    const PF_Pixel_BGRA_8u* RESTRICT pSrc,
    const MemHandler& memHndl,
    PF_Pixel_BGRA_8u* RESTRICT pDst,
    A_long sizeX,
    A_long sizeY,
    A_long srcPitch,
    A_long dstPitch
)
{
    const float* RESTRICT pR = memHndl.R_planar;
    const float* RESTRICT pG = memHndl.G_planar;
    const float* RESTRICT pB = memHndl.B_planar;

    const A_long spanX8 = sizeX & ~7;

    // Pre-calculate constants for AVX2 bounding
    const __m256 v_zero = _mm256_setzero_ps();
    const __m256 v_255 = _mm256_set1_ps(255.0f);

    // Mask to extract ONLY the top byte (Alpha channel) from a 32-bit integer
    const __m256i v_alpha_mask = _mm256_set1_epi32(0xFF000000);

    for (A_long j = 0; j < sizeY; j++)
    {
        const PF_Pixel_BGRA_8u* pSrcLine = pSrc + j * srcPitch;
        PF_Pixel_BGRA_8u* pOutLine = pDst + j * dstPitch;
        A_long i = 0;

        // --- AVX2 FAST PATH ---
        for (; i < spanX8; i += 8)
        {
            const A_long idx = j * sizeX + i;

            // 1. Load planar floats
            __m256 v_r_f = _mm256_loadu_ps(&pR[idx]);
            __m256 v_g_f = _mm256_loadu_ps(&pG[idx]);
            __m256 v_b_f = _mm256_loadu_ps(&pB[idx]);

            // 2. Clamp to [0.0f, 255.0f]
            v_r_f = _mm256_min_ps(_mm256_max_ps(v_r_f, v_zero), v_255);
            v_g_f = _mm256_min_ps(_mm256_max_ps(v_g_f, v_zero), v_255);
            v_b_f = _mm256_min_ps(_mm256_max_ps(v_b_f, v_zero), v_255);

            // 3. Convert to 32-bit integers
            __m256i v_r_i = _mm256_cvtps_epi32(v_r_f);
            __m256i v_g_i = _mm256_cvtps_epi32(v_g_f);
            __m256i v_b_i = _mm256_cvtps_epi32(v_b_f);

            // 4. Bit-shift integers into correct BGRA byte positions
            v_r_i = _mm256_slli_epi32(v_r_i, 16);
            v_g_i = _mm256_slli_epi32(v_g_i, 8);
            // v_b_i requires no shift (stays at bottom byte)

            // 5. Load 8 original pixels from pSrc and isolate ONLY the Alpha channel
            __m256i v_src_pixels = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(&pSrcLine[i]));
            __m256i v_src_alpha = _mm256_and_si256(v_src_pixels, v_alpha_mask);

            // 6. Pack R, G, B, and the source Alpha into a single 32-bit block via Bitwise OR
            __m256i v_bgra = _mm256_or_si256(v_b_i,
                _mm256_or_si256(v_g_i,
                    _mm256_or_si256(v_r_i, v_src_alpha)));

            // 7. Store exactly 8 packed structs (32 bytes) back to memory
            _mm256_storeu_si256(reinterpret_cast<__m256i*>(&pOutLine[i]), v_bgra);
        }

        for (; i < sizeX; i++)
        {
            const A_long idx = j * sizeX + i;

            float r = pR[idx];
            float g = pG[idx];
            float b = pB[idx];

            // Safe scalar clamping
            r = r < 0.0f ? 0.0f : (r > 255.0f ? 255.0f : r);
            g = g < 0.0f ? 0.0f : (g > 255.0f ? 255.0f : g);
            b = b < 0.0f ? 0.0f : (b > 255.0f ? 255.0f : b);

            pOutLine[i].R = static_cast<A_u_char>(r);
            pOutLine[i].G = static_cast<A_u_char>(g);
            pOutLine[i].B = static_cast<A_u_char>(b);

            // Grab Alpha directly from the source buffer
            pOutLine[i].A = pSrcLine[i].A;
        }
    }

    return;
}