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


void rgbp2planar
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
    const __m256 v_zero = _mm256_setzero_ps();
    const __m256 v_255 = _mm256_set1_ps(255.0f);

    for (A_long j = 0; j < sizeY; j++)
    {
        // Assuming linePitch is in pixel units based on your original snippet
        const PF_Pixel_BGRA_8u* pLine = pSrc + j * linePitch;
        A_long i = 0;

        // --- AVX2 FAST PATH ---
        for (; i < spanX8; i += 8)
        {
            // Load 8 pixels (32 bytes) directly into an integer register
            __m256i v_bgra = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(&pLine[i]));

            // Extract B, G, R, and Alpha
            __m256i v_b_int = _mm256_and_si256(v_bgra, mask_FF);
            __m256i v_g_int = _mm256_and_si256(_mm256_srli_epi32(v_bgra, 8), mask_FF);
            __m256i v_r_int = _mm256_and_si256(_mm256_srli_epi32(v_bgra, 16), mask_FF);
            __m256i v_a_int = _mm256_srli_epi32(v_bgra, 24); // Alpha is at the top byte

            // Convert to 32-bit floats
            __m256 v_b_f = _mm256_cvtepi32_ps(v_b_int);
            __m256 v_g_f = _mm256_cvtepi32_ps(v_g_int);
            __m256 v_r_f = _mm256_cvtepi32_ps(v_r_int);
            __m256 v_a_f = _mm256_cvtepi32_ps(v_a_int);

            // Create a mask to protect against Division by Zero (where Alpha > 0)
            __m256 mask_a_gt_0 = _mm256_cmp_ps(v_a_f, v_zero, _CMP_GT_OQ);

            // Calculate un-premultiply factor: 255.0 / Alpha
            __m256 v_factor = _mm256_div_ps(v_255, v_a_f);

            // Apply mask: if Alpha was 0, force the factor to 0.0f instead of Infinity
            v_factor = _mm256_and_ps(v_factor, mask_a_gt_0);

            // Un-premultiply and safely clamp to 255.0f (handles illegal premul artifacts)
            v_r_f = _mm256_min_ps(_mm256_mul_ps(v_r_f, v_factor), v_255);
            v_g_f = _mm256_min_ps(_mm256_mul_ps(v_g_f, v_factor), v_255);
            v_b_f = _mm256_min_ps(_mm256_mul_ps(v_b_f, v_factor), v_255);

            // Calculate linear memory index and store to planar buffers
            const A_long idx = j * sizeX + i;
            _mm256_storeu_ps(&pR[idx], v_r_f);
            _mm256_storeu_ps(&pG[idx], v_g_f);
            _mm256_storeu_ps(&pB[idx], v_b_f);
        }

        // --- SCALAR TAIL ---
        for (; i < sizeX; i++)
        {
            const A_long idx = j * sizeX + i;
            A_u_char a = pLine[i].A;

            if (a > 0)
            {
                float factor = 255.0f / static_cast<float>(a);
                float r = static_cast<float>(pLine[i].R) * factor;
                float g = static_cast<float>(pLine[i].G) * factor;
                float b = static_cast<float>(pLine[i].B) * factor;

                // Clamp to prevent blowout from illegal source pixels
                pR[idx] = r > 255.0f ? 255.0f : r;
                pG[idx] = g > 255.0f ? 255.0f : g;
                pB[idx] = b > 255.0f ? 255.0f : b;
            }
            else
            {
                // If completely transparent, color data is meaningless, snap to 0
                pR[idx] = 0.0f;
                pG[idx] = 0.0f;
                pB[idx] = 0.0f;
            }
        }
    }
    return;
}