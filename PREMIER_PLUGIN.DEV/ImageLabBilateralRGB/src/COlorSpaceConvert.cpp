#include <math.h>
#include "ImageLabBilateral.h"

AVX2_ALIGN static float pTable[256];

void CreateColorConvertTable(void)
{
	int i;

	// create table of coeffcients for rapid convert from RGB to CIELab color space
	__VECTOR_ALIGNED__
		for (i = 0; i < 256; i++)
			pTable[i] = pow (static_cast<float>(i) / 255.0f, 2.19921875f);
}


void DeleteColorConvertTable(void)
{
	// nothing to do...
}


void BGRA_convert_to_CIELab(const csSDK_uint32* __restrict pBGRA,   /* format B, G, R, A (each band as unsigned char) */
							      float*		__restrict pCEILab, /* format: L, a, b (each band as float) */
							const int                      sizeX,
							const int                      sizeY,
							const int                      rowBytes)
{
	int i, j, k;
	float x, y, z;
	float x1, y1, z1;

	csSDK_uint32* __restrict pSrc = const_cast<csSDK_uint32* __restrict>(pBGRA);
	if (nullptr == pSrc || nullptr == pCEILab)
		return;

	const size_t numPixels = static_cast<size_t>(sizeX * sizeY);
	if (CIELabBufferPixSize < numPixels)
		return;

	const int linePitch = rowBytes >> 2;

	for (k = 0; k < sizeY; k++)
	{
			__VECTOR_ALIGNED__
			for (i = 0; i < sizeX; i++)
			{
				const csSDK_uint32 BGRAPixel = *pSrc;
				pSrc++;

				const unsigned int r = (BGRAPixel >> 16) & 0x000000FFu;
				const unsigned int g = (BGRAPixel >> 8) & 0x000000FFu;
				const unsigned int b = BGRAPixel         & 0x000000FFu;

				const float tR = pTable[r];
				const float tG = pTable[g];
				const float tB = pTable[b];

				x = (tR * 0.60672089f) + (tG * 0.19521921f) + (tB * 0.19799678f);
				y = (tR * 0.29738000f) + (tG * 0.62735000f) + (tB * 0.07552700f);
				z = (tR * 0.02482481f) + (tG * 0.06492290f) + (tB * 0.91024310f);

				x1 = (x > 0.0088560f) ? acbrt(x) : 7.7870f * x + 0.1379310f;
				y1 = (y > 0.0088560f) ? acbrt(y) : 7.7870f * y + 0.1379310f;
				z1 = (z > 0.0088560f) ? acbrt(z) : 7.7870f * z + 0.1379310f;

				*pCEILab++ = 116.0f * y1 - 16.0f;
				*pCEILab++ = 500.0f * (x1 - y1);
				*pCEILab++ = 200.0f * (y1 - z1);

			}
		pSrc += linePitch - sizeX;
	}

	return;
}


void CIELab_convert_to_BGRA(const float*        __restrict pCIELab,
							const unsigned int* __restrict pSrcBGRA, /* original image required only for take data from alpha channel */
							unsigned int*		__restrict pDstBGRA,
							const int                      sizeX,
							const int                      sizeY,
							const int                      rowBytes)
{
	float x1, y1, z1;
	float rr, gg, bb;
	float r1, g1, b1;

	unsigned int iR, iG, iB;
	int i, k;

	csSDK_uint32* pSrc = const_cast<csSDK_uint32*>(pSrcBGRA);
	csSDK_uint32* pDst = const_cast<csSDK_uint32*>(pDstBGRA);
	if (nullptr == pSrc || nullptr == pDst || nullptr == pCIELab)
		return;

	const int linePitch = rowBytes >> 2;

	for (k = 0; k < sizeY; k++)
	{
			__VECTOR_ALIGNED__
			for (i = 0; i < sizeX; i++)
			{
				const float L = *pCIELab++;
				const float a = *pCIELab++;
				const float b = *pCIELab++;

				const float y = (L + 16.0f) / 116.0f;
				const float x = a / 500.0f + y;
				const float z = y - b / 200.0f;

				x1 = (x > 0.2068930f) ? x * x * x : (x - 0.1379310f) / 7.7870f;
				y1 = (y > 0.2068930f) ? y * y * y : (y - 0.1379310f) / 7.7870f;
				z1 = (z > 0.2068930f) ? z * z * z : (z - 0.1379310f) / 7.7870f;

				x1 *= 0.950470f;
				z1 *= 1.088830f;

				rr = (x1 *  2.041370f) + (y1 * -0.564950f) + (z1 * -0.344690f);
				gg = (x1 * -0.962700f) + (y1 *  1.876010f) + (z1 *  0.041560f);
				bb = (x1 *  0.013450f) + (y1 * -0.118390f) + (z1 *  1.015410f);

				r1 = aExp(0.4547070f * aLog(rr));
				g1 = aExp(0.4547070f * aLog(gg));
				b1 = aExp(0.4547070f * aLog(bb));

				iR = static_cast<unsigned int>(r1 * 255.0f);
				iG = static_cast<unsigned int>(g1 * 255.0f);
				iB = static_cast<unsigned int>(b1 * 255.0f);

				const csSDK_uint32 pSrcPixel = *pSrc++;
				const csSDK_uint32 pDstPixel =
					pSrcPixel & 0xFF000000u			|
							(iB & 0x000000FFu)		|
							(iG & 0x000000FFu) << 8 |
							(iR & 0x000000FFu) << 16;
				
				*pDst++ = pDstPixel;

			}

		pSrc += linePitch - sizeX;
		pDst += linePitch - sizeX;

	}

	return;
}
