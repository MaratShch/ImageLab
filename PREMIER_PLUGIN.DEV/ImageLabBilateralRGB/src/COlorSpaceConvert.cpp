#include <math.h>
#include "ImageLabBilateral.h"

AVX2_ALIGN static double pTable[256];

void CreateColorConvertTable(void)
{
	int i;

	// create table of coeffcients for rapid convert from RGB to CILELab color space
	__VECTOR_ALIGNED__
		for (i = 0; i < 256; i++)
			pTable[i] = pow (static_cast<double>(i) / 255.00, 2.1992187500);
}


void DeleteColorConvertTable(void)
{
	// nothing to do...
}


void BGRA_convert_to_CIELab(const csSDK_uint32* __restrict pBGRA,   /* format B, G, R, A (each band as unsigned char) */
							      double*		__restrict pCEILab, /* format: L, a, b (each band as double) */
							const int                      sizeX,
							const int                      sizeY,
							const int                      rowBytes)
{
	int i, j, k;
	double x, y, z;
	double x1, y1, z1;

	csSDK_uint32* __restrict pSrc = const_cast<csSDK_uint32* __restrict>(pBGRA);
	if (nullptr == pSrc || nullptr == pCEILab)
		return;

	const size_t numPixels = static_cast<size_t>(sizeX * sizeY);
	if (CIELabBufferPixSize < numPixels)
		return;

	const int linePitch = rowBytes >> 2;

//	j = 0;
	for (k = 0; k < sizeY; k++)
	{
//		__VECTOR_ALIGNED__
			for (i = 0; i < sizeX; i++)
			{
				const csSDK_uint32 BGRAPixel = *pSrc++;
				const unsigned int r = BGRAPixel         & 0x000000FFu;
				const unsigned int g = (BGRAPixel >> 8)  & 0x000000FFu;
				const unsigned int b = (BGRAPixel >> 16) & 0x000000FFu;

				const double tR = pTable[r];
				const double tG = pTable[g];
				const double tB = pTable[b];

				x = tR * 0.606720890 + tG * 0.195219210 + tB * 0.197996780;
				y = tR * 0.297380000 + tG * 0.627350000 + tB * 0.075527000;
				z = tR * 0.024824810 + tG * 0.064922900 + tB * 0.910243100;

				x1 = (x > 0.0088560) ? cbrt(x) : 7.7870 * x + 0.1379310;
				y1 = (y > 0.0088560) ? cbrt(y) : 7.7870 * y + 0.1379310;
				z1 = (z > 0.0088560) ? cbrt(z) : 7.7870 * z + 0.1379310;

//				pCEILab[j] = 116.0 * y1 - 16.0;
//				pCEILab[j + 1] = 500.0 * (x1 - y1);
//				pCEILab[j + 2] = 200.0 * (y1 - z1);
//				j += 3;
				*pCEILab++ = 116.0 * y1 - 16.0;
				*pCEILab++ = 500.0 * (x1 - y1);
				*pCEILab++ = 200.0 * (y1 - z1);

			}
		pSrc += linePitch - sizeX;
	}

	return;
}

void CIELab_convert_to_BGRA(const double*       __restrict pCIELab,
							const unsigned int* __restrict pSrcBGRA, /* original image required only for take data from alpha channel */
							unsigned int*		__restrict pDstBGRA,
							const int                      sizeX,
							const int                      sizeY,
							const int                      rowBytes)
{
	double x1 = 0.0, y1 = 0.0, z1 = 0.0;
	double rr = 0.0, gg = 0.0, bb = 0.0;
	double r1 = 0.0, g1 = 0.0, b1 = 0.0;

	unsigned int iR = 0u, iG = 0u, iB = 0u;
	int i = 0, j = 0, k = 0;

	csSDK_uint32* pSrc = const_cast<csSDK_uint32*>(pSrcBGRA);
	csSDK_uint32* pDst = const_cast<csSDK_uint32*>(pDstBGRA);
	if (nullptr == pSrc || nullptr == pDst || nullptr == pCIELab)
		return;

	const int linePitch = rowBytes >> 2;

	for (k = 0; k < sizeY; k++)
	{
//		__VECTOR_ALIGNED__
			for (i = 0; i < sizeX; i++)
			{
				const double L = *pCIELab++;
				const double a = *pCIELab++;
				const double b = *pCIELab++;

				const double y = (L + 16.0) / 116.0;
				const double x = a / 500.0 + y;
				const double z = y - b / 200.0;

				x1 = (x > 0.2068930) ? x * x * x : (x - 0.1379310) / 7.7870;
				y1 = (y > 0.2068930) ? y * y * y : (y - 0.1379310) / 7.7870;
				z1 = (z > 0.2068930) ? z * z * z : (z - 0.1379310) / 7.7870;

				x1 *= 0.950470;
				z1 *= 1.088830;

				rr = x1 *  2.041370 + y1 * -0.564950 + z1 * -0.344690;
				gg = x1 * -0.962700 + y1 *  1.876010 + z1 *  0.041560;
				bb = x1 *  0.013450 + y1 * -0.118390 + z1 *  1.015410;

				r1 = pow(rr, 0.4547070);
				g1 = pow(gg, 0.4547070);
				b1 = pow(bb, 0.4547070);

				iR = static_cast<unsigned int>(r1 * 255.0);
				iG = static_cast<unsigned int>(g1 * 255.0);
				iB = static_cast<unsigned int>(b1 * 255.0);

				const csSDK_uint32 pSrcPixel = *pSrc++;
				const csSDK_uint32 pDstPixel =
					pSrcPixel & 0xFF000000u			|
							(iR & 0x000000FFu)		|
							(iG & 0x000000FFu) << 8 |
							(iB & 0x000000FFu) << 16;
				
				*pDst++ = pDstPixel;

			}

		pSrc += linePitch - sizeX;
		pDst += linePitch - sizeX;

	}

	return;
}