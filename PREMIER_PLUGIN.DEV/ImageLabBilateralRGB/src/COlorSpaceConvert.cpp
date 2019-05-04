#include <windows.h>
#include <math.h>
#include "ImageLabBilateral.h"

static double* pTable = nullptr;

bool CreateColorConvertTable(void)
{
	int i;

	if (nullptr == pTable)
	{
		const size_t tblSize = sizeof(double) * 256;
		pTable = reinterpret_cast<double*>(_aligned_malloc(tblSize, 64));
	}

	if (nullptr != pTable)
	{
		CACHE_ALIGN double dTable[256];

#pragma vector aligned
		for (i = 0; i < 256; i++)
			dTable[i] = static_cast<double>(i) / 255.00;

#pragma vector aligned
		for (i = 0; i < 256; i++)
			pTable[i] = pow(dTable[i], 2.1992187500);
	}

	return (nullptr != pTable);
}


void DeleteColorConevrtTable(void)
{
	if (nullptr != pTable)
	{
		// for DBG purpose
		ZeroMemory(pTable, sizeof(double) * 256);
		_aligned_free(pTable);
		pTable = nullptr;
	}
	return;
}


void BGRA_convert_to_CIELab(const unsigned int* __restrict pBGRA,   /* format B, G, R, A (each band as unsigned char) */
							const double*       __restrict pTable,
							      double*		__restrict pCEILab, /* format: L, a, b (each band as double) */
							const int&                     sampNumber)
{
	int i, j;
	double x, y, z;
	double x1, y1, z1;

#pragma vector aligned
	for (j = i = 0; i < sampNumber; i++, j += 3)
	{
		const unsigned int r = (pBGRA[i] >> 8)  & 0x000000FFu;
		const unsigned int g = (pBGRA[i] >> 16) & 0x000000FFu;
		const unsigned int b = (pBGRA[i] >> 24) & 0x000000FFu;

		const double tR = pTable[r];
		const double tG = pTable[g];
		const double tB = pTable[b];

		x = tR * 0.606720890 + tG * 0.195219210 + tB * 0.197996780;
		y = tR * 0.297380000 + tG * 0.627350000 + tB * 0.075527000;
		z = tR * 0.024824810 + tG * 0.064922900 + tB * 0.910243100;

		x1 = (x > 0.0088560) ? cbrt(x) : 7.7870 * x + 0.1379310;
		y1 = (y > 0.0088560) ? cbrt(y) : 7.7870 * y + 0.1379310;
		z1 = (z > 0.0088560) ? cbrt(z) : 7.7870 * z + 0.1379310;

		pCEILab[j  ] = 116.0 * y1 - 16.0;
		pCEILab[j+1] = 500.0 * (x1 - y1);
		pCEILab[j+2] = 200.0 * (y1 - z1);
	}
}

void CIELab_conbert_to_RGB(	const double*       __restrict pCIELab,
							const unsigned int* __restrict pSrcBGRA, /* original image required only for copy data from alpha channel to processed buffer */
							unsigned int*		__restrict pDstBGRA,
							const int&                     sampNumber)
{
	double x1, y1, z1;
	double r, g, b;
	double r1, g1, b1;
	unsigned int iR, iG, iB;
	int i, j;

#pragma vector aligned
	for (i = j = 0; i < sampNumber; i++, j += 3)
	{
		const double y = (pCIELab[j] + 16.0) / 116.0;
		const double x = pCIELab [j+1] / 500.0 + y;
		const double z = y - pCIELab[j+2] / 200.0;

		x1 = (x > 0.2068930) ? x * x * x : (x - 0.1379310) / 7.7870;
		y1 = (y > 0.2068930) ? y * y * y : (y - 0.1379310) / 7.7870;
		z1 = (z > 0.2068930) ? z * z * z : (z - 0.1379310) / 7.7870;

		x1 *= 0.950470;
		z1 *= 1.088830;

		r = x1 *  2.041370 + y1 * -0.564950 + z1 * -0.344690;
		g = x1 * -0.962700 + y1 *  1.876010 + z1 *  0.041560;
		b = x1 *  0.013450 + y1 * -0.118390 + z1 *  1.015410;

		r1 = pow(r, 0.4547070);
		g1 = pow(g, 0.4547070);
		b1 = pow(b, 0.4547070);

		iR = static_cast<unsigned int>(r1 * 255.0);
		iG = static_cast<unsigned int>(g1 * 255.0);
		iB = static_cast<unsigned int>(b1 * 255.0);

		pDstBGRA[i] = pSrcBGRA[i] & 0x000000FFu  |
						(iR & 0x000000FFu) << 8  |
						(iG & 0x000000FFu) << 16 |
						(iB & 0x000000FFu) << 24;
	
	}

	return;
}
