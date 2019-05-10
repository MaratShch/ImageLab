#include "ImageLabBilateral.h"

AVX2_ALIGN static double gMesh[11][11] = { 0 };

void gaussian_weights(const double sigma, const int radius /* radius size in range of 3 to 10 */)
{
	double x1, y1;
	int i, j;
	int x, y;

	const int size = 11;
	const double divider = 2.0 * (sigma * sigma); // 2 * sigma ^ 2

	__VECTOR_ALIGNED__
	for (y = -radius, j = 0; j < size; j++, y++)
	{
		for (x = -radius, i = 0; i < size; i++, x++)
		{
			const double dSum = static_cast<double>(x * x + y * y);
			gMesh[j][i] = EXP(-dSum / divider);
		}
	}

	return;
}


void bilateral_filter_color(const double* __restrict pCIELab,
							double* __restrict pFiltered,
	                        const int sizeX,
	                        const int sizeY,
							const int radius,
	                        const double sigmaR) /* value sigmaR * 100 */
{
	AVX2_ALIGN double pI[6][6 * 3];
	AVX2_ALIGN double pH[6][6];
	AVX2_ALIGN double pF[6][6];

	double dL, da, db;
	const double divider = 2.0 * sigmaR * sigmaR;

	int i, j, k, l, m;
	int jk, il, idx;

	const int regionSize = 6;// radius + 1;
	const int CIELabLinePitch = sizeX * CIELabBufferbands;

	for (j = 0; j < sizeY; j++)
	{
		for (i = 0; i < sizeX; i++)
		{
			int iMin, iMax, jMin, jMax;

			// define processing window coordinates
			iMin = MAX(i - radius, 1);
			iMax = MIN(i + radius, sizeX);
			jMin = MAX(j - radius, 1);
			jMax = MIN(j + radius, sizeY);

			// copy window of pixels to temporal array
			for (k = 0; k < regionSize; k++)
			{
				memcpy(pI[k], &pCIELab[jMin*CIELabLinePitch + iMin], regionSize*CIELabBufferbands);
			}

			const int CIELabIdx = j * CIELabLinePitch + i;
			const double L = pCIELab[CIELabIdx];
			const double a = pCIELab[CIELabIdx + 1];
			const double b = pCIELab[CIELabIdx + 2];

			// compute Gaussian range weights
			__VECTOR_ALIGNED__
			for (k = 0; k < regionSize; k++)
			{
				for (m = l = 0; l < regionSize; l++, m += 3)
				{
					dL = pI[k][m]   - L;
					da = pI[k][m+1] - a;
					db = pI[k][m+2] - b;

					const double dotComp = dL * dL + da * da + db * db;
					pH[k][l] = EXP(-dotComp / divider);
				};
			}

			// calculate bilateral filter responce
			double norm_F = 0.0;
			int jIdx, iIdx;

			jIdx = jMin - j + radius;
			__VECTOR_ALIGNED__
			for (k = 0; k < regionSize; k++)
			{
				iIdx = iMin - i + radius;
				for (l = 0; l < regionSize; l++)
				{
					pF[k][l] = pH[k][l] * gMesh[jIdx][iIdx];
					norm_F += pF[k][l];
					iIdx++;
				}
				jIdx++;
			}

			// compute destination pixels
			double bSum1 = 0.0;
			double bSum2 = 0.0;
			double bSum3 = 0.0;

			__VECTOR_ALIGNED__
			for (k = 0; k < regionSize; k++)
			{
				for (m = l = 0; l < regionSize; l++, m += 3)
				{
					bSum1 += (pF[k][l] * pI[k][m]);
					bSum2 += (pF[k][l] * pI[k][m + 1]);
					bSum3 += (pF[k][l] * pI[k][m + 2]);
				}
			}

			pFiltered[i * 3]     = bSum1 / norm_F;
			pFiltered[i * 3 + 1] = bSum2 / norm_F;
			pFiltered[i * 3 + 2] = bSum3 / norm_F;
		}
	}

	return;
}

