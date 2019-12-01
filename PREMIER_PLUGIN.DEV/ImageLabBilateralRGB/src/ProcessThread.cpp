#include "ImageLabBilateral.h"

CACHE_ALIGN static float gMesh[11][11] = { 0 };

void gaussian_weights(const float sigma, const int radius /* radius size in range of 3 to 10 */)
{
	int i, j;
	int x, y;

	constexpr int size = 11;
	const float divider = 2.0f * (sigma * sigma); // 2 * sigma ^ 2

	__VECTOR_ALIGNED__
	for (y = -radius, j = 0; j < size; j++, y++)
	{
		for (x = -radius, i = 0; i < size; i++, x++)
		{
			const float dSum = static_cast<float>((x * x) + (y * y));
			gMesh[j][i] = aExp(-dSum / divider);
		}
	}

	return;
}


void bilateral_filter_color(const float* __restrict pCIELab,
							float* __restrict pFiltered,
	                        const int sizeX,
	                        const int sizeY,
							const int radius,
	                        const float sigmaR) /* value sigmaR * 100 */
{
#if !defined __INTEL_COMPILER 
	_MM_SET_FLUSH_ZERO_MODE(_MM_FLUSH_ZERO_ON);
	_MM_SET_DENORMALS_ZERO_MODE(_MM_DENORMALS_ZERO_ON);
#endif

	CACHE_ALIGN float pH[11][11] = {};
	CACHE_ALIGN float pF[11][11] = {};

	float bSum1;
	float bSum2;
	float bSum3;

	float dL, da, db;

	const float sigma = 100.00f * sigmaR;
	const float divider = 2.00f * sigma * sigma;

	int i, j, k, l, m;

	const int CIELabLinePitch = sizeX * CIELabBufferbands;
	const int lastPixelIdx = sizeX - 1;
	const int lastLineIdx  = sizeY - 1;

	for (j = 0; j < sizeY; j++)
	{
		for (i = 0; i < sizeX; i++)
		{
			int iMin, iMax, jMin, jMax;

			// define processing window coordinates
			iMin = MAX(i - radius, 0);
			iMax = MIN(i + radius, lastPixelIdx);
			jMin = MAX(j - radius, 0);
			jMax = MIN(j + radius, lastLineIdx);

			// define process window sizes
			const int jDiff = (jMax - jMin) + 1;
			const int iDiff = (iMax - iMin) + 1;

			// get processed pixel
			const int srcIdx = j * CIELabLinePitch + i * CIELabBufferbands;
			const float L = pCIELab[srcIdx];
			const float a = pCIELab[srcIdx + 1];
			const float b = pCIELab[srcIdx + 2];

   		    // compute Gaussian range weights
			for (k = 0; k < jDiff; k++)
			{
				const int jIdx = (jMin + k) * CIELabLinePitch + iMin * CIELabBufferbands;

				for (m = l = 0; l < iDiff; l++, m += 3)
				{
					dL = pCIELab[jIdx + m    ] - L;
					da = pCIELab[jIdx + m + 1] - a;
					db = pCIELab[jIdx + m + 2] - b;

					const float dotComp = (dL * dL) + (da * da) + (db * db);
					pH[k][l] = aExp(-dotComp / divider);
				}
			}

			// calculate bilateral filter responce
			float norm_F = 0.0;
			int jIdx, iIdx;

			jIdx = jMin - j + radius;

			for (k = 0; k < jDiff; k++)
			{
				iIdx = iMin - i + radius;
				__VECTOR_ALIGNED__
				for (l = 0; l < iDiff; l++)
				{
					pF[k][l] = pH[k][l] * gMesh[jIdx][iIdx];
					norm_F += pF[k][l];
					iIdx++;
				}
				jIdx++;
			}

			bSum1 = bSum2 = bSum3 = 0.0;
			for (k = 0; k < jDiff; k++)
			{
				const int kIdx = (jMin + k) * CIELabLinePitch + iMin * CIELabBufferbands;
				for (m = l = 0; l < iDiff; l++, m += 3)
				{
					bSum1 += (pF[k][l] * pCIELab[kIdx + m    ]);
					bSum2 += (pF[k][l] * pCIELab[kIdx + m + 1]);
					bSum3 += (pF[k][l] * pCIELab[kIdx + m + 2]);
				}
			}

			// compute destination pixels
			const int dstIdx = srcIdx;
			pFiltered[dstIdx]     = bSum1 / norm_F;
			pFiltered[dstIdx + 1] = bSum2 / norm_F;
			pFiltered[dstIdx + 2] = bSum3 / norm_F;

		} // for (i = 0; i < sizeX; i++)

	} // for (j = 0; j < sizeY; j++)

	return;
}


void* allocCIELabBuffer(const size_t& size)
{
	void* pMem = _aligned_malloc(size, CIELabBufferAlign);
#ifdef _DEBUG
	if (nullptr != pMem)
	{
		// for DBG purprose
		ZeroMemory(pMem, CIELabBufferAlign);
	}
#endif
	return pMem;
}

void freeCIELabBuffer(void* pMem)
{
	if (nullptr != pMem)
	{
#ifdef _DEBUG
		// for DBG purprose
		ZeroMemory(pMem, CIELabBufferAlign);
#endif
		_aligned_free(pMem);
		pMem = nullptr;
	}
}
