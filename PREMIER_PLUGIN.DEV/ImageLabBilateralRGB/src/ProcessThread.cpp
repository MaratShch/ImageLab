#include "ImageLabBilateral.h"

extern std::mutex globalMutex;


AVX2_ALIGN static double gMesh[11][11] = { 0 };

void gaussian_weights(const double sigma, const int radius /* radius size in range of 3 to 10 */)
{
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

			const int jDiff = jMax - jMin;
			const int iDiff = iMax - iMin;
			// copy window of pixels to temporal array
			for (k = 0; k < jDiff; k++)
			{
				const int jIdx  = (jMin + k) * CIELabLinePitch + iMin;
				const int iSize = iDiff * CIELabBufferbands;
				memcpy(pI[k], &pCIELab[jIdx], iSize);
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


void* allocCIELabBuffer(const size_t& size)
{
	void* pMem = _aligned_malloc(size, CIELabBufferAlign);
	if (nullptr != pMem)
	{
		// for DBG purprose
		ZeroMemory(pMem, CIELabBufferAlign);
	}
	return pMem;
}

void freeCIELabBuffer(void* pMem)
{
	if (nullptr != pMem)
	{
		// for DBG purprose
		ZeroMemory(pMem, CIELabBufferAlign);
		_aligned_free(pMem);
		pMem = nullptr;
	}
}


void waitForJob(AsyncQueue* pAsyncJob)
{
	std::unique_lock<std::mutex> lk(globalMutex);
	pAsyncJob->cv.wait(lk, [&] {return pAsyncJob->bNewJob;});
	pAsyncJob->bNewJob = false;
	lk.unlock();
}


DWORD WINAPI ProcessThread(LPVOID pParam)
{
	DWORD exitCode = EXIT_SUCCESS;
	double* pBuffer1 = nullptr;
	double* pBuffer2 = nullptr;

	AsyncQueue* pAsyncJob = reinterpret_cast<AsyncQueue*>(pParam);

	// verify parameters
	if (nullptr == pAsyncJob)
		return EXIT_FAILURE;
	if (sizeof(*pAsyncJob) != pAsyncJob->strSizeOf)
		return EXIT_FAILURE;

	const DWORD affinity = 1UL << pAsyncJob->idx;
	SetThreadAffinityMask(::GetCurrentThread(), affinity);


	__try {

		// allocate memory buffers for temporary procssing
		pBuffer1 = reinterpret_cast<double*>(allocCIELabBuffer(CIELabBufferSize));
		pBuffer2 = reinterpret_cast<double*>(allocCIELabBuffer(CIELabBufferSize));

		if (nullptr == pBuffer1 || nullptr == pBuffer2)
			__leave;

		const size_t bufferSizeInPixels = CIELabBufferSize / (CIELabBufferbands * sizeof(double));
			
		// thread main loop
		while (true)
		{
//			printf("Thread %u wait for new job\n", pAsyncJob->idx);

			// waits for start new job
			waitForJob(pAsyncJob);

			// test exit' flag
			if (pAsyncJob->mustExit == true)
				__leave;

//			printf("Thread %u execute job\n", pAsyncJob->idx);

			// get job's
			int numJobs = 0;
			int idxHead = pAsyncJob->head;
			int idxTail = pAsyncJob->tail;

			// perform job
			if (-1 == idxTail)
				numJobs = idxHead + 1;
			else
				numJobs = (idxHead > idxTail) ?
					idxHead - idxTail + 1 : jobsQueueSize - idxTail + idxHead + 1;

			// perform job on specific data slice
			for (int i = 0; i < numJobs; i++)
			{
				csSDK_uint32* pSrcBGRA = pAsyncJob->jobsQueue[idxTail].pSrcSlice;
				csSDK_uint32* pDstBGRA = pAsyncJob->jobsQueue[idxTail].pDstSlice;
				const int sizeX = pAsyncJob->jobsQueue[idxTail].sizeX;
				const int sizeY = pAsyncJob->jobsQueue[idxTail].sizeY;
				const int rowBytes = pAsyncJob->jobsQueue[idxTail].rowWidth;

				BGRA_convert_to_CIELab(
					pSrcBGRA,   /* format B, G, R, A (each band as unsigned char) */
					pBuffer1,	/* format: L, a, b (each band as double) */
					sizeX,
					sizeY,
					rowBytes
				);

				idxTail++;
				if (idxTail >= jobsQueueSize)
					idxTail = 0;

				pAsyncJob->tail = idxTail;
			}

			// report to consumer about job completion
			pAsyncJob->bJobComplete;
			pAsyncJob->cv.notify_all();
		}// while(true)

	} // __try

	// cleanup memory resources on exit
	__finally {
		if (nullptr != pBuffer1)
		{
			freeCIELabBuffer(pBuffer1);
			pBuffer1 = nullptr;
		}
		if (nullptr != pBuffer2)
		{
			freeCIELabBuffer(pBuffer2);
			pBuffer2 = nullptr;
		}

		// report to consumer about job completion
		pAsyncJob->bJobComplete;
		pAsyncJob->cv.notify_all();
	}

	return exitCode;
}

