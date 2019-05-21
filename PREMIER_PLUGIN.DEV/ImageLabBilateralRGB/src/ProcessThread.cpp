#include "ImageLabBilateral.h"

//extern std::mutex globalMutex;


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
	AVX2_ALIGN double pI[11][11 * 3] = { 0 };
	AVX2_ALIGN double pH[11][11] = { 0 };
	AVX2_ALIGN double pF[11][11] = { 0 };

	double bSum1 = 0.0;
	double bSum2 = 0.0;
	double bSum3 = 0.0;

	double dL = 0.0, da = 0.0, db = 0.0;

	const double sigma = 100.00 * sigmaR;
	const double divider = 2.00 * sigma * sigma;

	int i = 0, j = 0, k= 0, l = 0, m = 0;

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
			const int iSize = iDiff * CIELabBufferbands * sizeof(double);

			// get processed pixel
			const int srcIdx = j * CIELabLinePitch + i * CIELabBufferbands;
			const double L = pCIELab[srcIdx];
			const double a = pCIELab[srcIdx + 1];
			const double b = pCIELab[srcIdx + 2];

			// copy window of pixels to temporal array
			for (k = 0; k < jDiff; k++)
			{
				const int jIdx = (jMin + k) * CIELabLinePitch + iMin * CIELabBufferbands;
				memcpy(pI[k], &pCIELab[jIdx], iSize);
			} // for (k = 0; k < jDiff; k++)
			  
			  // compute Gaussian range weights
			for (k = 0; k < jDiff; k++)
			{
				__VECTOR_ALIGNED__
				for (m = l = 0; l < iDiff; l++, m += 3)
				{
					dL = pI[k][m] - L;
					da = pI[k][m + 1] - a;
					db = pI[k][m + 2] - b;

					const double dotComp = dL * dL + da * da + db * db;
					pH[k][l] = EXP(-dotComp / divider);
				}
			}

			// calculate bilateral filter responce
			double norm_F = 0.0;
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
				__VECTOR_ALIGNED__
				for (m = l = 0; l < iDiff; l++, m += 3)
				{
					bSum1 += (pF[k][l] * pI[k][m]);
					bSum2 += (pF[k][l] * pI[k][m + 1]);
					bSum3 += (pF[k][l] * pI[k][m + 2]);
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

#if 0
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
					pSrcBGRA,   /* format: A, B, G, R (each band as unsigned char) */
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

#endif
