#include "ImageLabBilateral.h"
//#include<thread>
//#include<vector>

#if 0
AVX2_ALIGN std::vector<HANDLE> pWorkers;
AVX2_ALIGN AsyncQueue** pAsyncQueues;


std::mutex globalMutex;


// limit parallel processing threads for DBG purpose only
const unsigned int dbgMaxParallelJobs = 1u;// UINT_MAX;


void startParallelJobs(
	csSDK_uint32* pSrc,
	csSDK_uint32* pDst,
	const int     sizeX,
	const int     sizeY,
	const int     rowBytes,
	const unsigned int dbgLimit)
{
	const unsigned int numThreads = MIN(dbgLimit, cpuCores);
	int head;

	std::unique_lock<std::mutex> lk(globalMutex);
	for (unsigned int i = 0u; i < numThreads; i++)
	{
		head = pAsyncQueues[i]->head + 1;
		if (head >= jobsQueueSize || head < 0)
			head = 0;

		pAsyncQueues[i]->jobsQueue[head].pSrcSlice = pSrc;
		pAsyncQueues[i]->jobsQueue[head].pDstSlice = pDst;
		pAsyncQueues[i]->jobsQueue[head].sizeX = sizeX;
		pAsyncQueues[i]->jobsQueue[head].sizeY = sizeY;
		pAsyncQueues[i]->jobsQueue[head].rowWidth = rowBytes;
		pAsyncQueues[i]->head = head;

		pAsyncQueues[i]->bNewJob = true;
		pAsyncQueues[i]->cv.notify_one();
	}
	lk.unlock();

	return;
}

int waitForJobsComplete(const unsigned int dbgLimit)
{
	const unsigned int numThreads = MIN(dbgLimit, cpuCores);

	for (unsigned int i = 0u; i < numThreads; i++)
	{
//		std::unique_lock<std::mutex> lk(globalMutex);
//		pAsyncJob->cv.wait(lk, [&] {return pAsyncJob->bNewJob; });
//		pAsyncJob->bNewJob = false;
//		lk.unlock();
	}

	return 0;
}

void createTaskServers(const unsigned int dbgLimit)
{
	pAsyncQueues = nullptr;
	const unsigned int numThreads = MIN(dbgLimit, cpuCores);

	pAsyncQueues = new AsyncQueue*[numThreads];
	if (nullptr == pAsyncQueues)
		return;

	for (unsigned int i = 0u; i < numThreads; i++)
	{
		pAsyncQueues[i] = new AsyncQueue;
		if (nullptr == pAsyncQueues[i])
			break;

		pAsyncQueues[i]->strSizeOf = sizeof(AsyncQueue);
		pAsyncQueues[i]->idx = i;
		pAsyncQueues[i]->head = -1;
		pAsyncQueues[i]->tail = -1;
		memset(pAsyncQueues[i]->jobsQueue, 0, sizeof(pAsyncQueues[i]->jobsQueue));
		pAsyncQueues[i]->mustExit = false;
		pAsyncQueues[i]->bNewJob = false;

		DWORD dwT = 0UL;
		HANDLE h = CreateThread(NULL,
								0,
								ProcessThread,
								pAsyncQueues[i],
								0,
								&dwT);

		if (INVALID_HANDLE_VALUE == h)
			break;

		pWorkers.push_back(h);
	}
}


void deleteTaskServers(const unsigned int dbgLimit)
{
	const unsigned int numThreads = MIN(dbgLimit, cpuCores);

	if (nullptr != pAsyncQueues)
	{
		// signal to thread exit
		for (unsigned int i = 0u; i < numThreads; i++)
		{
			if (nullptr != pAsyncQueues[i])
			{
				pAsyncQueues[i]->head = pAsyncQueues[i]->tail = -1;
				pAsyncQueues[i]->mustExit = true;

				std::unique_lock<std::mutex> lk(globalMutex);
				pAsyncQueues[i]->bNewJob = true; // set conditional variable for unlock the thread
				pAsyncQueues[i]->cv.notify_one();
				lk.unlock();

				std::this_thread::sleep_for(std::chrono::milliseconds(10));

				// wait for the worker
				{
					std::unique_lock<std::mutex> lk(globalMutex);
					pAsyncQueues[i]->cv.wait(lk, [&] {return pAsyncQueues[i]->bJobComplete;});
					lk.unlock();
				}

			}
		}

		// wait till all threads complete to job and exits
		for (HANDLE t : pWorkers)
		{
			CloseHandle(t);
			t = nullptr;
		}

		// memory cleanup
		for (unsigned int i = 0u; i < numThreads; i++)
		{
			delete pAsyncQueues[i];
			pAsyncQueues[i] = nullptr;
		}

		delete [] pAsyncQueues;
		pAsyncQueues = nullptr;
	}

	return;
}
#endif

extern float* pBuffer1;
extern float* pBuffer2;


BOOL APIENTRY DllMain(HMODULE /* hModule */, DWORD ul_reason_for_call, LPVOID /* lpReserved */)
{
	switch (ul_reason_for_call)
	{
		case DLL_PROCESS_ATTACH:
		{
			// allocate memory buffers for temporary procssing
			pBuffer1 = reinterpret_cast<float*>(allocCIELabBuffer(CIELabBufferSize));
			pBuffer2 = reinterpret_cast<float*>(allocCIELabBuffer(CIELabBufferSize));

			if (nullptr != pBuffer1 && nullptr != pBuffer2)
			{
				ZeroMemory(pBuffer1, CIELabBufferSize);
				ZeroMemory(pBuffer2, CIELabBufferSize);
			}
			CreateColorConvertTable();
			gaussian_weights();
//			createTaskServers(dbgMaxParallelJobs);
		}
		break;

		case DLL_THREAD_ATTACH:
		break;

		case DLL_THREAD_DETACH:
		break;

		case DLL_PROCESS_DETACH:
		{
//			deleteTaskServers(dbgMaxParallelJobs);
			DeleteColorConvertTable();
			freeCIELabBuffer(pBuffer1);
			freeCIELabBuffer(pBuffer2);
			pBuffer1 = pBuffer2 = nullptr;;
		}
		break;

		default:
		break;
		}

	return TRUE;
}

