#include "ImageLabBilateral.h"
#include<thread>
#include<vector>

AVX2_ALIGN std::vector<HANDLE> pWorkers;
AVX2_ALIGN AsyncQueue** pAsyncQueues;

std::mutex globalMutex;

// limit parallel processing threads for DBG purpose only
const unsigned int dbgMaxParallelJobs = UINT_MAX;


void startParallelJobs(const unsigned int dbgLimit)
{
	const unsigned int numThreads = MIN(dbgLimit, std::thread::hardware_concurrency());

	std::unique_lock<std::mutex> lk(globalMutex);
	for (unsigned int i = 0u; i < numThreads; i++)
	{
		pAsyncQueues[i]->bNewJob = true;
		pAsyncQueues[i]->cv.notify_one();
	}
	lk.unlock();

	return;
}


void createTaskServers(const unsigned int dbgLimit)
{
	pAsyncQueues = nullptr;
	const unsigned int numThreads = MIN(dbgLimit, std::thread::hardware_concurrency());

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

		DWORD dwT = 0;
		HANDLE h = CreateThread(NULL,
								0,
								ProcessThread,
								pAsyncQueues[i],
								0,
								&dwT);
		pWorkers.push_back(h);
	}
}


void deleteTaskServers(const unsigned int dbgLimit)
{
	const unsigned int numThreads = MIN(dbgLimit, std::thread::hardware_concurrency());

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



BOOL APIENTRY DllMain(HMODULE /* hModule */, DWORD ul_reason_for_call, LPVOID /* lpReserved */)
{
	switch (ul_reason_for_call)
	{
		case DLL_PROCESS_ATTACH:
		{
			CreateColorConvertTable();
			createTaskServers(dbgMaxParallelJobs);
		}
		break;

		case DLL_THREAD_ATTACH:
		break;

		case DLL_THREAD_DETACH:
		break;

		case DLL_PROCESS_DETACH:
		{
			deleteTaskServers(dbgMaxParallelJobs);
			DeleteColorConvertTable();
		}
		break;

		default:
		break;
		}

	return TRUE;
}

