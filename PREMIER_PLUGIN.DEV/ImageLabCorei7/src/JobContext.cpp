#include "JobContext.h"


JobContext::JobContext()
{
	idealProcessor = 0;
	currentJodIdx = -1;

	jobQueue = new jobQueueEntry[JOB_CONTEXT_MAX_JOBS];
	hJobEvents[JOB_EVENT_EXIT] = CreateEvent(NULL, FALSE, FALSE, NULL);
	hJobEvents[JOB_EVENT_RUN]  = CreateEvent(NULL, FALSE, FALSE, NULL);
	hJobExitConfirmEvent       = CreateEvent(NULL, FALSE, FALSE, NULL);

	//	hThread = CrateThread (NULL, )
//	return;
}

JobContext::JobContext(const unsigned int& coreNum)
{
	idealProcessor = static_cast<DWORD>(coreNum);
	hJobEvents[JOB_EVENT_EXIT] = CreateEvent(NULL, FALSE, FALSE, NULL);
	hJobEvents[JOB_EVENT_RUN] = CreateEvent(NULL, FALSE, FALSE, NULL);
	hJobExitConfirmEvent = CreateEvent(NULL, FALSE, FALSE, NULL);

	currentJodIdx = -1;
	jobQueue = new jobQueueEntry[JOB_CONTEXT_MAX_JOBS];
	return;
}

JobContext::~JobContext()
{
	SetEvent(hJobEvents[JOB_EVENT_EXIT]);

	// waits 500 mS till thread complete the current job and exit
	WaitForSingleObject(hJobExitConfirmEvent, 500);

	// for DBG purpose only
	memset(jobQueue, 0, JOB_CONTEXT_MAX_JOBS*sizeof(jobQueueEntry));

	delete[] jobQueue;
	jobQueue = nullptr;

	CloseHandle(hThread);
	hThread = INVALID_HANDLE_VALUE;

	CloseHandle(hJobEvents[JOB_EVENT_EXIT]);
	hJobEvents[JOB_EVENT_EXIT] = INVALID_HANDLE_VALUE;

	CloseHandle(hJobEvents[JOB_EVENT_RUN]);
	hJobEvents[JOB_EVENT_RUN] = INVALID_HANDLE_VALUE;

	CloseHandle(hJobExitConfirmEvent);
	hJobExitConfirmEvent = INVALID_HANDLE_VALUE;
}

bool JobContext::putAsyncJob(jobFuction pFunction, LPVOID pParam1, LPVOID pParam2)
{
	if (-1 == currentJodIdx)
		currentJodIdx = 0;
	else {
		currentJodIdx++;
		if (currentJodIdx >= (JOB_CONTEXT_MAX_JOBS))
			currentJodIdx = 0;
	}


	jobQueue[currentJodIdx].pFunction = pFunction;
	jobQueue[currentJodIdx].pParam1 = pParam1;
	jobQueue[currentJodIdx].pParam2 = pParam2;

	SetEvent(hJobEvents[JOB_EVENT_RUN]);
	return true;
}


DWORD WINAPI _In_ JobPerformer(LPVOID lpParam)
{
	JobContext* pJobContext = reinterpret_cast<JobContext*>(lpParam);
	if (nullptr != pJobContext)
	{
		const HANDLE* hJobEvents = pJobContext->hJobEvents;
		const HANDLE  hJobExitConfirmEvent = pJobContext->hJobExitConfirmEvent;

		SetThreadIdealProcessor(GetCurrentThread(), pJobContext->idealProcessor);

		// main loop
		while (true)
		{
			const DWORD jobType = WaitForMultipleObjects(JOB_TOTAL_EVENTS, hJobEvents, FALSE, INFINITE);
			
				switch (jobType)
				{
					case WAIT_OBJECT_0:
					{// exit event
						SetEvent(hJobExitConfirmEvent);
						ExitThread(EXIT_SUCCESS);
					}
					break;

					case ((WAIT_OBJECT_0)+1):
					{// next job event
						const jobQueueEntry* currentJob = &pJobContext->jobQueue[pJobContext->currentJodIdx];
						currentJob->pFunction(currentJob->pParam1, currentJob->pParam2);
					}
					break;
					
					default:
					break;
				}
		}
	}

	// this code line never reached - just leave the code for avoid compiler warnings 
	return EXIT_SUCCESS;
}
