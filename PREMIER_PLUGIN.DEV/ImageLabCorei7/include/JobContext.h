#pragma once
#include <windows.h>

#define JOB_CONTEXT_MAX_JOBS	32

#define JOB_EVENT_EXIT			0
#define JOB_EVENT_RUN			1
#define JOB_TOTAL_EVENTS		2

typedef VOID (*jobFuction)(LPVOID, LPVOID);

DWORD WINAPI _In_ JobPerformer(LPVOID lpParam);


typedef struct {
	jobFuction pFunction;
	LPVOID     pParam1;
	LPVOID     pParam2;
	LPVOID     pReserved;
}jobQueueEntry;

class JobContext
{
private:
	HANDLE hJobEvents[JOB_TOTAL_EVENTS];
	HANDLE hJobExitConfirmEvent;
	HANDLE hThread;
	jobQueueEntry* jobQueue;
	
	DWORD idealProcessor;
	int currentJodIdx;

public:
	JobContext();
	JobContext(const unsigned int& coreNumber);
	virtual ~JobContext();

	bool putAsyncJob(jobFuction pFunction, LPVOID pParam1, LPVOID pParam2);

	friend	DWORD WINAPI _In_  JobPerformer(LPVOID lpParam);
};

