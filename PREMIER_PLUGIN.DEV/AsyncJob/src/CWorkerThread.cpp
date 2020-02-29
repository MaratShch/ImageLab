#include "CWorkerThread.hpp"
#include "CAsyncJobQueue.hpp"
#include "CCommon.hpp"
#include <iostream>

using namespace std::chrono_literals;

std::atomic<uint32_t>CWorkerThread::totalWorkers = {0u};


void WorkerCyclicJob (CWorkerThread* p, void* pQueue)
{
#ifdef _DEBUG
	constexpr size_t dbgWorkerCacheSize = workerCacheSize + 64u;
	CACHE_ALIGN uint8_t workerCacheMem[dbgWorkerCacheSize] = { 0 };
	workerCacheMem[workerCacheSize + 1] = 'D';
	workerCacheMem[workerCacheSize + 2] = 'E';
	workerCacheMem[workerCacheSize + 3] = 'A';
	workerCacheMem[workerCacheSize + 4] = 'D';
	workerCacheMem[workerCacheSize + 5] = 'B';
	workerCacheMem[workerCacheSize + 6] = 'E';
	workerCacheMem[workerCacheSize + 7] = 'E';
	workerCacheMem[workerCacheSize + 8] = 'F';
#else
	CACHE_ALIGN uint8_t workerCacheMem[workerCacheSize];
#endif

	CAsyncJobQueue* pJobQueue = reinterpret_cast<CAsyncJobQueue*>(pQueue);

	if (nullptr == p)
		return;
	if (nullptr == pJobQueue)
		return;

	p->m_bAllJobCompleted = false;
	CWorkerThread::totalWorkers++;

	std::cout << "Worker " << CWorkerThread::totalWorkers << " started..." << std::endl;

	/* yield current worker on start, for take initialization time for another workers */
	std::this_thread::yield();

	/* !NOTE! private worker storage will be allocated here and deleted in class constructor -
	   this allow rapid worker restarting */
	if (0 != p->m_privateStorageSize && nullptr == p->m_privateStorage)
	{
		p->m_privateStorage = allocate_aligned_mem(p->m_privateStorageSize);
#ifdef _DEBUG
		/* cleanup private memory storage for debug purpose only */
		memset(p->m_privateStorage, 0, p->m_privateStorageSize);
#endif
	}

	/* apply affinity mask */
	p->applyAffinityMask();

	/* apply priority */
	p->setThreadPriority(p->m_priority);

	/* create lock object for synchronizaton with main thread */
	std::unique_lock<std::mutex> lk(p->m_JobMutex);

	// start worker main loop
	while (p->m_bShouldRun)
	{
		/* wait for job with default itmeout (call with timeout value = 0 for infinite wait) */
		const bool newJob = p->waitForJob(lk, timeoutDefault);

		/* test for 'should run' flag again */
		if (false == p->m_bShouldRun)
			break;
#ifdef _DEBUG
		std::cout << "Job status = " << newJob << " Total jobs = " << p->m_executedJobs  << std::endl;
#endif
		if (false == newJob)
			continue;

		/* get job */
		CAsyncJob& myJob = pJobQueue->getNewJob();
		myJob.assignTmpBuffer(p->m_privateStorage);
		myJob.assignCache(reinterpret_cast<void* RESTRICT>(workerCacheMem), workerCacheSize);

		/* execute job */
		myJob.Execute();

		/* increment execution counter */
		p->m_executedJobs++;

		/* release job (mark queue entry as "empty") */
		myJob.releaseJob();

		/* notify about job complete */
		if (myJob.isCompleteNotify())
		{
			p->notifyJobComplete(lk);
		}

	} /* while (p->m_bShouldRun) */

	/* notify about all jobs completed */
	p->m_bAllJobCompleted = true;

#ifdef _DEBUG
	/* exit from worker */
	std::cout << "Worker completed ..." << std::endl;
	std::cout << "Total jobs executed: " << p->m_executedJobs << std::endl;
#endif

	CWorkerThread::totalWorkers--;
	return;
}


#ifdef _WINDOWS
DWORD WINAPI _winThreadRoutine (LPVOID lpParam)
{
	CWorkerThread* p = reinterpret_cast<CWorkerThread*>(lpParam);
	WorkerCyclicJob(p, p->getJobQueue());
	return EXIT_SUCCESS;
}
#endif



CWorkerThread::CWorkerThread(const uint32_t affinity, void* pJobQueue)
{
	m_bShouldRun = true;
	m_bAllJobCompleted = true;
	m_bLocked = false;
	m_newJobCnt = 0u;

	m_privateStorage = nullptr;
	m_privateStorageSize = 0;

	m_executedJobs = 0ull;
	m_pendingJobs = 0ull;
	m_lastError = 0u;
	m_affinityMask = affinity;
	m_priority = priorityNormal;

	m_jobQueue = pJobQueue;

#ifdef _WINDOWS
	m_threadId = 0;
	m_pThread = CreateThread
	(
		NULL,
		0,
		_winThreadRoutine,
		reinterpret_cast<LPVOID>(this),
		CREATE_SUSPENDED,
		&m_threadId
	);

	if (NULL != m_pThread)
	{
		ResumeThread(m_pThread);
	}
#else
    m_pTthread = new std::thread(WorkerCyclicJob, this, pJobQueue);
	m_threadId = m_pTthread->get_id();
#endif
	return;
}


CWorkerThread::CWorkerThread(const size_t privStorage, const uint32_t affinity, void* pJobQueue)
{
	m_bShouldRun = true;
	m_bAllJobCompleted = true;
	m_bLocked = false;
	m_newJobCnt = 0u;

	m_privateStorage = nullptr;
	m_privateStorageSize = privStorage;

	m_executedJobs = 0ull;
	m_pendingJobs = 0ull;
	m_lastError = 0u;
	m_affinityMask = affinity;
	m_priority = priorityNormal;

	m_jobQueue = pJobQueue;

#ifdef _WINDOWS
	m_threadId = 0;
	m_pThread = CreateThread
	(
		nullptr,
		0,
		_winThreadRoutine,
		reinterpret_cast<LPVOID>(this),
		0,
		&m_threadId
	);

	if (NULL != m_pThread)
	{
		ResumeThread(m_pThread);
    }
#else
	m_pTthread = new std::thread(WorkerCyclicJob, this, pJobQueue);
	m_threadId = m_pTthread->get_id();
#endif
	return;
}

CWorkerThread::~CWorkerThread(void)
{
	m_bShouldRun = false;
	std::this_thread::yield();
#ifdef _WINDOWS
	const auto timeoutBeforeTerminate = timeoutDefault + timeoutDefault / 10;
	if (WAIT_TIMEOUT == WaitForSingleObject(m_pThread, timeoutBeforeTerminate))
	{
		TerminateThread(m_pThread, EXIT_SUCCESS);
	}
	CloseHandle(m_pThread);
#else
	if (true == m_pTthread->joinable())
	{
		m_pTthread->join();
	}

	delete m_pTthread;
#endif
	m_pThread = nullptr;

	if (0 != m_privateStorageSize && nullptr != m_privateStorage)
	{
#ifdef _DEBUG
		/* cleanup private storage for DBG purpose only */
		memset(m_privateStorage, 0, m_privateStorageSize);
#endif
		free_aligned_mem (m_privateStorage);
		m_privateStorageSize = 0;
		m_privateStorage = nullptr;
	}
	m_affinityMask = 0u;
	m_newJobCnt = 0u;

	return;
}


inline bool CWorkerThread::waitForJob(std::unique_lock<std::mutex>& lk, const uint32_t& timeOut)
{
	bool bWaitResult = false;

	if (0u != timeOut)
	{
		auto waitTill = std::chrono::steady_clock::now() + timeOut * 1ms;
		if (m_jobCV.wait_until(lk, waitTill, [this](){ return 0 != m_newJobCnt; }))
		{
			m_newJobCnt--;
			bWaitResult = true;
		}
	}
	else
	{
		m_jobCV.wait(lk, [this](){return 0 != m_newJobCnt;});
		m_newJobCnt--;
		bWaitResult = true;
	}

	m_bLocked = bWaitResult;

	return bWaitResult;
}


inline bool CWorkerThread::notifyJobComplete(std::unique_lock<std::mutex>& lk)
{
	if (true == m_bLocked)
		lk.unlock();
	
	m_jobCV.notify_one();
	return true;
}


uint32_t CWorkerThread::getTotalWorkersNumber(void)
{
	const uint32_t var = CWorkerThread::totalWorkers;
	return var;
}

bool CWorkerThread::waitForCurrentJobComplete(std::unique_lock<std::mutex>& lk, const uint32_t& timeOut)
{
	int i = 0;
	bool bWaitResult = false;

	if (0u != timeOut)
	{
		auto waitTill = std::chrono::steady_clock::now() + timeOut * 1ms;
		if (m_jobCV.wait_until(lk, waitTill, [&i] { return i == 1; }))
			bWaitResult = true;
	}
	else
	{
		m_jobCV.wait(lk);
		bWaitResult = true;
	}

	return bWaitResult;
}


bool CWorkerThread::waitForAllJobCompleted(std::unique_lock<std::mutex>& lk, const uint32_t& timeOut)
{
	return waitForCurrentJobComplete(lk, timeOut);
}


bool CWorkerThread::terminateJob(void)
{
	return true;
}


bool CWorkerThread::restartWorker(void* p)
{
#ifdef _WINDOWS
	return false;
#else
	m_bShouldRun = false;
	if (true == m_pThread->joinable())
	{
		m_pThread->join();
	}
	delete m_pThread;
	m_pThread = new std::thread(WorkerCyclicJob, this, p);

	return true;
#endif
}

#ifdef _WINDOWS
bool CWorkerThread::setWindowsThreadPriority(int winPrio)
{
	bool b = false;

	if (m_pThread)
	{
		if (0 != SetThreadPriority(m_pThread, winPrio))
			b = true;
	}

	return b;
}
#endif


bool CWorkerThread::setThreadPriority(const uint32_t& pri)
{
#ifdef _WINDOWS
	auto handle = m_pThread;
	BOOL bResult = SetThreadPriority(handle, translatePriority(pri));
#else
	int bResult = 1;
#endif
	return ((bResult != 0) ? true : false);
}


inline int CWorkerThread::translatePriority (uint32_t newPrio)
{
#ifdef _WINDOWS
	int windowsClassPrio = 0;

	if (newPrio >= 0 && newPrio < 15)
		windowsClassPrio = THREAD_PRIORITY_IDLE;
	else if (newPrio >= 15 && newPrio < 30)
		windowsClassPrio = THREAD_PRIORITY_LOWEST;
	else if (newPrio >= 30 && newPrio < 45)
		windowsClassPrio = THREAD_PRIORITY_BELOW_NORMAL;
	else if (newPrio >= 45 && newPrio < 60)
		windowsClassPrio = THREAD_PRIORITY_NORMAL;
	else if (newPrio >= 60 && newPrio < 75)
		windowsClassPrio = THREAD_PRIORITY_ABOVE_NORMAL;
	else if (newPrio >= 75 && newPrio < 90)
		windowsClassPrio = THREAD_PRIORITY_HIGHEST;
	else
		windowsClassPrio = THREAD_PRIORITY_TIME_CRITICAL;

	return windowsClassPrio;
#else
	return static_cast<int>(newPrio);
#endif
}


void CWorkerThread::applyAffinityMask(void)
{
#ifdef _WINDOWS
	auto handle = m_pThread;
#else
	auto handle = m_pThread->native_handle();
#endif

#ifdef _WINDOWS
	DWORD_PTR procMask = 0, sysMask = 0, newMask = 0;
	DWORD affinityMask = static_cast<DWORD>(m_affinityMask);
	if (GetProcessAffinityMask(GetCurrentProcess(), &procMask, &sysMask))
	{
		if (newMask = (procMask & affinityMask))
		{
			SetThreadAffinityMask(static_cast<HANDLE>(handle), newMask);
		}
	}

#else
#endif

	return;
}
