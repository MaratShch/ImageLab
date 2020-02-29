#pragma once

#include <thread>
#include <vector>
#include <functional>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include "CCommon.hpp"

#ifdef _WINDOWS
#include <windows.h>
#else
#include <pthread.h>
#endif

#include "CCommon.hpp"

//template <typename FA1, typename FA2, void(*FP)(FA1, FA2)>
//struct WorkerContext;

class CLASS_EXPORT CWorkerThread
{
public:
	explicit CWorkerThread(const uint32_t affinity = UINT_MAX, void* pJobQueue = nullptr);
	explicit CWorkerThread(const size_t privStorage, const uint32_t affinity = UINT_MAX, void* pJobQueue = nullptr);
	virtual ~CWorkerThread(void);
	uint32_t getTotalWorkersNumber(void);
	bool setThreadPriority(const uint32_t& pri);

#ifdef _WINDOWS
	bool setWindowsThreadPriority(int winPrio);
	bool setWindowsThreadPriorityAboveNormal(void) {
		return setWindowsThreadPriority(THREAD_PRIORITY_ABOVE_NORMAL);
	}
	bool setWindowsThreadPriorityBelowNormal(void) {
		return setWindowsThreadPriority(THREAD_PRIORITY_BELOW_NORMAL);
	}
#endif

	bool waitForCurrentJobComplete(std::unique_lock<std::mutex>& lk, const uint32_t& timeOut);
	bool waitForAllJobCompleted(std::unique_lock<std::mutex>& lk, const uint32_t& timeOut);
	bool terminateJob(void);
	bool restartWorker(void* p = nullptr);
	void signalNewJob (void) {m_newJobCnt++;}
	void sendTermination(void) { m_bShouldRun = false; }
	const size_t& getPrivateStorageSize (void) { return m_privateStorageSize; }

protected:
	void applyAffinityMask(void);


private:
	uint32_t m_affinityMask;
	uint32_t m_priority;
	uint32_t m_lastError;
	uint64_t m_executedJobs;
	uint64_t m_pendingJobs;

	void* RESTRICT m_privateStorage;
	size_t m_privateStorageSize;

	std::thread* m_pTthread;
	std::thread::id m_threadId;
	std::mutex m_JobMutex;
	std::condition_variable m_jobCV;

	std::atomic<uint32_t> m_newJobCnt;
	std::atomic<bool> m_bShouldRun;
	std::atomic<bool> m_bLocked;
	std::atomic<bool> m_bAllJobCompleted;

	static std::atomic<uint32_t> totalWorkers;


	inline int translatePriority(uint32_t newPrio);

	inline bool waitForJob(std::unique_lock<std::mutex>& lk, const uint32_t& timeOut = 0u);
	inline bool notifyJobComplete(std::unique_lock<std::mutex>& lk);

	friend void WorkerCyclicJob(CWorkerThread* p, void* pJobQueue);
};
