#pragma once
#include "CWorkerThread.hpp"

class CAsyncJobQueue;

class CThreadPool
{
public:
	explicit CThreadPool(void);
	explicit CThreadPool(const uint32_t& limit);

	/* non copyable and non movable, no assign operator */
	CThreadPool(const CThreadPool&) = delete;
	CThreadPool(const CThreadPool&&) = delete;
	CThreadPool operator = (const CThreadPool&) = delete;

	virtual ~CThreadPool();

	bool init(void);

	const CWorkerThread& getWorker(uint32_t id) const;
	const uint32_t getWorkersNumber(void) const;
	const uint32_t getCoresNumber(void) const;

	const static uint32_t getCpuCoresNumber(void* p = nullptr);

protected:
private:
	std::vector<CWorkerThread*> m_threadPool;
	CAsyncJobQueue* pJobQueue;
	uint32_t m_workers;
	uint32_t m_cores;
};