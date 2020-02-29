#pragma once
#include "CWorkerThread.hpp"


class CAsyncJobQueue;

class CLASS_EXPORT CThreadPool
{
public:
	explicit CThreadPool(void);
	explicit CThreadPool(const uint32_t& limit);

	/* non copyable and non movable, no assign operator */
	CLASS_NON_COPYABLE(CThreadPool);
	CLASS_NON_MOVABLE(CThreadPool);

	virtual ~CThreadPool();

	bool init(void);
	bool init(CAsyncJobQueue* pJobQueue);

	const CWorkerThread& getWorker(uint32_t id) const;
	const uint32_t getWorkersNumber(void) const;
	const uint32_t getCoresNumber(void) const;

	const static uint32_t getCpuCoresNumber(void* p = nullptr);

protected:
private:
	std::vector<CWorkerThread*> m_threadPool;
	CAsyncJobQueue* m_pJobQueue;
	uint32_t m_workers;
	uint32_t m_cores;

	void free(void);
};