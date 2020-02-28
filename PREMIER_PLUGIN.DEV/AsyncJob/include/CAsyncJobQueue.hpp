#pragma once

#include "CCommon.hpp"
#include "CAsyncJob.hpp"
#include "CThreadPool.hpp"

class CThreadPool;

class CAsyncJobQueue
{
public:
	explicit CAsyncJobQueue(void);
	explicit CAsyncJobQueue(size_t entries);
	explicit CAsyncJobQueue(uint32_t workers, size_t entries);
	explicit CAsyncJobQueue(const CThreadPool& tPool);
	explicit CAsyncJobQueue(const CThreadPool& tPool, size_t entries);
	virtual ~CAsyncJobQueue(void);

	/* non copyable, non movable, non assignable */
	CAsyncJobQueue(const CAsyncJobQueue&) = delete;
	CAsyncJobQueue(const CAsyncJobQueue&&) = delete;
	CAsyncJobQueue operator = (const CAsyncJobQueue&) = delete;

	CAsyncJob& createNewJob(void);
	CAsyncJob& getNewJob(void);

	void dropAllJobs(void);

protected:
private:
	CAsyncJob* m_asyncQueue;
	size_t m_queueEntries;
	uint32_t m_current;

	uint32_t m_head;
	uint32_t m_tail;
};