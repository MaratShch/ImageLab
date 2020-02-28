#include "CAsyncJobQueue.hpp"

std::mutex m_entryMutex;

CAsyncJobQueue::CAsyncJobQueue(void)
{
	const auto totalCores = CThreadPool::getCpuCoresNumber();
	m_queueEntries = workerQueueSize * totalCores;
	m_current = 0u;
	m_asyncQueue = new CAsyncJob[m_queueEntries];
	return;
}

CAsyncJobQueue::CAsyncJobQueue(size_t entries)
{
	m_queueEntries = entries;
	m_current = 0u;
	m_asyncQueue = new CAsyncJob[m_queueEntries];
	return;
}

CAsyncJobQueue::CAsyncJobQueue(uint32_t workers, size_t entries)
{
	m_queueEntries = workers * entries;
	m_current = 0u;
	m_asyncQueue = new CAsyncJob[m_queueEntries];
	return;
}

CAsyncJobQueue::CAsyncJobQueue(const CThreadPool& tPool)
{
	m_queueEntries = tPool.getCoresNumber() * workerQueueSize;
	m_current = 0u;
	m_asyncQueue = new CAsyncJob[m_queueEntries];
	return;
}

CAsyncJobQueue::CAsyncJobQueue(const CThreadPool& tPool, size_t entries)
{
	m_queueEntries = tPool.getCoresNumber() * entries;
	m_current = 0u;
	m_asyncQueue = new CAsyncJob[m_queueEntries];
	return;
}

CAsyncJobQueue::~CAsyncJobQueue(void)
{
	m_current = 0u;
	m_queueEntries = 0u;

	delete[] m_asyncQueue;
	m_asyncQueue = nullptr;
	return;
}


void CAsyncJobQueue::dropAllJobs(void)
{
	std::lock_guard<std::mutex>lk(m_entryMutex);
	m_current = 0u;
	m_head = m_tail = 0u;
	// m_entryMutex is automatically released when lock goes out of scope
	return;
}

CAsyncJob& CAsyncJobQueue::getNewJob(void)
{
	uint32_t current = 0u;
	{
		std::lock_guard<std::mutex> lock (m_entryMutex);
		if (m_tail == m_head)
		{
			// synchronization mechanism damaged. No new Jobs!!!
			// mark job as invalid
		}
		current = m_tail;

		m_tail++;
		if (m_tail >= static_cast<uint32_t>(m_queueEntries))
			m_tail = 0u;
	}

	return m_asyncQueue[m_tail];
}


CAsyncJob& CAsyncJobQueue::createNewJob(void)
{
	uint32_t current = 0u;
	{
		std::lock_guard<std::mutex> lock (m_entryMutex);

		current = m_head;

		m_head++;
		if (m_head >= static_cast<uint32_t>(m_queueEntries))
			m_head = 0u;

		// m_entryMutex is automatically released when lock goes out of scope
	}

	return m_asyncQueue[current];
}