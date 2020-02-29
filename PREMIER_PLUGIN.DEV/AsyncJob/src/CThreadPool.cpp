#include "CCommon.hpp"
#include "CThreadPool.hpp"

template <typename T>
T MAX(T a, T b) { return (a > b) ? a : b; }

template <typename T>
T MIN(T a, T b) { return (a < b) ? a : b; }


const uint32_t CThreadPool::getCpuCoresNumber (void* p)
{
	return std::thread::hardware_concurrency();
}

CThreadPool::CThreadPool(void)
{
	m_cores = CThreadPool::getCpuCoresNumber();
	m_workers = 0u;
	m_pJobQueue = nullptr;
	m_threadPool.clear();
}

CThreadPool::CThreadPool(const uint32_t& limit)
{
	m_cores = MAX(1u, MIN(limit, CThreadPool::getCpuCoresNumber()));
	m_workers = 0u;
	m_pJobQueue = nullptr;
	m_threadPool.clear();
	return;
}

CThreadPool::~CThreadPool(void)
{
	m_cores = m_workers = 0u;
	free();
	m_pJobQueue = nullptr;
	return;
}


bool CThreadPool::init(void)
{
	constexpr size_t privStorageSize = defaulStorageSize;
	uint32_t affinity = 0x00000001u;

	for (uint32_t i = 0; i < m_cores; i++)
	{
		CWorkerThread* pWorker = new CWorkerThread(privStorageSize, affinity, m_pJobQueue);
		m_threadPool.push_back(pWorker);
		affinity <<= 1;
	}

	return true;
}


bool CThreadPool::init(CAsyncJobQueue* pJobQueue)
{
	m_pJobQueue = pJobQueue;
	return init();
}

void CThreadPool::free(void)
{
	auto idx = m_threadPool.size();
	const auto workersNumber = idx;

	/* in first send termination flag to all workers */
	for (idx = 0; idx < workersNumber; idx++)
	{
		m_threadPool.at(idx)->sendTermination();
	}

	/* in second - destroy worker itself and destory worker thread with private memory pool */
	for (auto idx = 0; idx < workersNumber; idx++)
	{
		CWorkerThread* pWorker = m_threadPool.at(idx);
		delete pWorker;
		pWorker = nullptr;
	}

	/* in last - cleanup container */
	m_threadPool.clear();

	return;
}


const uint32_t CThreadPool::getWorkersNumber(void) const
{
	return m_workers;
}

const CWorkerThread& CThreadPool::getWorker(uint32_t id) const
{
	return *m_threadPool.at(id);
}

const uint32_t CThreadPool::getCoresNumber(void) const
{
	return m_cores; /* get number of cores available for CThreadPool object */
}