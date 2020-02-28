#include "CCommon.hpp"
#include "CThreadPool.hpp"

const uint32_t CThreadPool::getCpuCoresNumber (void* p)
{
	return std::thread::hardware_concurrency();
}

CThreadPool::CThreadPool(void)
{
	m_cores = CThreadPool::getCpuCoresNumber();
	m_workers = 0u;
	pJobQueue = nullptr;
}

CThreadPool::CThreadPool(const uint32_t& limit)
{
	m_cores = MAX(1u, MIN(limit, CThreadPool::getCpuCoresNumber()));
	m_workers = 0u;
	pJobQueue = nullptr;
	return;
}

CThreadPool::~CThreadPool(void)
{
	m_cores = m_workers = 0u;
	return;
}


bool CThreadPool::init(void)
{
	constexpr size_t privStorageSize = 8 * 1024 * 1024;
	uint32_t affinity = 0x00000001u;

	for (uint32_t i = 0; i < m_cores; i++)
	{
		CWorkerThread* pWorker = new CWorkerThread(privStorageSize, affinity, pJobQueue);
		m_threadPool.push_back(pWorker);
		affinity <<= 1;
	}

	return true;
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