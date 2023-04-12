#include "MemoryInterface.hpp"
#include "CommonBitsOperations.hpp"
#include "FastAriphmetics.hpp"

using namespace ImageLabMemoryUtils;

std::atomic<CMemoryInterface*> CMemoryInterface::s_instance;
std::mutex CMemoryInterface::s_protectMutex;


int32_t CMemoryInterface::allocMemoryBlock(const int32_t size, void** pMem, const int32_t alignment)
{
	void* pMemory = nullptr;
	(void)alignment; // just for avoid compilation warning on currently non used parameter
	const int32_t blockIdx = m_MemHolder.AllocMemory(static_cast<uint32_t>(size), &pMemory);
	*pMem = (INVALID_MEMORY_BLOCK != blockIdx ? pMemory : nullptr);
	return blockIdx;
}


void CMemoryInterface::releaseMemoryBlock (int32_t id) 
{
	m_MemHolder.ReleaseMemory(id);
	return;
}

int64_t CMemoryInterface::getMemoryStatistics (void)
{
	return m_MemHolder.GetTotalAllocatedMem();
}