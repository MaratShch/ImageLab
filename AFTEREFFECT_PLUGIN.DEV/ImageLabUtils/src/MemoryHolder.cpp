#include <thread>
#include "Common.hpp"
#include "FastAriphmetics.hpp"
#include "MemoryHolder.hpp"

using namespace ImageLabMemoryUtils;


CMemoryHolder::CMemoryHolder () :
	m_HolderCapacity(FastCompute::Min(32u, std::thread::hardware_concurrency() + 1u)),
	m_Semaphore(m_HolderCapacity)
{
	m_TotalAllocated = 0ull;
	m_Holder.resize(m_HolderCapacity);

	for (int32_t i = 0; i < static_cast<int32_t>(m_HolderCapacity); i++)
	{
		m_Holder[i] = new CMemoryBlock;
		m_FreeBlocks.push_front(i);
	}

	return;
}

CMemoryHolder::~CMemoryHolder()
{

}


int32_t CMemoryHolder::searchMemoryBlock (uint32_t reqSize)
{
	int32_t blockId = INVALID_MEMORY_BLOCK;
	
	m_Semaphore.Wait();

	/* search already pre-allocated blocks */
	for (uint32_t i = 0; i < m_HolderCapacity && -1 == blockId; i++)
	{
		if (m_Holder[i]->getMemSize() >= reqSize)
		{
			/* pre-allocated block found, let's check if this block is not busy */
			/* lock queue access */
			std::unique_lock<std::mutex> lock(m_QueueMutualAccess);
			const int32_t freeQueueCapacity = static_cast<int32_t>(m_FreeBlocks.size());
			for (int32_t j = 0; j < freeQueueCapacity; j++)
			{
				if (m_FreeBlocks[j] == i)
				{
					/* this block in free queue, so we may use it */
					blockId = static_cast<int>(i);

					/* remove founded block from free queue and put it to busy queue */
					m_FreeBlocks.erase(m_FreeBlocks.begin() + j);
					m_BusyBlocks.push_front(blockId);
					break;
				}
			}
		}
	}
	/* we not found pre-allocated buffer so, let use first free element and make memory re-alloc */
	if (INVALID_MEMORY_BLOCK == blockId)
	{
		/* lock queue access */
		std::unique_lock<std::mutex> lock(m_QueueMutualAccess);

		int32_t idx = m_FreeBlocks.front();
		m_FreeBlocks.pop_front(); /* remove this element from empty queue */

		/* re-allocate requred memory */
		m_TotalAllocated -= m_Holder[idx]->getMemSize();
		m_Holder[idx]->memBlockFree();
		m_Holder[idx]->memBlockAlloc(reqSize, CACHE_LINE);
		m_TotalAllocated += m_Holder[idx]->getMemSize();

		/* put this idx into busy queue */
		m_BusyBlocks.push_front(idx);
		blockId = idx;
	}
	
	return blockId;
}


void CMemoryHolder::releaseMemoryBlock (int32_t blockIdx)
{
	if (INVALID_MEMORY_BLOCK != blockIdx && blockIdx < m_HolderCapacity)
	{
		/* lock queue access */
		std::unique_lock<std::mutex> lock(m_QueueMutualAccess);

		/* check if this blocvk in busy queue */
		const int32_t busyQueueCapacity = static_cast<int32_t>(m_BusyBlocks.size());
		for (int32_t i = 0; i < busyQueueCapacity; i++)
		{
			if (blockIdx == m_BusyBlocks[i])
			{
				/* this block in busy queue, let's move it and place into free queue */
				m_BusyBlocks.erase(m_BusyBlocks.begin() + i);
				m_FreeBlocks.push_front(blockIdx);
				break;
			}
		}
	}

	m_Semaphore.Release();

	return;
}


int32_t CMemoryHolder::AllocMemory (uint32_t memSize, void** ptr, const MemOwnedPolicy /* memory policy reserved for future implementation */)
{
	int32_t blockId = INVALID_MEMORY_BLOCK;
	if (0 != memSize && nullptr != ptr)
	{
		blockId = searchMemoryBlock(memSize);
		*ptr = m_Holder[blockId]->getMemPtr();
	}

	return ::CreateMemHanler(blockId);
}

void CMemoryHolder::ReleaseMemory (int32_t blockId)
{
	const int32_t Id = ::CreateBlockIdx(blockId);
	releaseMemoryBlock(Id);
	return;
}
