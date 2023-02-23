#include "MemoryInterface.hpp"
#include "CommonBitsOperations.hpp"
#include "FastAriphmetics.hpp"

using namespace ImageLabMemoryUtils;

DLL_LINK std::atomic<CMemoryInterface*> CMemoryInterface::s_instance;
DLL_LINK std::mutex CMemoryInterface::s_protectMutex;



CMemoryInterface* ImageLabMemoryUtils::getMemoryInterface (void) noexcept
{
	return CMemoryInterface::getInstance();
}

CMemoryInterface::CMemoryInterface()
{
	return;
}

CMemoryInterface::~CMemoryInterface()
{
	return;
}


int32_t CMemoryInterface::allocMemoryBlock(const int32_t& size, void** pMem, const int32_t& alignment)
{
	void* pMemory = nullptr;
	const int32_t blockIdx = m_MemHolder.AllocMemory (static_cast<uint32_t>(size), &pMemory);
	*pMem = pMemory;
	return blockIdx;
}


void CMemoryInterface::releaseMemoryBlock (int32_t id)
{
	m_MemHolder.ReleaseMemory(id);
	return;
}
