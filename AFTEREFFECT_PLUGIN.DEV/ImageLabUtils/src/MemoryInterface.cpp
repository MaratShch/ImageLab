#include "MemoryInterface.hpp"
#include "UtilsSemaphore.hpp"
#include "CommonBitsOperations.hpp"

using namespace ImageLabMemoryUtils;

DLL_LINK std::atomic<MemoryInterface*> MemoryInterface::s_instance;
DLL_LINK std::mutex MemoryInterface::s_protectMutex;


MemoryAccess::MemoryAccess (uint32_t cpu_cores) :
	mSemaphore (cpu_cores),
	m_accessPool(cpu_cores)
{
	uint32_t busyMask = 0x0;
	m_Busy = 0u;
	m_MemHandle.resize(cpu_cores);
	for (auto i = 0; i < m_MemHandle.size(); i++)
	{
		m_MemHandle[i] = new CMemoryBlock;
		busyMask = IMLAB_BIT_SET(busyMask, i);
	}
	m_busyMask = busyMask;

	return;
}

MemoryAccess::~MemoryAccess()
{
	for (auto i = 0; i < m_MemHandle.size(); i++)
	{
		delete m_MemHandle[i];
		m_MemHandle[i] = nullptr;
	}

	return;
}

uint32_t MemoryAccess::GetMemoryBlock (uint32_t requestedSize) noexcept
{
	return 0u;
}

void MemoryAccess::ReleaseMemoryBlock(uint32_t blockId) noexcept
{
	return;
}

MemoryInterface* getMemoryInterface(void) noexcept
{
	return MemoryInterface::getInstance();
}

MemoryInterface::MemoryInterface() :
	mMemAccess(std::thread::hardware_concurrency() + 1)
{
	return;
}

MemoryInterface::~MemoryInterface()
{
	return;
}
