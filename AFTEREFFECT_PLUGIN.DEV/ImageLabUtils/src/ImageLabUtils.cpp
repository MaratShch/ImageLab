#include "ImageLabUtils.hpp"
#include "MemoryHolder.hpp"
#include "MemoryInterface.hpp"


void* CreateMemoryHandler(void)
{
	return reinterpret_cast<void*>(ImageLabMemoryUtils::CMemoryInterface::getInstance());
}

void  ReleaseMemoryHandler(void* p)
{
	/* nothing to do */
	(void)p;
	return;
}


int32_t AllocMemoryBlock (void* pMemHandle, int32_t size, int32_t align, void** pMem)
{
	int32_t blockId = ImageLabMemoryUtils::INVALID_MEMORY_BLOCK;
	ImageLabMemoryUtils::CMemoryInterface* p = reinterpret_cast<ImageLabMemoryUtils::CMemoryInterface*>(pMemHandle);
	if (nullptr != p && nullptr != pMem)
	{
		void* pMemPtr = nullptr;
		blockId = p->allocMemoryBlock (size, &pMemPtr);
		if (ImageLabMemoryUtils::INVALID_MEMORY_BLOCK != blockId && nullptr != pMemPtr)
			*pMem = pMemPtr;
		else
		{
			*pMem = nullptr;
			blockId = ImageLabMemoryUtils::INVALID_MEMORY_BLOCK;
		}
	}

	return blockId;
}


void ReleaseMemoryBlock (void* pMemHandle, int32_t id)
{
	ImageLabMemoryUtils::CMemoryInterface* p = reinterpret_cast<ImageLabMemoryUtils::CMemoryInterface*>(pMemHandle);
	if (nullptr != p)
		p->releaseMemoryBlock(id);
}

int64_t GetMemoryStatistics (void* pMemHandle)
{
	ImageLabMemoryUtils::CMemoryInterface* p = reinterpret_cast<ImageLabMemoryUtils::CMemoryInterface*>(pMemHandle);
	return (nullptr != p) ? p->getMemoryStatistics() : -1LL;
}
