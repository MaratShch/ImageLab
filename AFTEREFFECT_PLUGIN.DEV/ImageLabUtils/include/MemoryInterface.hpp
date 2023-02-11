#pragma once 

#include <atomic>
#include <mutex>
#include <thread>
#include <vector>
#include "LibExport.hpp"
#include "MemoryBlock.hpp"
#include "UtilsSemaphore.hpp"


namespace ImageLabMemoryUtils
{
	class DLL_LINK MemoryAccess
	{
		public:
			CLASS_NON_COPYABLE(MemoryAccess);
			CLASS_NON_MOVABLE(MemoryAccess);

			MemoryAccess (uint32_t cpu_cores);
			~MemoryAccess();
			uint32_t GetMemoryBlock(uint32_t requestSize);
			void ReleaseMemoryBlock(uint32_t blockId);
	private:
			Semaphore mSemaphore;
			uint32_t m_accessPool;
			uint32_t m_busyMask;
			std::vector<CMemoryBlock*> m_MemHandle;
			std::atomic<uint32_t> m_Busy;
	};


	class DLL_LINK MemoryInterface
	{
		public:
		static MemoryInterface* getInstance()
		{
			MemoryInterface* iMemory = s_instance.load (std::memory_order_acquire);
			if (nullptr == iMemory)
			{
				std::lock_guard<std::mutex> myLock(s_protectMutex);
				iMemory = s_instance.load(std::memory_order_relaxed);
				if (nullptr == iMemory)
				{
					iMemory = new MemoryInterface();
					s_instance.store(iMemory, std::memory_order_release);
				}
			}
			return iMemory;
		} /* static MemoryInterface* getInstance() */

		private:
		MemoryInterface();
		~MemoryInterface();

		CLASS_NON_COPYABLE(MemoryInterface);
		CLASS_NON_MOVABLE(MemoryInterface);

		static std::atomic<MemoryInterface*> s_instance;
		static std::mutex s_protectMutex;

		MemoryAccess mMemAccess;
	};

	MemoryInterface* getMemoryInterface(void) noexcept;
}
