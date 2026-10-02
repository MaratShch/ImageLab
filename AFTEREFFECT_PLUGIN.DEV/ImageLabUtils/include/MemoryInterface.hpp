#pragma once 

#include <atomic>
#include <mutex>
#include <thread>
#include <vector>
#include "LibExport.hpp"
#include "ClassRestrictions.hpp"
#include "MemoryHolder.hpp"


namespace ImageLabMemoryUtils
{
	class CMemoryInterface final
	{
		public:

        CLASS_NON_COPYABLE(CMemoryInterface);
        CLASS_NON_MOVABLE(CMemoryInterface);

        static CMemoryInterface* getInstance()
		{
			CMemoryInterface* iMemory = s_instance.load (std::memory_order_acquire);
			if (nullptr == iMemory)
			{
				std::lock_guard<std::mutex> myLock(s_protectMutex);
				iMemory = s_instance.load(std::memory_order_relaxed);
				if (nullptr == iMemory)
				{
					iMemory = new CMemoryInterface();
					s_instance.store(iMemory, std::memory_order_release);
				}
			}
			return iMemory;
		} /* static MemoryInterface* getInstance() */

        static void destroyInstance() noexcept
        {
            std::lock_guard<std::mutex> myLock(s_protectMutex);
            CMemoryInterface* p = s_instance.exchange(nullptr, std::memory_order_acq_rel);
            delete p; // ~CMemoryHolder frees blocks; ~CSemaphore closes handle
        }

		int32_t allocMemoryBlock(const int32_t size, void** pMem, const int32_t alignment = 0);
		void releaseMemoryBlock (int32_t id);
		int64_t getMemoryStatistics(void);

		private:
			CMemoryInterface() {};
			~CMemoryInterface(){};

		CMemoryHolder m_MemHolder;

		static std::atomic<CMemoryInterface*> s_instance;
		static std::mutex s_protectMutex;
	};

}