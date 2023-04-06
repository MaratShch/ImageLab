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
	class DLL_LINK CMemoryInterface
	{
		public:
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

		int32_t allocMemoryBlock(const int32_t size, void** pMem, const int32_t alignment = 0);
		void releaseMemoryBlock(int32_t id);

		private:
			CMemoryInterface() {};
			~CMemoryInterface(){};

		CLASS_NON_COPYABLE(CMemoryInterface);
		CLASS_NON_MOVABLE(CMemoryInterface);

		static std::atomic<CMemoryInterface*> s_instance;
		static std::mutex s_protectMutex;

		CMemoryHolder m_MemHolder;
	};

}