#pragma once

#include <atomic>
#include <deque>
#include <mutex>
#include <vector>
#include "ClassRestrictions.hpp"
#include "MemoryBlock.hpp"
#include "MemoryPolicy.hpp"
#include "UtilsSemaphore.hpp"

namespace ImageLabMemoryUtils
{
	inline int32_t CreateMemHanler(int32_t idx) noexcept { return idx  | 0xFF0000; }
	inline int32_t CreateBlockIdx (int32_t hndl)noexcept { return hndl & 0x00FFFF; }

	constexpr int32_t INVALID_MEMORY_BLOCK = -1;

	class CMemoryHolder
	{
		public:
			CLASS_NON_COPYABLE(CMemoryHolder);
			CLASS_NON_MOVABLE(CMemoryHolder);

			CMemoryHolder ();
			virtual ~CMemoryHolder();

			int32_t AllocMemory(uint32_t memSize, void** ptr, const MemOwnedPolicy = MemOwnedPolicy::MEM_POLICY_NORMAL);
			void ReleaseMemory(int32_t blockId);

			const int64_t GetTotalAllocatedMem (void) const
			{
				const int64_t totalBytes{ m_TotalAllocated };
				return totalBytes;
			}

		private:
			int32_t searchMemoryBlock (uint32_t reqSize = 0);
			void releaseMemoryBlock (int32_t blockIdx);

			const uint32_t m_HolderCapacity;
			std::atomic<uint64_t> m_TotalAllocated;
			std::mutex m_QueueMutualAccess;
			std::deque<int32_t> m_FreeBlocks;
			std::deque<int32_t> m_BusyBlocks;
			std::vector<CMemoryBlock*>m_Holder;
			CSemaphore m_Semaphore;
	};

} // namespace ImageLabMemoryUtils
