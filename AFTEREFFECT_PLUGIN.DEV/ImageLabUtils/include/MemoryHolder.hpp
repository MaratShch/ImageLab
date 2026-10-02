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
    constexpr int32_t MEM_HANDLE_TAG = 0x00FF0000;
    constexpr int32_t INVALID_MEMORY_BLOCK = -1;

	inline int32_t CreateMemHanler(int32_t idx) noexcept { return idx  | MEM_HANDLE_TAG; }

    inline int32_t GetBlockIdx(int32_t hndl) noexcept
    {
        return ((hndl & static_cast<int32_t>(0xFFFF0000u)) == MEM_HANDLE_TAG)
            ? (hndl & 0x0000FFFF) : INVALID_MEMORY_BLOCK;
    }



	class CMemoryHolder
	{
		public:
			CLASS_NON_COPYABLE(CMemoryHolder);
			CLASS_NON_MOVABLE(CMemoryHolder);

			CMemoryHolder ();
			virtual ~CMemoryHolder();

			int32_t AllocMemory(uint32_t memSize, void** ptr, const MemOwnedPolicy = MemOwnedPolicy::MEM_POLICY_NORMAL);
			void ReleaseMemory(int32_t blockId);

			const uint64_t GetTotalAllocatedMem (void) const noexcept
			{
				return m_TotalAllocated;
			}

		private:
			int32_t searchMemoryBlock (uint32_t reqSize = 0);
			void releaseMemoryBlock (int32_t blockIdx);

			const int32_t m_HolderCapacity;
			std::atomic<uint64_t> m_TotalAllocated;
			std::mutex m_QueueMutualAccess;
			std::deque<int32_t> m_FreeBlocks;
			std::deque<int32_t> m_BusyBlocks;
			std::vector<CMemoryBlock*>m_Holder;
			CSemaphore m_Semaphore;
	};

} // namespace ImageLabMemoryUtils
