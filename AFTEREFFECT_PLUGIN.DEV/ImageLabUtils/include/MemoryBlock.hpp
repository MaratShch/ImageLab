#pragma once
#include <stdint.h>
#include <type_traits>
#include "LibExport.hpp"

namespace ImageLabMemoryUtils
{

	class CMemoryBlock
	{
		public:
			CMemoryBlock() = default;
			virtual ~CMemoryBlock(void) { memBlockFree(); };

			bool getMemProperties(void** pMem, uint32_t* pSize, uint32_t* pAlign = nullptr) const noexcept
			{
				if (nullptr != pMem)
					*pMem = m_memoryPtr;
				if (nullptr != pSize)
					*pSize = m_memorySize;
				if (nullptr != pAlign)
					*pAlign = m_alignment;
				return true;
			}

			bool memBlockAlloc(uint32_t mSize, uint32_t mAlign = 0u) noexcept;
			void memBlockFree(void) noexcept;

			inline uint32_t getMemSize(void) const noexcept { return m_memorySize; }
			inline void*    getMemPtr(void)  const noexcept { return m_memoryPtr; }

			inline bool operator >  (const CMemoryBlock& mBlock) const noexcept { return (this->m_memorySize >  mBlock.m_memorySize ? true : false); }
			inline bool operator <  (const CMemoryBlock& mBlock) const noexcept { return (this->m_memorySize <  mBlock.m_memorySize ? true : false); }
			inline bool operator >= (const CMemoryBlock& mBlock) const noexcept { return (this->m_memorySize >= mBlock.m_memorySize ? true : false); }
			inline bool operator <= (const CMemoryBlock& mBlock) const noexcept { return (this->m_memorySize <= mBlock.m_memorySize ? true : false); }

			template <typename T>
			inline typename std::enable_if<std::is_integral<T>::value>::type operator > (const T& mSize) const noexcept
			{
				return (this->m_memorySize > mSize ? true : false);
			}

			template <typename T>
			inline typename std::enable_if<std::is_integral<T>::value>::type operator < (const T& mSize) const noexcept
			{
				return (this->m_memorySize < mSize ? true : false);
			}

			template <typename T>
			inline typename std::enable_if<std::is_integral<T>::value>::type operator >= (const T& mSize) const noexcept
			{
				return (this->m_memorySize >= mSize ? true : false);
			}

			template <typename T>
			inline typename std::enable_if<std::is_integral<T>::value>::type operator <= (const T& mSize) const noexcept
			{
				return (this->m_memorySize <= mSize ? true : false);
			}

		private:
			uint32_t m_memorySize = 0;
			uint32_t m_alignment = 0;
			void*    m_memoryPtr = nullptr;
	};

} // namespace ImageLabMemoryUtils;

