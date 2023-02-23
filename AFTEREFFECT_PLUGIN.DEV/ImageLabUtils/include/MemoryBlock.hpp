#pragma once
#include <stdint.h>
#include <type_traits>

namespace ImageLabMemoryUtils
{

	class CMemoryBlock
	{
		public:
			CMemoryBlock(void);
			virtual ~CMemoryBlock(void);

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

			bool memBlockAlloc(uint32_t mSize, uint32_t mAlign = 0u);
			void memBlockFree(void);

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
			uint32_t m_memorySize;
			uint32_t m_alignment;
			void*    m_memoryPtr;
	};

} // namespace ImageLabMemoryUtils;