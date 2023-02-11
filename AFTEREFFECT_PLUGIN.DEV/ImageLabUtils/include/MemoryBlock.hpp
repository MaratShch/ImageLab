#pragma once 

#include <memory>
#include "LibExport.hpp"
#include "ClassRestrictions.hpp"

namespace ImageLabMemoryUtils
{
	
	class DLL_LINK CMemoryBlock
	{
		private:
			int32_t m_memorySize;
			int32_t m_alignment;
			void*   m_memoryPtr;

			void memBlockFree (void) noexcept;
			void* memBlockAlloc(int32_t mSize, int32_t mAlign) noexcept;

		public:
			CLASS_NON_COPYABLE(CMemoryBlock);
			CLASS_NON_MOVABLE(CMemoryBlock);
			CMemoryBlock (void);
			CMemoryBlock (int32_t memSize, int32_t alignment = 0);
			virtual ~CMemoryBlock(void);

			void* memBlockRealloc  (int32_t memSize, int32_t alignment) noexcept;

			bool getMemProperties (void** pMem, int32_t* pSize, int32_t* pAlign) const noexcept
			{
				if (nullptr != pMem)
					*pMem = m_memoryPtr;
				if (nullptr != pSize)
					*pSize = m_memorySize;
				if (nullptr != pAlign)
					*pAlign = m_alignment;
				return true;
			}

			void* getMemoryBlock(void) const noexcept { return m_memoryPtr; }
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

	};

};
