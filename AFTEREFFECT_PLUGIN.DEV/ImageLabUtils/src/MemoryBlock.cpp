#include <windows.h>
#include "MemoryBlock.hpp"

using namespace ImageLabMemoryUtils;


bool CMemoryBlock::memBlockAlloc (uint32_t mSize, uint32_t mAlign)
{
	bool bRet = false;
    (void)mAlign; // this parameter not used currently

	if (mSize > 0u)
	{
		const SIZE_T memSize = static_cast<SIZE_T>(mSize);
		constexpr DWORD allocType = MEM_RESERVE | MEM_COMMIT | MEM_TOP_DOWN;
		LPVOID p = VirtualAlloc (NULL, memSize, allocType, PAGE_READWRITE);
		if (NULL != p)
		{
			m_memoryPtr = p;
			m_memorySize = mSize;
			m_alignment = mAlign;
			bRet = true;
		}
	}

	return bRet;
}

void CMemoryBlock::memBlockFree (void) 
{
	if (nullptr != m_memoryPtr && 0u != m_memorySize)
	{
		VirtualFree (m_memoryPtr, 0, MEM_RELEASE);
		m_memoryPtr = nullptr;
		m_memorySize = m_alignment = 0;
	}
	return;
}

