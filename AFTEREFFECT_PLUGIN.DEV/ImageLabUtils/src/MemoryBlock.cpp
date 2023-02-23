#include "MemoryBlock.hpp"

using namespace ImageLabMemoryUtils;


bool CMemoryBlock::memBlockAlloc (uint32_t mSize, uint32_t mAlign)
{
	void* p = nullptr;
	bool bRet = false;

	if (0u == mAlign)
		p = malloc(static_cast<size_t>(mSize));
	else
		p = _aligned_malloc(static_cast<size_t>(mSize), static_cast<size_t>(mAlign));

	if (nullptr != p)
	{
		m_memoryPtr = p;
		m_memorySize = mSize;
		m_alignment = mAlign;
		bRet = true;
	}

	return bRet;
}

void CMemoryBlock::memBlockFree(void) 
{
	if (nullptr != m_memoryPtr && 0u != m_memorySize)
	{
		if (0 != m_alignment)
			_aligned_free(m_memoryPtr);
		else
			free(m_memoryPtr);

		m_memoryPtr = nullptr;
		m_memorySize = m_alignment = 0;
	}
	return;
}

CMemoryBlock::CMemoryBlock()
{
	m_memorySize = m_alignment = 0u;
	m_memoryPtr = nullptr;
}

CMemoryBlock::~CMemoryBlock()
{
	memBlockFree();
}

