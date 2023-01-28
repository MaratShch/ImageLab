#include "MemoryBlock.hpp"

using namespace ImageLabMemoryUtils;

CMemoryBlock::CMemoryBlock (void)
{
	m_memoryPtr = nullptr;
	m_memorySize = m_alignment = 0;
#ifdef _DEBUG
	m_usageCounter = 0ull;
#endif
	return;
}


CMemoryBlock::CMemoryBlock (int32_t memSize, int32_t alignment)
{
	m_memoryPtr = nullptr;
	m_memorySize = m_alignment = 0;
#ifdef _DEBUG
	m_usageCounter = 0ull;
#endif
	memBlockAlloc(memSize, alignment);
	return;
}


CMemoryBlock::~CMemoryBlock (void)
{
	memBlockFree();
	return;
}


bool CMemoryBlock::memBlockAlloc(int32_t mSize, int32_t mAlign)
{
	void* p = nullptr;
	bool bRet = false;

	if (0 == mAlign)
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
	if (nullptr != m_memoryPtr)
	{
		if (0 != m_alignment)
			_aligned_free(m_memoryPtr);
		else
			free(m_memoryPtr);

		m_memoryPtr = nullptr;
	}
	m_memorySize = m_alignment = 0;
	return;
}


bool CMemoryBlock::memBlockRealloc (int32_t memSize, int32_t alignment)
{
	memBlockFree();
	return memBlockAlloc(memSize, alignment);
}