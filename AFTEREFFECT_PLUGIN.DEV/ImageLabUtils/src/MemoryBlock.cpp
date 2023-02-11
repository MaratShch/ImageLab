#include "MemoryBlock.hpp"

using namespace ImageLabMemoryUtils;

CMemoryBlock::CMemoryBlock (void)
{
	m_memoryPtr = nullptr;
	m_memorySize = m_alignment = 0;
	return;
}


CMemoryBlock::CMemoryBlock (int32_t memSize, int32_t alignment)
{
	m_memoryPtr = nullptr;
	m_memorySize = m_alignment = 0;
	memBlockAlloc(memSize, alignment);
	return;
}


CMemoryBlock::~CMemoryBlock (void)
{
	memBlockFree();
	return;
}


void* CMemoryBlock::memBlockAlloc(int32_t mSize, int32_t mAlign)
{
	void* p = nullptr;

	if (0 == mAlign)
		p = malloc(static_cast<size_t>(mSize));
	else
		p = _aligned_malloc(static_cast<size_t>(mSize), static_cast<size_t>(mAlign));

	if (nullptr != p)
	{
		m_memoryPtr = p;
		m_memorySize = mSize;
		m_alignment = mAlign;
	}
	return m_memoryPtr;
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


void* CMemoryBlock::memBlockRealloc (int32_t memSize, int32_t alignment)
{
	memBlockFree();
	return memBlockAlloc(memSize, alignment);
}