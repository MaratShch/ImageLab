#ifndef __NOISE_CLEAN_ALGO_MEMORY_BUFFERS__
#define __NOISE_CLEAN_ALGO_MEMORY_BUFFERS__

#include "CompileTimeUtils.hpp"
#include "CommonAuxPixFormat.hpp"
#include "NoiseClean.hpp"

template <typename T>
inline A_long MemoryBufferAlloc
(
	const A_long&  sizeX,
	const A_long&  sizeY,
	T**   pBuf1,
	T**   pBuf2
) noexcept
{
	constexpr size_t doubleBuffer{ 2 };
	constexpr size_t sizeAlignment{ CACHE_LINE };
	const size_t frameSize = static_cast<size_t>(sizeX * sizeY);
	const size_t requiredMemSize = ::CreateAlignment(frameSize * (sizeof(T) * doubleBuffer), sizeAlignment);
	void* pAlgoMemory = nullptr;

	const A_long blockId = ::GetMemoryBlock(static_cast<int32_t>(requiredMemSize), 0, &pAlgoMemory);

	if (nullptr != pAlgoMemory)
	{
#ifdef _DEBUG
		memset(pAlgoMemory, 0, requiredMemSize);
#endif
		*pBuf1 = static_cast<T*>(pAlgoMemory);
		*pBuf2 = *pBuf1 + frameSize;
	}
	else
		*pBuf1 = *pBuf2 = nullptr;

	return blockId;
}


inline void MemoryBufferRelease
(
	A_long blockId
) noexcept
{
	::FreeMemoryBlock(blockId);
	return;
}


#endif /* __NOISE_CLEAN_ALGO_MEMORY_BUFFERS__ */