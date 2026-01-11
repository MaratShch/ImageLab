#include "AlgoMemHandler.hpp"
#include "CompileTimeUtils.hpp"


MemHandler alloc_memory_buffers (int32_t sizeX, int32_t sizeY, const bool dbgPrn)
{
	MemHandler algoMemHandler{};
	const int32_t frameSize = sizeX * sizeY;
	const int32_t LumaBytesSize = static_cast<int32_t>(frameSize * sizeof(float));
	const int32_t alignedFloat = CreateAlignment (LumaBytesSize, static_cast<int32_t>(CACHE_LINE));
	const int32_t alignedFloatx2 = alignedFloat * 2;

	return algoMemHandler;
}


void free_memory_buffers (MemHandler& algoMemHandler)
{
	return;
}