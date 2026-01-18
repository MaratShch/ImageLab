#ifndef __IMAGE_LAB_ART_POINTILISM_MEMORY_BUFFERS_HANDLER__
#define __IMAGE_LAB_ART_POINTILISM_MEMORY_BUFFERS_HANDLER__

#include <cstdint>
#include "CommonAuxPixFormat.hpp"
#include "AlgoDotEngine.hpp"
#include "AlgoJFA.hpp"
#include "AlgoArtisticsRendering.hpp"

struct MemHandler
{
    int64_t memBlockId;
	uint8_t* SuperBufferHead;
	float* L;
	float* ab;
	float* Luma1;
	float* Luma2;
	float* DencityMap;
	FlatQuadNode* NodePool;
	Point2D* PointOut;
	JFAPixel* JfaBufferPing;
	JFAPixel* JfaBufferPong;
	float* AccumX;
	float* AccumY;
	float* AccumW;
	RenderScratchMemory Scratch;
	float* CanvasLab;
	float* dst_L;
	float* dst_ab;
	int32_t NodeElemNumber;
};


MemHandler alloc_memory_buffers (int32_t sizeX, int32_t sizeY, const bool dbgPrn = false);
void free_memory_buffers (MemHandler& algoMemHandler);

inline bool mem_handler_valid(const MemHandler& hndl) noexcept
{
    return (hndl.memBlockId >= 0 && hndl.SuperBufferHead != nullptr) ? true : false;
}

#endif // __IMAGE_LAB_ART_POINTILISM_MEMORY_BUFFERS_HANDLER__ 