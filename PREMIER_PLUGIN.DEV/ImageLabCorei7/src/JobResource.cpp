#include <windows.h>
#include "JobResource.h"

JobResource::JobResource()
{
	coreNumber = 0;
	histogramBufferPtr = nullptr;
	binBufferPtr = nullptr;
	cumSumBufferPtr = nullptr;

	hitogramBufferSize = binBufferSize = cumSumBufferSize = 0ul;
	resourceValid = false;
}

JobResource::JobResource(const unsigned int& coreNum)
{
	coreNumber = coreNum;
	histogramBufferPtr = nullptr;
	binBufferPtr = nullptr;
	cumSumBufferPtr = nullptr;

	hitogramBufferSize = binBufferSize = cumSumBufferSize = 0ul;
	resourceValid = false;
}


JobResource::~JobResource()
{
	freeMemoryResources();
	coreNumber = 0; /* for DBG purpose only */
	resourceValid = false;
}

bool JobResource::allocateHistogramBuffer(unsigned int alignment /* reserved for future usage */)
{
	hitogramBufferSize = 65536 * sizeof(unsigned int); /* 256KB for maximal allowed pixel width size = 16 bits */
	const SIZE_T dwSize = static_cast<SIZE_T>(hitogramBufferSize);
	histogramBufferPtr = reinterpret_cast<unsigned int*>(VirtualAlloc(NULL, dwSize, MEM_RESERVE | MEM_COMMIT | MEM_TOP_DOWN, PAGE_READWRITE));
	return (NULL != histogramBufferPtr) ? true : false;
}

void JobResource::freeHistogramBuffer(void)
{
	if (0u != hitogramBufferSize && NULL != histogramBufferPtr)
	{
		LPVOID lpPtr = reinterpret_cast<LPVOID>(histogramBufferPtr);
		VirtualFree(lpPtr, 0, MEM_RELEASE);
		hitogramBufferSize = 0ul;
		histogramBufferPtr = nullptr;
		lpPtr = nullptr;
	}
	return;
}

bool JobResource::allocateBinBuffer(unsigned int alignment)
{
	binBufferSize = 65536 * sizeof(unsigned char); /* 64KB buffer for histogram binarization */
	const SIZE_T dwSize = static_cast<SIZE_T>(binBufferSize);
	binBufferPtr = reinterpret_cast<unsigned char*>(VirtualAlloc(NULL, dwSize, MEM_RESERVE | MEM_COMMIT | MEM_TOP_DOWN, PAGE_READWRITE));
	return (NULL != binBufferPtr) ? true : false;
}

void JobResource::freeBinBuffer(void)
{
	if (0u != binBufferSize && NULL != binBufferPtr)
	{
		LPVOID lpPtr = reinterpret_cast<LPVOID>(binBufferPtr);
		VirtualFree(lpPtr, 0, MEM_RELEASE);
		binBufferSize = 0ul;
		binBufferPtr = nullptr;
		lpPtr = nullptr;
	}
	return;
}

bool JobResource::allocateCumSumBuffer(unsigned int alignment)
{
	cumSumBufferSize = 65536 * sizeof(unsigned short); /* 128KB buffer for histogram binarization */
	const SIZE_T dwSize = static_cast<SIZE_T>(cumSumBufferSize);
	cumSumBufferPtr = reinterpret_cast<unsigned short*>(VirtualAlloc(NULL, dwSize, MEM_RESERVE | MEM_COMMIT | MEM_TOP_DOWN, PAGE_READWRITE));
	return (NULL != cumSumBufferPtr) ? true : false;
}

void JobResource::freeeCumSumBuffer(void)
{
	if (0u != cumSumBufferSize && NULL != cumSumBufferPtr)
	{
		LPVOID lpPtr = reinterpret_cast<LPVOID>(cumSumBufferPtr);
		VirtualFree(lpPtr, 0, MEM_RELEASE);
		cumSumBufferSize = 0ul;
		cumSumBufferPtr = nullptr;
		lpPtr = nullptr;
	}
	return;
}

bool JobResource::allocateMemoryResources(void)
{
	resourceValid = allocateHistogramBuffer(0) && allocateBinBuffer(0) && allocateCumSumBuffer(0);
	return resourceValid;
}

void JobResource::freeMemoryResources(void)
{
	freeHistogramBuffer();
	freeBinBuffer();
	freeeCumSumBuffer();
	resourceValid = false;
	return;
}
