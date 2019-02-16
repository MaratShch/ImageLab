#include <Windows.h>
#include "AdobeImageLabHDR.h"
#include "ImageLabHDR.h"


bool ImageLabHDR_AllocSystemMemory(PImageLabHDR_SystemMemoryBlock* ppBlock)
{
	PImageLabHDR_SystemMemoryBlock pBlock = nullptr;
	int i;
	int validHistBuffers = 0;
	bool err = false;

	if (nullptr != ppBlock)
	{
		*ppBlock = nullptr;

		pBlock = new ImageLabHDR_SystemMemoryBlock;
		memset(pBlock, 0, sizeof(*pBlock));

		// fill strcture fields
		pBlock->strSizeOf = sizeof(*pBlock);
		const SIZE_T mainBufferSize = (IMAGE_LAB_HDR_MAX_IMAGE_WIDTH) *
									  (IMAGE_LAB_HDR_MAX_IMAGE_HEIGHT) *
									  3 *  // 3 channels
									  sizeof(unsigned short); // maximal pixel width from single channel - 16 bits 

		// allocate main operation buffer
		LPVOID pMainBuffer = VirtualAlloc(NULL, mainBufferSize, MEM_RESERVE | MEM_COMMIT | MEM_TOP_DOWN, PAGE_READWRITE);
		if (nullptr != pMainBuffer)
		{
			pBlock->mainBufferBytesSize = mainBufferSize;
			pBlock->pMainBuffer = pMainBuffer;
		}

		// allocate sub-buffers for histograms
		const SIZE_T historamBufferSize = 65536 * sizeof(int);

		for (i = 0; i < IMAGE_LAB_HDR_HIST_AVERAGE_DEPTH_MAX; i++)
		{
			LPVOID pHistogramBuffer = VirtualAlloc(NULL, historamBufferSize, MEM_RESERVE | MEM_COMMIT | MEM_TOP_DOWN, PAGE_READWRITE);
			if (nullptr == pHistogramBuffer)
				break;

			pBlock->pHistogramSequence[i] = reinterpret_cast<int*>(pHistogramBuffer);
			validHistBuffers++;
		}

		pBlock->histogramBlocks = validHistBuffers;

		*ppBlock = pBlock;
		err = true;
	}

	return err;
}

void ImageLabHDR_FreeSystemMemory(PImageLabHDR_SystemMemoryBlock pBlock)
{
	if (nullptr != pBlock)
	{
		if (nullptr != pBlock->pMainBuffer)
		{
			VirtualFree(pBlock->pMainBuffer, 0, MEM_RELEASE);
			pBlock->pMainBuffer = nullptr;
			pBlock->mainBufferBytesSize = 0;
		}

		for (int i = 0; i < IMAGE_LAB_HDR_HIST_AVERAGE_DEPTH_MAX; i++)
		{
			LPVOID pHistogramBuffer = reinterpret_cast<LPVOID>(pBlock->pHistogramSequence[i]);
			pBlock->pHistogramSequence[i] = nullptr;
			if (nullptr != pHistogramBuffer)
			{
				VirtualFree(pHistogramBuffer, 0, MEM_RELEASE);
				pHistogramBuffer = nullptr;
			}
		}
		pBlock->histogramBlocks = 0;
		pBlock->strSizeOf = 0;

		delete pBlock;
		pBlock = nullptr;
	}

	return;
}