#include <windows.h>
#include <stdio.h>
#include <tchar.h>
#include <Windows.h>
#include "AdobeImageLabHDR.h"

static PImageLAB_MemStr pInternalMemory = nullptr;

static bool singleStreamMemAlloc(PImageLAB_MemStr* pMemParamStr)
{
	bool result = false;

	if (nullptr != pMemParamStr)
	{
		PImageLAB_MemStr pMemStr = reinterpret_cast<PImageLAB_MemStr>(_aligned_malloc(sizeof(ImageLAB_MemStr), 32));
		if (nullptr != pMemStr)
		{
			pMemStr->strSizeoF = sizeof(ImageLAB_MemStr);
			pMemStr->parallel_streams = 1;

			pMemStr->pBufPoolHistogram = _aligned_malloc(IMAGE_LAB_HIST_BUFFER_SIZE, 4096);
			if (nullptr != pMemStr->pBufPoolHistogram)
			{
				memset(pMemStr->pBufPoolHistogram, 0, IMAGE_LAB_HIST_BUFFER_SIZE);
			}

			pMemStr->pBufPoolLUT = _aligned_malloc(IMAGE_LAB_LUT_BUFFER_SIZE, 4096);
			if (nullptr != pMemStr->pBufPoolLUT)
			{
				memset(pMemStr->pBufPoolLUT, 0, IMAGE_LAB_LUT_BUFFER_SIZE);
			}

			pMemStr->pBufPoolBinary = _aligned_malloc(IMAGE_LAB_BIN_BUFFER_SIZE, 4096);
			if (nullptr != pMemStr->pBufPoolBinary)
			{
				memset(pMemStr->pBufPoolBinary, 0, IMAGE_LAB_BIN_BUFFER_SIZE);
			}

			*pMemParamStr = pMemStr;
			result = true;
		}
	}

	return result;
}


static bool singleStreamMemFree (PImageLAB_MemStr* pMemParamStr)
{
	bool result = false;

	if (nullptr != pMemParamStr)
	{
		PImageLAB_MemStr pMemStr = *pMemParamStr;
		if (nullptr != pMemStr)
		{
			if (nullptr != pMemStr->pBufPoolHistogram)
			{
				memset(pMemStr->pBufPoolHistogram, 0, IMAGE_LAB_HIST_BUFFER_SIZE);
				_aligned_free(pMemStr->pBufPoolHistogram);
			}

			if (nullptr != pMemStr->pBufPoolLUT)
			{
				memset(pMemStr->pBufPoolLUT, 0, IMAGE_LAB_LUT_BUFFER_SIZE);
				_aligned_free(pMemStr->pBufPoolLUT);
			}

			if (nullptr != pMemStr->pBufPoolBinary)
			{
				memset(pMemStr->pBufPoolBinary, 0, IMAGE_LAB_BIN_BUFFER_SIZE);
				_aligned_free(pMemStr->pBufPoolBinary);
			}

			memset(pMemStr, 0, sizeof(*pMemStr));
			_aligned_free(pMemStr);
			pMemStr = nullptr;
		}
	}

	return result;
}

void* APIENTRY GetStreamMemory(void)
{
	return reinterpret_cast<void*>(pInternalMemory);
}

void* APIENTRY GetHistogramBuffer(void)
{
	void* pHist = nullptr;

	if (nullptr != pInternalMemory)
	{
		if (nullptr != pInternalMemory->pBufPoolHistogram)
		{
			pHist = pInternalMemory->pBufPoolHistogram;
		}
	}

	return pHist;
}


void* APIENTRY GetBinarizationBuffer(void)
{
	void* pBuf = nullptr;

	if (nullptr != pInternalMemory)
	{
		if (nullptr != pInternalMemory->pBufPoolBinary)
		{
			pBuf = pInternalMemory->pBufPoolBinary;
		}
	}

	return pBuf;
}

void* APIENTRY GetLUTBuffer(void)
{
	void* pBuf = nullptr;

	if (nullptr != pInternalMemory)
	{
		if (nullptr != pInternalMemory->pBufPoolLUT)
		{
			pBuf = pInternalMemory->pBufPoolLUT;
		}
	}

	return pBuf;
}


BOOL APIENTRY DllMain(HMODULE /* hModule */, DWORD ul_reason_for_call, LPVOID /* lpReserved */)
{
	
	switch (ul_reason_for_call)
    {
		case DLL_PROCESS_ATTACH:
			singleStreamMemAlloc(&pInternalMemory);
		break;

		case DLL_THREAD_ATTACH:
		break;

		case DLL_THREAD_DETACH:
		break;

		case DLL_PROCESS_DETACH:
			if (nullptr != pInternalMemory)
			{
				singleStreamMemFree(&pInternalMemory);
				pInternalMemory = nullptr;
			}
		break;

		default:
		break;
    }

    return TRUE;
}

