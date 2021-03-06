#include <windows.h>
#include <stdio.h>
#include <tchar.h>
#include "ImageLabColorCorrectionHSL.h"


static filterMemoryHandle fMemHndl{ 0 };


filterMemoryHandle* get_tmp_memory_handler (void)
{
	filterMemoryHandle* pMem = &fMemHndl;
	return pMem;
}

void* get_tmp_buffer (size_t* pBufBytesSize)
{
	if (nullptr != pBufBytesSize)
		*pBufBytesSize = fMemHndl.tmpBufferSizeBytes;
	return fMemHndl.tmpBufferAlignedPtr;
}

void set_tmp_buffer (void* __restrict pBuffer, const size_t& bufBytesSize)
{
	fMemHndl.tmpBufferAlignedPtr = pBuffer;
	fMemHndl.tmpBufferSizeBytes = bufBytesSize;
	return;
}


BOOL APIENTRY DllMain(HMODULE /* hModule */, DWORD ul_reason_for_call, LPVOID /* lpReserved */)
{
	switch (ul_reason_for_call)
	{
		case DLL_PROCESS_ATTACH:
			fMemHndl.tmpBufferAlignedPtr = nullptr;
			fMemHndl.tmpBufferSizeBytes = 0;
		break;

		case DLL_THREAD_ATTACH:
		break;

		case DLL_THREAD_DETACH:
		break;

		case DLL_PROCESS_DETACH:
			free_aligned_buffer (&fMemHndl);
			fMemHndl.tmpBufferAlignedPtr = nullptr;
			fMemHndl.tmpBufferSizeBytes = 0;
		break;
	
		default:
		break;
	}

	return TRUE;
}

