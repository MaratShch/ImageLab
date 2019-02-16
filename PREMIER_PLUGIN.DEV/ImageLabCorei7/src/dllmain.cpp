#include <windows.h>
#include <stdio.h>
#include <tchar.h>
#include "ImageLabHDR.h"

static DWORD tlsIdx;

extern "C" {
	DWORD WINAPI __imp_TlsAlloc() {
		return FlsAlloc(nullptr);
	}
	BOOL WINAPI __imp_TlsFree(DWORD index) {
		return FlsFree(index);
	}
	BOOL WINAPI __imp_TlsSetValue(DWORD dwTlsIndex, LPVOID lpTlsValue) {
		return FlsSetValue(dwTlsIndex, lpTlsValue);
	}
	LPVOID WINAPI __imp_TlsGetValue(DWORD dwTlsIndex) {
		return FlsGetValue(dwTlsIndex);
	}
}

BOOL APIENTRY DllMain(HMODULE /* hModule */, DWORD ul_reason_for_call, LPVOID /* lpReserved */)
{
	PImageLabHDR_SystemMemoryBlock pMemoryBlock = nullptr;
	
	switch (ul_reason_for_call)
    {
		case DLL_PROCESS_ATTACH:
//			tlsIdx = TlsAlloc();
//			if (TLS_OUT_OF_INDEXES == tlsIdx)
//				return FALSE;
			// no break!!!	
		break;

		case DLL_THREAD_ATTACH:
//			if (true == ImageLabHDR_AllocSystemMemory(&pMemoryBlock))
//			{
//				TlsSetValue(tlsIdx, pMemoryBlock);
//			}
		break;

		case DLL_THREAD_DETACH:
//			pMemoryBlock = reinterpret_cast<PImageLabHDR_SystemMemoryBlock>(TlsGetValue(tlsIdx));
//			if (nullptr != pMemoryBlock)
//			{
//				TlsSetValue(tlsIdx, nullptr);
//				ImageLabHDR_FreeSystemMemory(pMemoryBlock);
//				pMemoryBlock = nullptr;
//			}
		break;

		case DLL_PROCESS_DETACH:
//			pMemoryBlock = reinterpret_cast<PImageLabHDR_SystemMemoryBlock>(TlsGetValue(tlsIdx));
//			if (nullptr != pMemoryBlock)
//			{
//				TlsSetValue(tlsIdx, nullptr);
//				ImageLabHDR_FreeSystemMemory(pMemoryBlock);
//				pMemoryBlock = nullptr;
//			}
//			TlsFree(tlsIdx);
//			tlsIdx = ULONG_MAX;
		break;

		default:
		break;
    }

    return TRUE;
}

