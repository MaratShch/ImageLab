#include "AutomaticWhiteBalance.hpp"
#include "AlgMemoryHandler.hpp"
#include <windows.h>

#ifdef _DEBUG
#pragma comment(lib, "..\\BUILD.OUT\\Debug\\ImageLabUtils.lib")
#else 
#pragma comment(lib, "..\\BUILD.OUT\\Release\\ImageLabUtils.lib")
#endif

static CAlgMemHandler* pAlgMemHandler = nullptr;

CAlgMemHandler* getMemoryHandler(void) { return pAlgMemHandler; }


BOOL APIENTRY DllMain (HMODULE /* hModule */, DWORD ul_reason_for_call, LPVOID /* lpReserved */)
{
	switch (ul_reason_for_call)
	{
		case DLL_PROCESS_ATTACH:
		// allocate memory buffers for hold temporary processing result
			pAlgMemHandler = new CAlgMemHandler;
		break;

		case DLL_THREAD_ATTACH:
		break;

		case DLL_THREAD_DETACH:
		break;

		case DLL_PROCESS_DETACH:
			// free memory buffers
			delete pAlgMemHandler;
			pAlgMemHandler = nullptr;
		break;

		default:
		break;
	}

	return TRUE;
}