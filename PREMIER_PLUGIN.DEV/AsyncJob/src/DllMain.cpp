#include "CCommon.hpp"
#include "CThreadPool.hpp"

CACHE_ALIGN CThreadPool _threadPool;
__declspec(dllexport) CThreadPool& GetThreadPool(void) { return _threadPool; }

BOOL APIENTRY DllMain(HMODULE /* hModule */, DWORD ul_reason_for_call, LPVOID /* lpReserved */)
{
	switch (ul_reason_for_call)
	{
		case DLL_PROCESS_ATTACH:
			_threadPool.init();
		break;

		case DLL_THREAD_ATTACH:
		break;

		case DLL_THREAD_DETACH:
		break;

		case DLL_PROCESS_DETACH:
		break;

		default:
		break;
	}

	return TRUE;
}
