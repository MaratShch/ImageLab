#include <windows.h>
#include "ImageLabUtils.hpp"


BOOL APIENTRY DllMain (HMODULE /* hModule */, DWORD ul_reason_for_call, LPVOID /* lpReserved */)
{
	switch (ul_reason_for_call)
	{
		case DLL_PROCESS_DETACH:
			ReleaseMemoryHandler(nullptr);
		break;

		case DLL_PROCESS_ATTACH:
			/* singletone memory interface object */
			(void)CreateMemoryHandler();
		break;

		case DLL_THREAD_ATTACH:
		break;

		case DLL_THREAD_DETACH:
		break;

		default:
		break;
	}

	return TRUE;
}