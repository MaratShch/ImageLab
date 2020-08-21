#include "ImageLabSketch.h"


CACHE_ALIGN static AlgMemStorage algStorage;


AlgMemStorage* getAlgStorageStruct (void)
{
	return &algStorage;
}


BOOL APIENTRY DllMain(HMODULE /* hModule */, DWORD ul_reason_for_call, LPVOID /* lpReserved */)
{
	switch (ul_reason_for_call)
	{
		case DLL_PROCESS_ATTACH:
			memset(&algStorage, 0, sizeof(algStorage));
		break;

		case DLL_THREAD_ATTACH:
		break;

		case DLL_THREAD_DETACH:
		break;

		case DLL_PROCESS_DETACH:
			algMemStorageFree (&algStorage);
			memset(&algStorage, 0, sizeof(algStorage));
		break;

		default:
		break;
	}

	return TRUE;
}
