#include <windows.h>
#include <mutex>
#include <thread>
#include "ImageLabAnisotropicDiffusion.h"

std::mutex algStorageMutex;
CACHE_ALIGN static AlgMemStorage algStorage;


AlgMemStorage& getAlgStorageStruct (void)
{
	std::lock_guard<std::mutex> guard(algStorageMutex);
	return algStorage;
}

void setAlgStorageStruct (const AlgMemStorage& storage)
{
	std::lock_guard<std::mutex> guard(algStorageMutex);
	algStorage = storage;
	return;
}


BOOL APIENTRY DllMain(HMODULE /* hModule */, DWORD ul_reason_for_call, LPVOID /* lpReserved */)
{
	switch (ul_reason_for_call)
	{
		case DLL_PROCESS_ATTACH:
			memset (&algStorage, 0, sizeof(algStorage));
		break;

		case DLL_THREAD_ATTACH:
		break;

		case DLL_THREAD_DETACH:
		break;

		case DLL_PROCESS_DETACH:
			algMemStorageFree(algStorage);
		break;

		default:
		break;
	}

	return TRUE;
}
