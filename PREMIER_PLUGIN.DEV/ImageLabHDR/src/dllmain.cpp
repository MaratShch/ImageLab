#include <windows.h>

static int numberCpuCores;

int GetCpuCoresNumber(void)
{
	return numberCpuCores;
}


BOOL APIENTRY DllMain(HMODULE /* hModule */, DWORD ul_reason_for_call, LPVOID /* lpReserved */)
{
	SYSTEM_INFO sysinfo = {0};
	numberCpuCores = 0;

    switch (ul_reason_for_call)
    {
		case DLL_PROCESS_ATTACH:
			GetSystemInfo(&sysinfo);
			numberCpuCores = sysinfo.dwNumberOfProcessors;
		break;

		case DLL_THREAD_ATTACH:
		case DLL_THREAD_DETACH:
		case DLL_PROCESS_DETACH:
		break;

		default:
		break;
    }
    return TRUE;
}
