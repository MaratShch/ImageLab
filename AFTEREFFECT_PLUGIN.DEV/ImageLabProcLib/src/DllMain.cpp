#include <windows.h>


BOOL WINAPI DllMain
(
    HINSTANCE hinstDLL,  // handle to DLL module
    DWORD fdwReason,     // reason for calling function
    LPVOID lpvReserved   // reserved
)
{
	switch (fdwReason)
	{
		case DLL_PROCESS_DETACH:
		break;

		case DLL_PROCESS_ATTACH:
            DisableThreadLibraryCalls(hinstDLL);
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

