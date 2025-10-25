#include <windows.h>
#include "VulkanBinding.hpp"

BOOL WINAPI DllMain
(
    HINSTANCE hinstDLL,  // handle to DLL module
    DWORD fdwReason,     // reason for calling function
    LPVOID lpvReserved   // reserved
)
{
    (void)lpvReserved;

    // Perform actions based on the reason for calling.
    switch (fdwReason)
    {
        case DLL_PROCESS_ATTACH:
            CleanupOnDllLoad ();
            ::DisableThreadLibraryCalls(hinstDLL);
        break;

        case DLL_THREAD_ATTACH:
        break;

        case DLL_THREAD_DETACH:
        break;

        case DLL_PROCESS_DETACH:
        break;
    } // switch (fdwReason)

    return TRUE;
}
