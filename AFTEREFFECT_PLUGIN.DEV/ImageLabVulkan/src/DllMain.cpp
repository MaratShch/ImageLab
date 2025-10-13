#include <windows.h>
#include "ImageLabVulkanHandler.hpp"


BOOL WINAPI DllMain
(
    HINSTANCE hinstDLL,  // handle to DLL module
    DWORD fdwReason,     // reason for calling function
    LPVOID lpvReserved   // reserved
)
{
    BOOL loadResult = TRUE;
    (void)lpvReserved;

    // Perform actions based on the reason for calling.
    switch (fdwReason)
    {
        case DLL_PROCESS_ATTACH:
            ::DisableThreadLibraryCalls(hinstDLL);
            if (false == InitVulkanFramework())
                loadResult = FALSE;
        break;

        case DLL_THREAD_ATTACH:
        break;

        case DLL_THREAD_DETACH:
        break;

        case DLL_PROCESS_DETACH:
            CleanupVulkanFramework();
        break;
    } // switch (fdwReason)

    return loadResult;
}
