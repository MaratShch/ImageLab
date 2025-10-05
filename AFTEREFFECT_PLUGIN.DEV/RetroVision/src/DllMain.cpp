#include <windows.h>

static HMODULE hLib = nullptr;

void LoadResourceDll(void)
{
    hLib = ::LoadLibraryEx(__TEXT("ImageLabResource.dll"), NULL, LOAD_LIBRARY_AS_DATAFILE | LOAD_LIBRARY_SEARCH_DEFAULT_DIRS);
}

void FreeResourceDll(void)
{
    if (nullptr != hLib)
    {
        ::FreeLibrary(hLib);
        hLib = nullptr;
    }
}

HMODULE GetResourceLibHandler(void)
{
    return hLib;
}



BOOL WINAPI DllMain
(
    HINSTANCE hinstDLL,  // handle to DLL module
    DWORD fdwReason,     // reason for calling function
    LPVOID lpvReserved   // reserved
)
{
    // Perform actions based on the reason for calling.
    switch (fdwReason)
    {
        case DLL_PROCESS_ATTACH:
            hLib = nullptr;
            LoadResourceDll();
            DisableThreadLibraryCalls(hinstDLL);
        break;

        case DLL_THREAD_ATTACH:
        break;

        case DLL_THREAD_DETACH:
        break;

        case DLL_PROCESS_DETACH:
            FreeResourceDll();
        break;
    } // switch (fdwReason)

    return TRUE;
}