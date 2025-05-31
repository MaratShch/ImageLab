#include "ImageLabQTBinding.hpp"
#include "ApplicationHandler.hpp"
#include <windows.h>

static ImageLabGUI::CGUIInterface* guiInstance = nullptr;
static QApplication* app = nullptr;

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
            DisableThreadLibraryCalls(hinstDLL);
            guiInstance = ImageLabGUI::CGUIInterface::getInstance();
            if (nullptr != guiInstance)
                app = AllocQTApplication();
        break;

        case DLL_THREAD_ATTACH:
        break;

        case DLL_THREAD_DETACH:
        break;

        case DLL_PROCESS_DETACH:
            ImageLabGUI::DeleteQTApplication();
            guiInstance = nullptr;
            app = nullptr;
        break;
    } // switch (fdwReason)

    return TRUE;
}