#include <string>
#include <memory>
#include <atomic>
#include "ImageLabVulkanLoader.hpp"
#include "CommonAdobeAE.hpp"

static HMODULE hLib = nullptr;

bool LoadVulkanAlgoDll (PF_InData* in_data)
{
    A_char pluginFullPath[AEFX_MAX_PATH]{};
    PF_Err extErr = PF_GET_PLATFORM_DATA(PF_PlatData_EXE_FILE_PATH_DEPRECATED, &pluginFullPath);
    bool err = false;

    if (PF_Err_NONE == extErr && '\0' != pluginFullPath[0])
    {
        const std::string dllName{ "\\ImageLabVulkan.dll" };
        const std::string aexPath{ pluginFullPath };
        const std::string::size_type pos = aexPath.rfind("\\", aexPath.length());
        const std::string dllPath = aexPath.substr(0, pos) + dllName;

        // Load Memory Management DLL
        hLib = ::LoadLibraryEx(dllPath.c_str(), NULL, LOAD_LIBRARY_SEARCH_DEFAULT_DIRS);
        if (NULL != hLib)
        {
            ::DisableThreadLibraryCalls (hLib);
            err = true;
        }
    }

    return err;
}


void UnloadVulkanAlgoDll (void)
{
    if (nullptr != hLib)
    {
        ::FreeLibrary (hLib);
        hLib = nullptr;
    }
    return;
}