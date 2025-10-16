#include <string>
#include <memory>
#include <atomic>
#include "ImageLabVulkanLoader.hpp"
#include "PrismaVulkan.hpp"
#include "CommonAdobeAE.hpp"

static HMODULE hLib = nullptr;
static PrismaAlgoVulkanHandler PrismaAlgoHandler{};

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

            PrismaAlgoHandler.getVulkanVersion      = reinterpret_cast<GetVulkanVersion1>(GetProcAddress(hLib, __TEXT("GetVulkanVersion")));
            PrismaAlgoHandler.createVulkanContext   = reinterpret_cast<CreateVulkanContext1>(GetProcAddress(hLib, __TEXT("CreateVulkanContext")));
            PrismaAlgoHandler.freeVulkanContext     = reinterpret_cast<FreeVulkanContext1> (GetProcAddress(hLib, __TEXT("FreeVulkanContext")));
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
        memset(&PrismaAlgoHandler, 0, sizeof(PrismaAlgoHandler));
        hLib = nullptr;
    }
    return;
}

uint32_t getVulkanVersionNumber (void)
{
    return (nullptr != hLib && nullptr != PrismaAlgoHandler.getVulkanVersion) ? PrismaAlgoHandler.getVulkanVersion() : 0xFFFFFFFFu;
}


void* createVulkanContext (uint32_t proc, uint32_t mem, uint32_t reserved)
{
    return (nullptr != hLib && nullptr != PrismaAlgoHandler.createVulkanContext) ? PrismaAlgoHandler.createVulkanContext (proc, mem, reserved) : nullptr;
}

void freeVulkanContext (void* pHndl)
{
    if (nullptr != pHndl && nullptr != hLib && nullptr != PrismaAlgoHandler.freeVulkanContext)
        PrismaAlgoHandler.freeVulkanContext (pHndl);
    return;
}