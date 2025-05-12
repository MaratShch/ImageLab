#include <string>
#include <memory>
#include <atomic>
#include "ImageLabMemInterface.hpp"
#include "CommonAdobeAE.hpp"

static HMODULE hLib = nullptr;
static void* MemoryInterfaceHndl = nullptr;
static MemoryManagerInterface memInterface{};


bool LoadMemoryInterfaceProvider(PF_InData* in_data)
{
    A_char pluginFullPath[AEFX_MAX_PATH]{};
    PF_Err extErr = PF_GET_PLATFORM_DATA(PF_PlatData_EXE_FILE_PATH_DEPRECATED, &pluginFullPath);
    bool err = false;

    if (PF_Err_NONE == extErr && 0 != pluginFullPath[0])
    {
        const std::string dllName{ "\\ImageLabUtils.dll" };
        const std::string aexPath{ pluginFullPath };
        const std::string::size_type pos = aexPath.rfind("\\", aexPath.length());
        const std::string dllPath = aexPath.substr(0, pos) + dllName;

        // Load Memory Management DLL
        hLib = ::LoadLibraryEx(dllPath.c_str(), NULL, LOAD_LIBRARY_SEARCH_DEFAULT_DIRS);
        memInterface._dbgLastError = ::GetLastError();
        if (NULL != hLib)
        {
            DisableThreadLibraryCalls(hLib);
            memInterface.MemoryInterfaceOpen = reinterpret_cast<OpenMemInterface>    (GetProcAddress(hLib, __TEXT("CreateMemoryHandler")));
            memInterface.MemoryInterfaceClose = reinterpret_cast<CloseMemInterface>  (GetProcAddress(hLib, __TEXT("ReleaseMemoryHandler")));
            memInterface.MemoryInterfaceAllocBlock = reinterpret_cast<AllocMemBlock> (GetProcAddress(hLib, __TEXT("AllocMemoryBlock")));
            memInterface.MemoryInterfaceReleaseBlock = reinterpret_cast<FreeMemBlock>(GetProcAddress(hLib, __TEXT("ReleaseMemoryBlock")));

            if (nullptr != memInterface.MemoryInterfaceOpen)
            {
                // open memory interface handler
                MemoryInterfaceHndl = memInterface.MemoryInterfaceOpen();
                memInterface._dbgLastError = ::GetLastError();
                err = true;
            } // if (nullptr != memInterface.MemoryInterfaceOpen)
        } // if (NULL != hLib)
    } // if (PF_Err_NONE == extErr && 0 != pluginFullPath[0])

    return err;
}


int32_t GetMemoryBlock(int32_t size, int32_t align, void** pMem) noexcept
{
    if (NULL != hLib && NULL != memInterface.MemoryInterfaceAllocBlock && nullptr != MemoryInterfaceHndl && nullptr != pMem)
        return memInterface.MemoryInterfaceAllocBlock(MemoryInterfaceHndl, size, align, pMem);
    return -1;
}

void FreeMemoryBlock(int32_t id) noexcept
{
    if (NULL != hLib && NULL != memInterface.MemoryInterfaceReleaseBlock && nullptr != MemoryInterfaceHndl && id >= 0)
        memInterface.MemoryInterfaceReleaseBlock(MemoryInterfaceHndl, id);
    id = -1;
    return;
}

A_long memGetLastError(void) noexcept
{
    return static_cast<A_long>(memInterface._dbgLastError);
}


void UnloadMemoryInterfaceProvider(void)
{
    if (nullptr != hLib)
    {
        ::FreeLibrary(hLib);
        hLib = nullptr;

        MemoryInterfaceHndl = nullptr;
        memset(&memInterface, 0, sizeof(memInterface));
    }
    return;
}