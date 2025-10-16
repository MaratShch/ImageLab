#include "ImageLabVulkan.hpp"
#include "ImageLabVulkanHandler.hpp"

DLL_API_EXPORT ILVulkanHndl AllocVulkanNode (uint32_t core, uint32_t memory, uint32_t reserved)
{
    ILVulkanHandler* vkHndl = new ILVulkanHandler;
    if (nullptr != vkHndl)
    {
        vkHndl->strSizeof   = ILVulkanHandlerSize;
        vkHndl->hndlVersion = ILVulkanHandlerVersion;
        vkHndl->vkInstance  = GetVulkanInstance();
    }

    return vkHndl;
}

DLL_API_EXPORT void FreeVulkanNode (ILVulkanHndl vkHndl)
{
    if (nullptr != vkHndl)
    {
        ILVulkanHandler* hndlStrP = reinterpret_cast<ILVulkanHandler*>(vkHndl);
        // verify structure fields
        if (sizeof(*hndlStrP) == hndlStrP->strSizeof && ILVulkanHandlerVersion == hndlStrP->hndlVersion)
        {

            // cleanup before memory free
            memset(hndlStrP, 0, sizeof(*hndlStrP));
        }

        delete vkHndl;
        vkHndl = nullptr;
    }
    return;
}