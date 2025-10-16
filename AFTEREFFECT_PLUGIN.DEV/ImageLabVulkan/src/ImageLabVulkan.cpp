#include "ImageLabVulkan.hpp"
#include "ImageLabVulkanHandler.hpp"
#include "VulkanDevicePrivate.hpp"

/* 
core policy:
    0 - any
    1 - low performance allowed
    2 - medium performance allowed
    3 - high performance only

memory policy:
    0 - any
    <N> - number of GB memory avaialble for GPU
*/

DLL_API_EXPORT ILVulkanHndl CreateVulkanContext (uint32_t core, uint32_t memory, uint32_t reserved)
{
    ILVulkanHandler* vkHndl = new ILVulkanHandler;
    if (nullptr != vkHndl)
    {
        vkHndl->strSizeof    = ILVulkanHandlerSize;
        vkHndl->hndlVersion  = ILVulkanHandlerVersion;
        vkHndl->deviceNumber = setDeviceNodeIdx (core, memory, reserved);
        vkHndl->vkInstance = GetVulkanInstance();
        vkHndl->vkPhysicalDevice = getDeviceArray()[vkHndl->deviceNumber];
    }

    return vkHndl;
}

DLL_API_EXPORT void FreeVulkanContext (ILVulkanHndl vkHndl)
{
    if (nullptr != vkHndl)
    {
        ILVulkanHandler* hndlStrP = reinterpret_cast<ILVulkanHandler*>(vkHndl);
        // verify structure fields
        if (sizeof(*hndlStrP) == hndlStrP->strSizeof && ILVulkanHandlerVersion == hndlStrP->hndlVersion)
        {

            resetDeviceNodeIdx(hndlStrP->deviceNumber);
        }

        delete hndlStrP;
        hndlStrP = nullptr;
        vkHndl = nullptr;
    }
    return;
}