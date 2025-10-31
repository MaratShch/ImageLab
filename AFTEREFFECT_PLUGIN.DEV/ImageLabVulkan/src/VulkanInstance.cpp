#include "VulkanBinding.hpp"

std::mutex gGlobalProtect;
std::atomic<uint32_t> gInstanceRefCnt{0u};
std::atomic<VkInstance> gInstance{nullptr};

constexpr uint32_t gMaxGpuSupport = 16u;
std::array<VulkanGPU::CVulkanGpuContext*, gMaxGpuSupport> gGpuGlobalMap{};

void CleanupOnDllLoad (void)
{
    for (uint32_t i = 0u; i < gMaxGpuSupport; i++)
        gGpuGlobalMap[i] = nullptr;
}


DLL_API_EXPORT bool CreateVulkanInstance (void)
{
    bool bRes = false;
    VkInstance instance = static_cast<VkInstance>(VK_NULL_HANDLE);

    VkApplicationInfo appInfo{};
    appInfo.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    appInfo.pApplicationName = "VulkanBinding";
    appInfo.apiVersion = VK_API_VERSION_1_3;

    VkInstanceCreateInfo createInfo{};
    createInfo.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    createInfo.pApplicationInfo = &appInfo;

    gInstanceRefCnt++;

    if (static_cast<VkInstance>(VK_NULL_HANDLE) == gInstance)
    {
        std::lock_guard<std::mutex> lk(gGlobalProtect);
        {
            if (static_cast<VkInstance>(VK_NULL_HANDLE) == gInstance)
            {
                VkResult res = vkCreateInstance(&createInfo, nullptr, &instance);
                if (VK_SUCCESS == res && static_cast<VkInstance>(VK_NULL_HANDLE) != instance)
                {
                    gInstance = instance;

                    //scan GPU's number
                    uint32_t deviceCount = 0u;
                    vkEnumeratePhysicalDevices(instance, &deviceCount, nullptr);
                    if (0u != deviceCount)
                    {
                        std::vector<VkPhysicalDevice> physDevices(deviceCount);
                        vkEnumeratePhysicalDevices(instance, &deviceCount, physDevices.data());

                        if (deviceCount >= gMaxGpuSupport)
                            deviceCount = gMaxGpuSupport;

                        for (uint32_t i = 0u; i < deviceCount; i++)
                        {
                            gGpuGlobalMap[i] = new VulkanGPU::CVulkanGpuContext(i, physDevices[i]);
                            
                            uint32_t count = 0;
                            vkEnumerateDeviceExtensionProperties (physDevices[i], nullptr, &count, nullptr);
                            std::vector<VkExtensionProperties> exts(count);
                            vkEnumerateDeviceExtensionProperties (physDevices[i], nullptr, &count, exts.data());
                            gGpuGlobalMap[i]->fill_extension_properties (exts);
                        }

                        bRes = true;
                    }
                    else
                        bRes = false;
                }
                else
                    bRes = false;
            }
            else
                bRes = true;
        }
    }
    else
        bRes = true;

    return bRes;
}


DLL_API_EXPORT void DeleteVulkanInstance (void)
{
    std::lock_guard<std::mutex> lk(gGlobalProtect);
    {
        if (gInstanceRefCnt > 0u)
            gInstanceRefCnt--;

        if (static_cast<VkInstance>(VK_NULL_HANDLE) != gInstance && 0u == gInstanceRefCnt)
        {
            for (uint32_t i = 0u; i < gMaxGpuSupport; i++)
            {
                if (nullptr != gGpuGlobalMap[i])
                {
                    delete gGpuGlobalMap[i];
                    gGpuGlobalMap[i] = nullptr;
                }
            }
            VkInstance instance = gInstance.exchange(static_cast<VkInstance>(VK_NULL_HANDLE));
            vkDestroyInstance (instance, nullptr);
            instance = static_cast<VkInstance>(VK_NULL_HANDLE);
        }
    }
    return;
}