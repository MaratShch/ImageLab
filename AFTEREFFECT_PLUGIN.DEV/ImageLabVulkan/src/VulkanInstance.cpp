#include "ImageLabVulkanHandler.hpp"

constexpr uint32_t vulkanFrameworkVersion = VK_HEADER_VERSION_COMPLETE;

static VkInstance gImageLabVkInstance    { VK_NULL_HANDLE };
static bool g_VulkanAvailable = false;


VkInstance getVulkanInstance (void)
{
    return gImageLabVkInstance;
}

bool IsVulkanAvailable (void)
{
    return g_VulkanAvailable;
}



bool InitVulkanFramework (void)
{
    bool bVulkanInstanceResult = false;

    VkApplicationInfo appInfo{};
    appInfo.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    appInfo.pApplicationName = "ImageLab2 Vulkan Interface";
    appInfo.apiVersion = VK_API_VERSION_1_3;

    VkInstanceCreateInfo instanceInfo{};
    instanceInfo.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    instanceInfo.pApplicationInfo = &appInfo;

    VkResult res = vkCreateInstance (&instanceInfo, nullptr, &gImageLabVkInstance);
    if (VK_SUCCESS == res)
    {
        uint32_t gpuCount = 0u;
        vkEnumeratePhysicalDevices (gImageLabVkInstance, &gpuCount, nullptr);
        if (0u != gpuCount)
        {
            std::vector<VkPhysicalDevice> devices(gpuCount);
            vkEnumeratePhysicalDevices (gImageLabVkInstance, &gpuCount, devices.data());
            fillDeviceVector(devices);

            bVulkanInstanceResult = g_VulkanAvailable = true;
        }
        else
            bVulkanInstanceResult = g_VulkanAvailable = false;
    }

    return bVulkanInstanceResult;
}


void CleanupVulkanFramework (void)
{
    if (VK_NULL_HANDLE != gImageLabVkInstance)
    {
        vkDestroyInstance (gImageLabVkInstance, nullptr);
        gImageLabVkInstance = nullptr;
    }

    g_VulkanAvailable = false;

    return;
}