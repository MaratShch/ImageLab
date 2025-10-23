#include <atomic>
#include "VulkanBinding.hpp"

std::atomic<VkInstance> gInstance;

VkInstance CreateVulkanInstance  (void)
{
    uint32_t version = 0;
    VkResult res = vkEnumerateInstanceVersion(&version);
    if (res != VK_SUCCESS)
        return false;

    uint32_t gpuCount = 0;
    VkInstanceCreateInfo ci{ VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO };
    VkInstance instance = VK_NULL_HANDLE;
    if (vkCreateInstance(&ci, nullptr, &instance) != VK_SUCCESS)
        return false;

    vkEnumeratePhysicalDevices(instance, &gpuCount, nullptr);
    vkDestroyInstance(instance, nullptr);

    return instance;
}