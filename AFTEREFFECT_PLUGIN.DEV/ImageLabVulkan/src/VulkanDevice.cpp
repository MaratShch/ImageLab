#include <vulkan/vulkan.h>
#include "VulkanDevice.hpp"


uint32_t GetVulkanVersion(void)
{
    uint32_t version = 0u;
    vkEnumerateInstanceVersion(&version);
    return version;
}