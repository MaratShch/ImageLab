#ifndef __IMAGE_LAB2_VULKAN_INTERNAL_INTERFACE__
#define __IMAGE_LAB2_VULKAN_INTERNAL_INTERFACE__

// THIS FILE SHOULD BE INCLUDED ONLY INTO *.CPP FILES FROM THIS DLL
#include <vector>
#include <vulkan/vulkan.h>

bool InitVulkanFramework (void);
void CleanupVulkanFramework (void);
VkInstance getVulkanInstance (void);
bool IsVulkanAvailable(void);

void fillDeviceVector (const std::vector<VkPhysicalDevice>&);

struct ILVulkanHandler
{
    size_t   strSizeof;
    uint32_t hndlVersion;
    uint32_t deviceNumbers;
    void*    vkDeviceArray;
    VkInstance vkInstance;
    VkPhysicalDevice vkPhysicalDevice;
    VkDevice vkDevice;
    VkQueue  vkQueue;
};

constexpr size_t ILVulkanHandlerSize = sizeof(ILVulkanHandler);
constexpr uint32_t ILVulkanHandlerVersion = 1u;

#endif // __IMAGE_LAB2_VULKAN_INTERNAL_INTERFACE__