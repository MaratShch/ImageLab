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

#endif // __IMAGE_LAB2_VULKAN_INTERNAL_INTERFACE__