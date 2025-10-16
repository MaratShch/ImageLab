#ifndef __IMAGE_LAB2_VULKAN_DEVICE_HANDLER_PRIVAYE_INLUDE_FILE__
#define __IMAGE_LAB2_VULKAN_DEVICE_HANDLER_PRIVAYE_INLUDE_FILE__

#include "VulkanDevice.hpp"

uint32_t setDeviceNodeIdx (uint32_t policyCore, uint32_t policyMem, uint32_t reserved);
void resetDeviceNodeIdx (uint32_t devIdx);

const std::vector<VkPhysicalDevice>& getDeviceArray(void);

#endif // __IMAGE_LAB2_VULKAN_DEVICE_HANDLER_PRIVAYE_INLUDE_FILE__