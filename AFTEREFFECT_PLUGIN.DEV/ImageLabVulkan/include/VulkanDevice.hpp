#ifndef __IMAGE_LAB2_VULKAN_FRAMEWORK_DEVICE_APIs__
#define __IMAGE_LAB2_VULKAN_FRAMEWORK_DEVICE_APIs__

#include <cstdint>
#include <utility>
#include "LibExport.hpp"

using VulkanDeviceEnumeration = std::pair<uint32_t, char*>;

DLL_API_EXPORT uint32_t GetVulkanVersion(void);


#endif // __IMAGE_LAB2_VULKAN_FRAMEWORK_DEVICE_APIs__