#ifndef __IMAGE_LAB2_VULKAN_LIBRARY_ALGORITHMS_INTERFACE__
#define __IMAGE_LAB2_VULKAN_LIBRARY_ALGORITHMS_INTERFACE__

using ILVulkanHndl = void*;

#include "VulkanDevice.hpp"

DLL_API_EXPORT ILVulkanHndl CreateVulkanContext (uint32_t core, uint32_t memory, uint32_t reserved);
DLL_API_EXPORT void FreeVulkanContext (ILVulkanHndl);

#endif // __IMAGE_LAB2_VULKAN_LIBRARY_ALGORITHMS_INTERFACE__