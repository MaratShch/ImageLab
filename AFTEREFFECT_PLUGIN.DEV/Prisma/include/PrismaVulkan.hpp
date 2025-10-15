#ifndef __IMAGE_LAB2_VULKAN_ALGORITHM_BINDING_LIBRARY__
#define __IMAGE_LAB2_VULKAN_ALGORITHM_BINDING_LIBRARY__

#include <cstdint>

void* VulkanAllocNode (uint32_t proc, uint32_t mem, uint32_t reserved);
void VulkanFreeNode (void* pNodeHndl);

#endif // __IMAGE_LAB2_VULKAN_ALGORITHM_BINDING_LIBRARY__