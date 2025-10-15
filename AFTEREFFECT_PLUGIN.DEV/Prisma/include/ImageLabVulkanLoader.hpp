#ifndef __IMAGE_LAB_PRISMA_FILTER_VULKAN_ALGORITHN_LOADER__
#define __IMAGE_LAB_PRISMA_FILTER_VULKAN_ALGORITHN_LOADER__

#include <windows.h>
#include "AE_Effect.h"

bool LoadVulkanAlgoDll (PF_InData* in_data);
void UnloadVulkanAlgoDll (void);

typedef void* (WINAPI *VulkanAllocNode1) (uint32_t, uint32_t, uint32_t);
typedef void  (WINAPI *VulkanFreeNode1) (void*);

struct PrismaAlgoVulkanHandler
{
    VulkanAllocNode1 allocNode;
    VulkanFreeNode1  freeNode;
};

#endif // __IMAGE_LAB_PRISMA_FILTER_VULKAN_ALGORITHN_LOADER__