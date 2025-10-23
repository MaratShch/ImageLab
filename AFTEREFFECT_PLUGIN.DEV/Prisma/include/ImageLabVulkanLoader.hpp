#ifndef __IMAGE_LAB_PRISMA_FILTER_VULKAN_ALGORITHN_LOADER__
#define __IMAGE_LAB_PRISMA_FILTER_VULKAN_ALGORITHN_LOADER__

#include <windows.h>
#include "AE_Effect.h"

bool LoadVulkanAlgoDll (PF_InData* in_data);
void UnloadVulkanAlgoDll (void);
uint32_t getVulkanVersionNumber (void);

typedef void*    (WINAPI *CreateVulkanContext1)  (uint32_t, uint32_t, uint32_t);
typedef void     (WINAPI *FreeVulkanContext1)    (void*);
typedef uint32_t (WINAPI *GetVulkanVersion1)     (void);

struct PrismaAlgoVulkanHandler
{
    GetVulkanVersion1       getVulkanVersion;
    CreateVulkanContext1    createVulkanContext;
    FreeVulkanContext1      freeVulkanContext;
};


void* createVulkanContext (uint32_t proc, uint32_t mem, uint32_t reserved);
void freeVulkanContext (void* pHndl);

#endif // __IMAGE_LAB_PRISMA_FILTER_VULKAN_ALGORITHN_LOADER__