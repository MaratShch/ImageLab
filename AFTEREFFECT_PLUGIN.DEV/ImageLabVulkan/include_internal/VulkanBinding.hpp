#ifndef __IMAGE_LAB2_VULKAN_FRAMEWORK_INTERFACE_BINDING_HEADER__
#define __IMAGE_LAB2_VULKAN_FRAMEWORK_INTERFACE_BINDING_HEADER__

#include <array>
#include <atomic>
#include <mutex>
#include <vector>
#include <vulkan/vulkan.h>
#include "LibExport.hpp"

namespace VulkanGPU
{
    class CVulkanGpuContext
    {
        public:
            CVulkanGpuContext(uint32_t idx, VkPhysicalDevice physDevice)
            { 
                gpuId = idx; 
                isInitialized = false;
                m_PhysDevice = physDevice;
            }

            ~CVulkanGpuContext(void) { ; }

        private:
            uint32_t gpuId;
            std::atomic<bool> isInitialized;
            std::mutex gpuQueue;
            VkPhysicalDevice m_PhysDevice;
            VkDevice         m_Device;
            VkQueue          m_Queue;

    };
}

void CleanupOnDllLoad (void);

DLL_API_EXPORT bool CreateVulkanInstance(void);
DLL_API_EXPORT void DeleteVulkanInstance(void);


#endif // __IMAGE_LAB2_VULKAN_FRAMEWORK_INTERFACE_BINDING_HEADER__