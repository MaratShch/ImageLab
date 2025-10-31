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
                exts.clear();
            }

            ~CVulkanGpuContext(void) { exts.clear(); }

            const std::vector<VkExtensionProperties> get_extension_properties(void) { return exts; }
            
            void fill_extension_properties (const std::vector<VkExtensionProperties>& ext)
            {
                exts.resize(ext.size());
                std::copy(ext.begin(), ext.end(), exts.begin());
                return;
            }

        private:
            uint32_t gpuId;
            std::atomic<bool> isInitialized;
            std::mutex gpuQueue;
            VkPhysicalDevice m_PhysDevice;
            VkDevice         m_Device;
            VkQueue          m_Queue;
            std::vector<VkExtensionProperties> exts;

    };

}

void CleanupOnDllLoad (void);

DLL_API_EXPORT bool CreateVulkanInstance(void);
DLL_API_EXPORT void DeleteVulkanInstance(void);


#endif // __IMAGE_LAB2_VULKAN_FRAMEWORK_INTERFACE_BINDING_HEADER__