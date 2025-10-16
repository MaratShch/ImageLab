#include <array>
#include <atomic>
#include <mutex>
#include <vector>
#include "ImageLabVulkanHandler.hpp"
#include "VulkanDevice.hpp"

constexpr size_t maxGpuSupported = 16u;
std::vector<VkPhysicalDevice> vulkanDevices;

std::array<std::atomic<int32_t>, maxGpuSupported> g_DeviceRefCount{};
std::array<std::vector<VkExtensionProperties>, maxGpuSupported> devExts{};
std::array<VkPhysicalDeviceProperties, maxGpuSupported> devProps{};


constexpr int32_t VulkanDevicesEnumSize = 7;
constexpr VulkanDeviceEnumeration dev[VulkanDevicesEnumSize]
{
    {0x12D2, "NVIDIA HPC"}, // Sometimes seen in HPC or simulation hardware (alias to 0x10DE).
    {0x10DE, "NVIDIA"},     // GeForce, Quadro, RTX, Tesla — all supported on Windows.
    {0x106B, "APPLE"},      // Apple Silicon GPUs (Vulkan via MoltenVK on macOS/iOS).
    {0x1002, "AMD"},        // Radeon, FirePro, RDNA-based GPUs.
    {0x8086, "Intel"},      // Integrated GPUs (UHD, Iris Xe, Arc series).
    {0x1414, "Microsoft"},  // Software fallback adapter (used when no GPU present).
    {0x15AD, "VMWare"}      // Virtual GPU support, often with Vulkan 1.1-1.3 passthrough.
};

void IncrementDevice (size_t idx)
{
    g_DeviceRefCount[idx].fetch_add(1, std::memory_order_relaxed);
}

void DecrementDevice (size_t idx)
{
    g_DeviceRefCount[idx].fetch_sub(1, std::memory_order_relaxed);
}

int32_t GetRefCount (size_t idx)
{
    return g_DeviceRefCount[idx].load(std::memory_order_relaxed);
}


uint32_t GetVulkanVersion(void)
{
    uint32_t version = 0u;
    vkEnumerateInstanceVersion(&version);
    return version;
}

void fillDeviceVector(const std::vector<VkPhysicalDevice>& inDev)
{
    uint32_t idx = 0u;

    for (auto& dev : inDev)
    {
        vulkanDevices.push_back(dev);
        vkGetPhysicalDeviceProperties (dev, &devProps[idx]);

        uint32_t propertiesCount = 0u;
        vkEnumerateDeviceExtensionProperties (dev, nullptr, &propertiesCount, nullptr);
        
        devExts[idx].resize (propertiesCount);
        vkEnumerateDeviceExtensionProperties (dev, nullptr, &propertiesCount, devExts[idx].data());

        idx++;
    }

    return;
}


uint32_t setDeviceNodeIdx (uint32_t policyCore, uint32_t policyMem, uint32_t reserved)
{
    uint32_t devIdx;

    if (1u == vulkanDevices.size()) // if only one GPU found - return this GPU index
        devIdx = 0u;
    else // search best GPU by policy (will be implemented later)
        devIdx = 0u;

    IncrementDevice (devIdx);
    return devIdx;
}

void resetDeviceNodeIdx (uint32_t devIdx)
{
    DecrementDevice (devIdx);
    return;
}

const std::vector<VkPhysicalDevice>& getDeviceArray(void)
{
    return vulkanDevices;
}

