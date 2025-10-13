#include <atomic>
#include <vector>
#include "ImageLabVulkanHandler.hpp"
#include "VulkanDevice.hpp"

std::atomic<uint32_t> vulkanRefCount{};
std::vector<VkPhysicalDevice> vulkanDevices;

static VkPhysicalDevice g_PhysicalDevice { VK_NULL_HANDLE };
static VkDevice g_Device                 { VK_NULL_HANDLE };


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

uint32_t GetVulkanVersion(void)
{
    uint32_t version = 0u;
    vkEnumerateInstanceVersion(&version);
    return version;
}

void fillDeviceVector(const std::vector<VkPhysicalDevice>& in)
{
    vulkanDevices.resize(in.size());
    std::copy(in.begin(), in.end(), vulkanDevices.begin());
    return;
}
