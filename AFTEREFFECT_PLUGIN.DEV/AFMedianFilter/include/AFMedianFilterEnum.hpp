#ifndef __IMAGE_LAB_IMAGE_GEOMETRY_ENUMERATORS_FILTER__
#define __IMAGE_LAB_IMAGE_GEOMETRY_ENUMERATORS_FILTER__

#include <cstdint>
#include "AE_Effect.h"

enum class AFMF : int32_t
{
    eIMAGE_AFMEDIAN_INPUT,
    eIMAGE_AFMEDIAN_TOTAL_CONTROLS
};

constexpr int32_t kerenlSizeMin = 3;
constexpr int32_t kernelSizeMax = 31;
constexpr int32_t video_vga_kernel_max = 7;             // 480p
constexpr int32_t video_1080p_kernel_max = 11;          // 1080p
constexpr int32_t video_4K_kernel_max = 21;             // 4K
constexpr int32_t video_8k_kernel_max = kernelSizeMax;  // 8K


#endif // __IMAGE_LAB_IMAGE_GEOMETRY_ENUMERATORS_FILTER__