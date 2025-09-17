#ifndef __IMAGE_LAB_RETRO_VISION_PALETTE_HERCULES_VALUES__
#define __IMAGE_LAB_RETRO_VISION_PALETTE_HERCULES_VALUES__

#include "PaletteEntry.hpp"
#include <cstdint>

constexpr int32_t Hercules_width   = 720;
constexpr int32_t Hercules_heightd = 348;

constexpr int32_t HERCULES_White_levels = 3;
using HERCULES_White    = std::array<PEntry<uint8_t>, HERCULES_White_levels>;
using HERCULES_WhiteF32 = std::array<PEntry<float>, HERCULES_White_levels>;

CACHE_ALIGN constexpr HERCULES_White HERCULES_White_Color =
{{
    { 255, 255, 255 }, // pure white
    { 250, 250, 255 }, // slightly bluish
    { 245, 245, 255 }  // bluish
}};

CACHE_ALIGN constexpr HERCULES_WhiteF32 HERCULES_White_ColorF32 =
{{
    { static_cast<float>(HERCULES_White_Color[0].r) / 255.f, static_cast<float>(HERCULES_White_Color[0].g) / 255.f, static_cast<float>(HERCULES_White_Color[0].b) / 255.f },
    { static_cast<float>(HERCULES_White_Color[1].r) / 255.f, static_cast<float>(HERCULES_White_Color[1].g) / 255.f, static_cast<float>(HERCULES_White_Color[1].b) / 255.f },
    { static_cast<float>(HERCULES_White_Color[2].r) / 255.f, static_cast<float>(HERCULES_White_Color[2].g) / 255.f, static_cast<float>(HERCULES_White_Color[2].b) / 255.f },
}};

#endif // __IMAGE_LAB_RETRO_VISION_PALETTE_HERCULES_VALUES__