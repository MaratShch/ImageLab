#ifndef __IMAGE_LAB_RETRO_VISION_PALETTE_CGA_VALUES__
#define __IMAGE_LAB_RETRO_VISION_PALETTE_CGA_VALUES__

#include "Common.hpp"
#include "PaletteEntry.hpp"
#include <array>

constexpr int32_t CGA_width = 320;
constexpr int32_t CGA_height = 240;

constexpr int32_t CGA_PaletteSize = 4;

using CGA_Palette    = std::array<PEntry<uint8_t>, CGA_PaletteSize>;
using CGA_PaletteF32 = std::array<PEntry<float>, CGA_PaletteSize>;

// CGA-0 palette with intencity bit disabled
CACHE_ALIGN constexpr CGA_Palette CGA0_u8 =
{{
    {   0,   0,   0 },
    {   0, 170, 170 },
    { 170,   0, 170 },
    { 255, 255, 255 }
}};

// CGA-0 palette with intencity bit enabled
CACHE_ALIGN constexpr CGA_Palette CGA0i_u8 =
{{
    {   0,   0,   0 },
    {  85, 255, 255 },
    { 255,  85, 255 },
    { 255, 255, 255 }
}};

// CGA-1 palette with intencity bit disabled
CACHE_ALIGN constexpr CGA_Palette CGA1_u8 =
{{
    {   0,   0,   0 },
    {   0, 170,   0 },
    { 170,   0,   0 },
    { 170,  85,   0 }
}};

// CGA-1 palette with intencity bit enabled
CACHE_ALIGN constexpr CGA_Palette CGA1i_u8 =
{{
    {   0,   0,   0 },
    {  85, 255,  85 },
    { 255,  85,  85 },
    { 255, 255,  85 }
}};

#endif // __IMAGE_LAB_RETRO_VISION_PALETTE_CGA_VALUES__