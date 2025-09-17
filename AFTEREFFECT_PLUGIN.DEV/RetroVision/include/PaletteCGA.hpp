#ifndef __IMAGE_LAB_RETRO_VISION_PALETTE_CGA_VALUES__
#define __IMAGE_LAB_RETRO_VISION_PALETTE_CGA_VALUES__

#include "Common.hpp"
#include "PaletteEntry.hpp"
#include <cstdint>
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

CACHE_ALIGN constexpr CGA_PaletteF32 CGA0_f32 =
{{
    { static_cast<float>(CGA0_u8[0].r) / 255.f, static_cast<float>(CGA0_u8[0].g) / 255.f, static_cast<float>(CGA0_u8[0].b) / 255.f },
    { static_cast<float>(CGA0_u8[1].r) / 255.f, static_cast<float>(CGA0_u8[1].g) / 255.f, static_cast<float>(CGA0_u8[1].b) / 255.f },
    { static_cast<float>(CGA0_u8[2].r) / 255.f, static_cast<float>(CGA0_u8[2].g) / 255.f, static_cast<float>(CGA0_u8[2].b) / 255.f },
    { static_cast<float>(CGA0_u8[3].r) / 255.f, static_cast<float>(CGA0_u8[3].g) / 255.f, static_cast<float>(CGA0_u8[3].b) / 255.f }
}};


// CGA-0 palette with intencity bit enabled
CACHE_ALIGN constexpr CGA_Palette CGA0i_u8 =
{{
    {   0,   0,   0 },
    {  85, 255, 255 },
    { 255,  85, 255 },
    { 255, 255, 255 }
}};

CACHE_ALIGN constexpr CGA_PaletteF32 CGA0i_f32 =
{{
    { static_cast<float>(CGA0i_u8[0].r) / 255.f, static_cast<float>(CGA0i_u8[0].g) / 255.f, static_cast<float>(CGA0i_u8[0].b) / 255.f },
    { static_cast<float>(CGA0i_u8[1].r) / 255.f, static_cast<float>(CGA0i_u8[1].g) / 255.f, static_cast<float>(CGA0i_u8[1].b) / 255.f },
    { static_cast<float>(CGA0i_u8[2].r) / 255.f, static_cast<float>(CGA0i_u8[2].g) / 255.f, static_cast<float>(CGA0i_u8[2].b) / 255.f },
    { static_cast<float>(CGA0i_u8[3].r) / 255.f, static_cast<float>(CGA0i_u8[3].g) / 255.f, static_cast<float>(CGA0i_u8[3].b) / 255.f }
}};


// CGA-1 palette with intencity bit disabled
CACHE_ALIGN constexpr CGA_Palette CGA1_u8 =
{{
    {   0,   0,   0 },
    {   0, 170,   0 },
    { 170,   0,   0 },
    { 170,  85,   0 }
}};

CACHE_ALIGN constexpr CGA_PaletteF32 CGA1_f32 =
{{
    { static_cast<float>(CGA1_u8[0].r) / 255.f, static_cast<float>(CGA1_u8[0].g) / 255.f, static_cast<float>(CGA1_u8[0].b) / 255.f },
    { static_cast<float>(CGA1_u8[1].r) / 255.f, static_cast<float>(CGA1_u8[1].g) / 255.f, static_cast<float>(CGA1_u8[1].b) / 255.f },
    { static_cast<float>(CGA1_u8[2].r) / 255.f, static_cast<float>(CGA1_u8[2].g) / 255.f, static_cast<float>(CGA1_u8[2].b) / 255.f },
    { static_cast<float>(CGA1_u8[3].r) / 255.f, static_cast<float>(CGA1_u8[3].g) / 255.f, static_cast<float>(CGA1_u8[3].b) / 255.f }
}};


// CGA-1 palette with intencity bit enabled
CACHE_ALIGN constexpr CGA_Palette CGA1i_u8 =
{{
    {   0,   0,   0 },
    {  85, 255,  85 },
    { 255,  85,  85 },
    { 255, 255,  85 }
}};

CACHE_ALIGN constexpr CGA_PaletteF32 CGA1i_f32 =
{{
    { static_cast<float>(CGA1i_u8[0].r) / 255.f, static_cast<float>(CGA1i_u8[0].g) / 255.f, static_cast<float>(CGA1i_u8[0].b) / 255.f },
    { static_cast<float>(CGA1i_u8[1].r) / 255.f, static_cast<float>(CGA1i_u8[1].g) / 255.f, static_cast<float>(CGA1i_u8[1].b) / 255.f },
    { static_cast<float>(CGA1i_u8[2].r) / 255.f, static_cast<float>(CGA1i_u8[2].g) / 255.f, static_cast<float>(CGA1i_u8[2].b) / 255.f },
    { static_cast<float>(CGA1i_u8[3].r) / 255.f, static_cast<float>(CGA1i_u8[3].g) / 255.f, static_cast<float>(CGA1i_u8[3].b) / 255.f }
}};


#endif // __IMAGE_LAB_RETRO_VISION_PALETTE_CGA_VALUES__