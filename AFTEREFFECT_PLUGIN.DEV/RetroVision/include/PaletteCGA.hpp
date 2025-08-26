#ifndef __IMAGE_LAB_RETRO_VISION_PALETTE_CGA_VALUES__
#define __IMAGE_LAB_RETRO_VISION_PALETTE_CGA_VALUES__

#include "PaletteEntry.hpp"
#include <array>

constexpr int32_t CGA_width = 320;
constexpr int32_t CGA_height = 240;

// CGA-0 palette with intencity bit disabled
constexpr std::array<PEntry<uint8_t>, 4> CGA0_u8 =
{{
    {   0,   0,   0 },
    {   0, 170, 170 },
    { 170,   0, 170 },
    { 255, 255, 255 }
}};

// CGA-0 palette with intencity bit enabled
constexpr std::array<PEntry<uint8_t>, 4> CGA0i_u8 =
{{
    {   0,   0,   0 },
    {  85, 255, 255 },
    { 255,  85, 255 },
    { 255, 255, 255 }
}};

// CGA-1 palette with intencity bit disabled
constexpr std::array<PEntry<uint8_t>, 4> CGA1_u8 =
{{
    {   0,   0,   0 },
    {   0, 170,   0 },
    { 170,   0,   0 },
    { 170,  85,   0 }
}};

// CGA-1 palette with intencity bit enabled
constexpr std::array<PEntry<uint8_t>, 4> CGA1i_u8 =
{{
    {   0,   0,   0 },
    {  85, 255,  85 },
    { 255,  85,  85 },
    { 255, 255,  85 }
}};

#endif // __IMAGE_LAB_RETRO_VISION_PALETTE_CGA_VALUES__