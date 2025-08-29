#ifndef __IMAGE_LAB_RETRO_VISION_PALETTE_EGA_VALUES__
#define __IMAGE_LAB_RETRO_VISION_PALETTE_EGA_VALUES__

#include "Common.hpp"
#include "PaletteEntry.hpp"
#include <array>

constexpr int32_t EGA_width  = 640;
constexpr int32_t EGA_height = 350;
constexpr int32_t EGA_PaletteSize = 16;

using EGA_Palette    = std::array<PEntry<uint8_t>, EGA_PaletteSize>;
using EGA_PaletteF32 = std::array<PEntry<float>, EGA_PaletteSize>;

// EGA standard palette
CACHE_ALIGN constexpr EGA_Palette EGA_Standard_u8 =
{{
    {   0,   0,   0 },
    {   0,   0, 170 },
    {   0, 170,   0 },
    {   0, 170, 170 },
    { 170,   0,   0 },
    { 170,   0, 170 },
    { 170,  85,   0 },
    { 170, 170, 170 },
    {  85,  85,  85 },
    {  85,  85, 255 },
    {  85, 255,  85 },
    {  85, 255, 255 },
    { 255,  85,  85 },
    { 255,  85, 255 },
    { 255, 255,  85 },
    { 255, 255, 255 }
}};

// EGA King's Quest III Approximation
CACHE_ALIGN constexpr EGA_Palette EGA_KQ3_u8 =
{{
    {   0,   0,   0 },
    {   0,   0, 128 },
    {   0,  80,   0 },
    {   0,  80,  80 },
    { 128,   0,   0 },
    { 128,   0, 128 },
    { 128,  64,   0 },
    { 170, 170, 170 },
    { 85,   85,  85 },
    { 85,   85, 255 },
    { 85,  200,  85 },
    { 85,  200, 200 },
    { 200,  85,  85 },
    { 200,  85, 200 },
    { 255, 255,  85 },
    { 255, 255, 255 }
}};

// EGA Kyrandia-Inspired
CACHE_ALIGN constexpr EGA_Palette EGA_Kyrandia_u8 =
{{
    {   0,   0,   0 },
    {   0,   0, 140 },
    {   0, 100,   0 },
    {   0, 100, 100 },
    { 140,   0,   0 },
    { 140,   0, 140 },
    { 140,  70,   0 },
    { 160, 160, 160 },
    {  88,  80,  80 },
    {  80,  80, 150 },
    {  80, 200,  80 },
    {  80, 200, 200 },
    { 255,  50,  50 },
    { 255,  50, 255 },
    { 255, 255,  80 },
    { 255, 255, 255 }
}};

// EGA Thexder 
CACHE_ALIGN constexpr EGA_Palette EGA_Thexder_u8 =
{{
    {   0,   0,   0 },
    {  85,   0,  85 },
    {  50,  85,  50 },
    {   0, 170, 170 },
    { 170,   0,   0 },
    { 170,   0, 170 },
    { 170,  85,   0 },
    { 170, 170, 170 },
    {  85,  85,  85 },
    {  85,  85, 255 },
    {  85, 255,  85 },
    {  85, 255, 255 },
    { 255,  85,  85 },
    { 255,  85, 255 },
    { 255, 255,  80 },
    { 255, 255, 255 }
}};

// EGA Dune
CACHE_ALIGN constexpr EGA_Palette EGA_Dune_u8 =
{{
    {   0,   0,   0 },
    {   0,   0,  80 },
    {  40,  80,  80 },
    {   0,  80,  80 },
    { 120,  40,  40 },
    { 120,  40, 120 },
    { 140, 100,  60 },
    { 100, 100, 100 },
    {  80,  80,  80 },
    {  80,  80, 160 },
    {  80, 120,  80 },
    {  80, 120, 120 },
    { 200,  80,  60 },
    { 160,  80, 160 },
    { 200, 160,  80 },
    { 220, 200, 160 }
}};

// EGA Doom
CACHE_ALIGN constexpr EGA_Palette EGA_Doom_u8 =
{{
    {   0,   0,   0 },
    {  40,  40,  40 },
    {  80,  70,  60 },
    {  40,  80,  40 },
    { 120,   0,   0 },
    {  80,   0,  80 },
    { 100,  60,  40 },
    { 100, 100, 100 },
    {  60,  80,  80 },
    {  40,  40,  60 },
    {   0, 200,   0 },
    {   0, 120, 120 },
    { 255,  40,  40 },
    { 160,   0, 160 },
    { 255, 255,   0 },
    { 255, 255, 255 }
}};

// EGA Metal Mutant
CACHE_ALIGN constexpr EGA_Palette EGA_MetalMutant_u8 =
{{
    {   0,   0,   0 },
    {   0,  80,   0 },
    {   0, 120,   0 },
    {   0, 100,  80 },
    {  80,  80,  80 },
    { 120, 120, 120 },
    { 160, 160, 160 },
    {   0,   0, 100 },
    {  50,  50, 120 },
    {  80, 200,  80 },
    {  80, 200, 200 },
    { 160,  80,  40 },
    { 200,   0,   0 },
    { 255,   0,   0 },
    { 200, 160,   0 },
    { 255, 255, 255 }
}};

// EGA Wolfenstein 3D
CACHE_ALIGN constexpr EGA_Palette EGA_Wolfenstein_u8 =
{{
    {   0,   0,   0 },
    {  50,  50,  50 },
    {  80,  70,  60 },
    { 100,  70,  40 },
    { 120,   0,   0 },
    { 100,  50,  40 },
    { 150,  60,  40 },
    { 180, 120,  80 },
    { 100, 100, 100 },
    { 160, 160, 160 },
    {  80, 100,  80 },
    {  70,  80, 100 },
    { 255,   0,   0 },
    { 200, 160,  80 },
    { 255, 255,  80 },
    { 255, 255, 255 }
}};

#endif // __IMAGE_LAB_RETRO_VISION_PALETTE_EGA_VALUES__