#ifndef __IMAGE_LAB_ART_POINTILISM_PAINTER_MATISSE_TECHNICS__
#define __IMAGE_LAB_ART_POINTILISM_PAINTER_MATISSE_TECHNICS__

#include <array>
#include "ArtPointilismPalette.hpp"
#include "Common.hpp"
    
using Matisse_Palette_u8  = std::array<PEntry<uint8_t>, 10>;
using Matisse_Palette_f32 = std::array<PEntry<float>, 10>;

CACHE_ALIGN constexpr Matisse_Palette_u8 Matisse_u8 =
{{
    { 255,   0,   0 },
    { 255, 102,   0 },
    { 255, 255,   0 },
    { 102, 255,   0 },
    {   0, 255, 255 },
    {   0, 102, 255 },
    { 153,   0, 255 },
    { 255, 255, 255 },
    {   0,   0,   0 },
    { 128, 128, 128 }
}};

CACHE_ALIGN constexpr Matisse_Palette_f32 Matisse_f32 =
{{
    { F32(Matisse_u8[0].r), F32(Matisse_u8[0].g), F32(Matisse_u8[0].b) },
    { F32(Matisse_u8[1].r), F32(Matisse_u8[1].g), F32(Matisse_u8[1].b) },
    { F32(Matisse_u8[2].r), F32(Matisse_u8[2].g), F32(Matisse_u8[2].b) },
    { F32(Matisse_u8[3].r), F32(Matisse_u8[3].g), F32(Matisse_u8[3].b) },
    { F32(Matisse_u8[4].r), F32(Matisse_u8[4].g), F32(Matisse_u8[4].b) },
    { F32(Matisse_u8[5].r), F32(Matisse_u8[5].g), F32(Matisse_u8[5].b) },
    { F32(Matisse_u8[6].r), F32(Matisse_u8[6].g), F32(Matisse_u8[6].b) },
    { F32(Matisse_u8[7].r), F32(Matisse_u8[7].g), F32(Matisse_u8[7].b) },
    { F32(Matisse_u8[8].r), F32(Matisse_u8[8].g), F32(Matisse_u8[8].b) },
    { F32(Matisse_u8[9].r), F32(Matisse_u8[9].g), F32(Matisse_u8[9].b) }
}};


#endif // __IMAGE_LAB_ART_POINTILISM_PAINTER_MATISSE_TECHNICS__