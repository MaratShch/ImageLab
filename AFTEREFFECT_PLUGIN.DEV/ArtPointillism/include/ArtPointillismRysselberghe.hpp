#ifndef __IMAGE_LAB_ART_POINTILISM_PAINTER_RYSSELBERGHE_TECHNICS__
#define __IMAGE_LAB_ART_POINTILISM_PAINTER_RYSSELBERGHE_TECHNICS__

#include <array>
#include "ArtPointillismPalette.hpp"
#include "Common.hpp"

using Rysselberghe_Palette_u8 = std::array<PEntry<uint8_t>, 16>;
using Rysselberghe_Palette_f32 = std::array<PEntry<float>, 16>;

CACHE_ALIGN constexpr Rysselberghe_Palette_u8 Rysselberghe_u8 =
{{
    { 255, 255, 255 },  // white highlight
    { 248, 236, 120 },  // bright yellow
    { 252, 202,  85 },  // warm yellow - ochre
    { 240, 155,  72 },  // orange
    { 225,  94,  66 },  // orange
    { 186,  52,  72 },  // carmine red
    { 164,  80, 140 },  // violet - magenta
    { 124,  58, 110 },  // deep purple
    { 108, 120, 200 },  // lavender blue
    {  78,  98, 185 },  // cobalt blue
    {  48,  72, 155 },  // ultramarine blue
    {  32,  56, 118 },  // deep marine blue
    { 108, 168, 145 },  // turquoise - green
    {  72, 128, 110 },  // sea green
    {  52, 102,  78 },  // dark green
    {  28,  42,  52 }   // black cool shadow
}};

CACHE_ALIGN constexpr Rysselberghe_Palette_f32 Rysselberghe_f32 =
{{
    { F32(Rysselberghe_u8[ 0].r), F32(Rysselberghe_u8[ 0].g), F32(Rysselberghe_u8[ 0].b) },
    { F32(Rysselberghe_u8[ 1].r), F32(Rysselberghe_u8[ 1].g), F32(Rysselberghe_u8[ 1].b) },
    { F32(Rysselberghe_u8[ 2].r), F32(Rysselberghe_u8[ 2].g), F32(Rysselberghe_u8[ 2].b) },
    { F32(Rysselberghe_u8[ 3].r), F32(Rysselberghe_u8[ 3].g), F32(Rysselberghe_u8[ 3].b) },
    { F32(Rysselberghe_u8[ 4].r), F32(Rysselberghe_u8[ 4].g), F32(Rysselberghe_u8[ 4].b) },
    { F32(Rysselberghe_u8[ 5].r), F32(Rysselberghe_u8[ 5].g), F32(Rysselberghe_u8[ 5].b) },
    { F32(Rysselberghe_u8[ 6].r), F32(Rysselberghe_u8[ 6].g), F32(Rysselberghe_u8[ 6].b) },
    { F32(Rysselberghe_u8[ 7].r), F32(Rysselberghe_u8[ 7].g), F32(Rysselberghe_u8[ 7].b) },
    { F32(Rysselberghe_u8[ 8].r), F32(Rysselberghe_u8[ 8].g), F32(Rysselberghe_u8[ 8].b) },
    { F32(Rysselberghe_u8[ 9].r), F32(Rysselberghe_u8[ 9].g), F32(Rysselberghe_u8[ 9].b) },
    { F32(Rysselberghe_u8[10].r), F32(Rysselberghe_u8[10].g), F32(Rysselberghe_u8[10].b) },
    { F32(Rysselberghe_u8[11].r), F32(Rysselberghe_u8[11].g), F32(Rysselberghe_u8[11].b) },
    { F32(Rysselberghe_u8[12].r), F32(Rysselberghe_u8[12].g), F32(Rysselberghe_u8[12].b) },
    { F32(Rysselberghe_u8[13].r), F32(Rysselberghe_u8[13].g), F32(Rysselberghe_u8[13].b) },
    { F32(Rysselberghe_u8[14].r), F32(Rysselberghe_u8[14].g), F32(Rysselberghe_u8[14].b) },
    { F32(Rysselberghe_u8[15].r), F32(Rysselberghe_u8[15].g), F32(Rysselberghe_u8[15].b) }
}};


#endif // __IMAGE_LAB_ART_POINTILISM_PAINTER_RYSSELBERGHE_TECHNICS__