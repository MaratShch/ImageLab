#ifndef __IMAGE_LAB_ART_POINTILISM_PAINTER_LUCE_TECHNICS__
#define __IMAGE_LAB_ART_POINTILISM_PAINTER_LUCE_TECHNICS__

#include <array>
#include "ArtPointillismPalette.hpp"
#include "Common.hpp"

using Luce_Palette_u8 = std::array<PEntry<uint8_t>, 16>;
using Luce_Palette_f32 = std::array<PEntry<float>, 16>;

CACHE_ALIGN constexpr Luce_Palette_u8 Luce_u8 =
{{
    { 250, 250, 250 },  // white / lamp glare
    { 240, 224, 150 },  // warm streetlight yellow
    { 225, 176,  92 },  // golden - ochre light
    { 208, 132,  64 },  // warm amber
    { 182,  90,  60 },  // reddish light source
    { 155,  42,  60 },  // crimson accent
    { 128,  72, 140 },  // violet haze
    {  96,  52, 112 },  // smoky indigo
    {  80, 110, 185 },  // bright blue accent
    {  58,  88, 165 },  // cobalt steel - blue
    {  34,  62, 132 },  // ultramarine night blue
    {  22,  48, 102 },  // deep navy
    {  68, 120, 108 },  // industrial green - cyan
    {  48,  88,  78 },  // muted teal
    {  32,  60,  52 },  // dirty green shadow
    {  15,  22,  28 }   // black - blue deep shadow
}};


CACHE_ALIGN constexpr Luce_Palette_f32 Luce_f32 =
{{
    { F32(Luce_u8[ 0].r), F32(Luce_u8[ 0].g), F32(Luce_u8[ 0].b) },
    { F32(Luce_u8[ 1].r), F32(Luce_u8[ 1].g), F32(Luce_u8[ 1].b) },
    { F32(Luce_u8[ 2].r), F32(Luce_u8[ 2].g), F32(Luce_u8[ 2].b) },
    { F32(Luce_u8[ 3].r), F32(Luce_u8[ 3].g), F32(Luce_u8[ 3].b) },
    { F32(Luce_u8[ 4].r), F32(Luce_u8[ 4].g), F32(Luce_u8[ 4].b) },
    { F32(Luce_u8[ 5].r), F32(Luce_u8[ 5].g), F32(Luce_u8[ 5].b) },
    { F32(Luce_u8[ 6].r), F32(Luce_u8[ 6].g), F32(Luce_u8[ 6].b) },
    { F32(Luce_u8[ 7].r), F32(Luce_u8[ 7].g), F32(Luce_u8[ 7].b) },
    { F32(Luce_u8[ 8].r), F32(Luce_u8[ 8].g), F32(Luce_u8[ 8].b) },
    { F32(Luce_u8[ 9].r), F32(Luce_u8[ 9].g), F32(Luce_u8[ 9].b) },
    { F32(Luce_u8[10].r), F32(Luce_u8[10].g), F32(Luce_u8[10].b) },
    { F32(Luce_u8[11].r), F32(Luce_u8[11].g), F32(Luce_u8[11].b) },
    { F32(Luce_u8[12].r), F32(Luce_u8[12].g), F32(Luce_u8[12].b) },
    { F32(Luce_u8[13].r), F32(Luce_u8[13].g), F32(Luce_u8[13].b) },
    { F32(Luce_u8[14].r), F32(Luce_u8[14].g), F32(Luce_u8[14].b) },
    { F32(Luce_u8[15].r), F32(Luce_u8[15].g), F32(Luce_u8[15].b) }
}};


#endif // __IMAGE_LAB_ART_POINTILISM_PAINTER_LUCE_TECHNICS__