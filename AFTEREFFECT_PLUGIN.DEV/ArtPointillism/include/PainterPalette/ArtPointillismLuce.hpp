#ifndef __IMAGE_LAB_ART_POINTILISM_PAINTER_LUCE_TECHNICS__
#define __IMAGE_LAB_ART_POINTILISM_PAINTER_LUCE_TECHNICS__

#include <array>
#include "ArtPointillismPalette.hpp"
#include "Common.hpp"

using Luce_Palette_u8 = std::array<PEntry<uint8_t>, 24>;
using Luce_Palette_f32 = std::array<PEntry<float>, 24>;

CACHE_ALIGN constexpr Luce_Palette_u8 Luce_u8 =
{{
    // --- 1. The Fire & Gaslight (Intense Warmth) ---
    { 255, 250, 240 }, // Floral White (White hot center)
    { 255, 215,   0 }, // Gold (Yellow flame)
    { 255, 140,   0 }, // Dark Orange (Orange flame)
    { 255,  69,   0 }, // Red Orange (Embers)

    // --- 2. The Brick & Rust (Industrial) ---
    { 205,  92,  92 }, // Indian Red (Brick walls)
    { 178,  34,  34 }, // Firebrick (Dark brick)
    { 160,  82,  45 }, // Sienna (Rust)
    { 139,  69,  19 }, // Saddle Brown (Dark wood/Dirt)

    // --- 3. The Pavement & Smoke (Chromatic Grays) ---
    { 119, 136, 153 }, // Light Slate Gray (Concrete/Smoke)
    { 112, 128, 144 }, // Slate Gray (Shadows on street)
    { 188, 143, 143 }, // Rosy Brown (Warm smog)
    { 105, 105, 105 }, // Dim Gray (Soot)

    // --- 4. The Night Sky (Cool Blues) ---
    { 100, 149, 237 }, // Cornflower Blue (Twilight)
    {  70, 130, 180 }, // Steel Blue (Industrial sky)
    {  25,  25, 112 }, // Midnight Blue (Deepest night)
    {  10,  10,  30 }, // Obsidian (Warm black - very dark blue)

    // --- 5. The Shadows (Purples/Teals) ---
    {  72,  61, 139 }, // Dark Slate Blue
    {  75,   0, 130 }, // Indigo
    {  47,  79,  79 }, // Dark Slate Gray (Green-tinted shadow)
    {   0, 100,   0 }, // Dark Green (Shadowed foliage)

    // --- 6. The Accents (Reflections) ---
    { 220,  20,  60 }, // Crimson (Traffic/Warning lights)
    { 240, 230, 140 }, // Khaki (Reflected light on wet street)
    {  95, 158, 160 }, // Cadet Blue (Oxidized copper)
    { 128, 128, 128 }  // Gray (Neutral bridge)
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
    { F32(Luce_u8[15].r), F32(Luce_u8[15].g), F32(Luce_u8[15].b) },
    { F32(Luce_u8[16].r), F32(Luce_u8[16].g), F32(Luce_u8[16].b) },
    { F32(Luce_u8[17].r), F32(Luce_u8[17].g), F32(Luce_u8[17].b) },
    { F32(Luce_u8[18].r), F32(Luce_u8[18].g), F32(Luce_u8[18].b) },
    { F32(Luce_u8[19].r), F32(Luce_u8[19].g), F32(Luce_u8[19].b) },
    { F32(Luce_u8[20].r), F32(Luce_u8[20].g), F32(Luce_u8[20].b) },
    { F32(Luce_u8[21].r), F32(Luce_u8[21].g), F32(Luce_u8[21].b) },
    { F32(Luce_u8[22].r), F32(Luce_u8[22].g), F32(Luce_u8[22].b) },
    { F32(Luce_u8[23].r), F32(Luce_u8[23].g), F32(Luce_u8[23].b) }
}};


#endif // __IMAGE_LAB_ART_POINTILISM_PAINTER_LUCE_TECHNICS__