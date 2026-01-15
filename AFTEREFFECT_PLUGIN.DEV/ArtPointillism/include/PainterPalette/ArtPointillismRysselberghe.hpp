#ifndef __IMAGE_LAB_ART_POINTILISM_PAINTER_RYSSELBERGHE_TECHNICS__
#define __IMAGE_LAB_ART_POINTILISM_PAINTER_RYSSELBERGHE_TECHNICS__

#include <array>
#include "ArtPointillismPalette.hpp"
#include "Common.hpp"

using Rysselberghe_Palette_u8 = std::array<PEntry<uint8_t>, 24>;
using Rysselberghe_Palette_f32 = std::array<PEntry<float>, 24>;

CACHE_ALIGN constexpr Rysselberghe_Palette_u8 Rysselberghe_u8 =
{{
    // --- 1. The Skin Tones (Luminosity) ---
    { 255, 239, 213 }, // Papaya Whip (High highlight)
    { 255, 218, 185 }, // Peach Puff (Warm skin)
    { 255, 160, 122 }, // Light Salmon (Blush)
    { 233, 150, 122 }, // Dark Salmon (Shadowed skin)

    // --- 2. The Jewel Tones (Clothing/Backgrounds) ---
    { 123, 104, 238 }, // Medium Slate Blue (Velvet)
    {  72,  61, 139 }, // Dark Slate Blue (Deep velvet)
    { 138,  43, 226 }, // Blue Violet (Rich purple)
    {  75,   0, 130 }, // Indigo (Darkest purple)

    // --- 3. The Sea & Eyes (Teals) ---
    {  64, 224, 208 }, // Turquoise (Bright)
    {   0, 128, 128 }, // Teal (Mid)
    {  47,  79,  79 }, // Dark Slate Gray (Deep teal shadow)
    {  32, 178, 170 }, // Light Sea Green

    // --- 4. The Warmth (Oranges/Golds) ---
    { 255, 215,   0 }, // Gold (Jewelry/Light)
    { 255, 140,   0 }, // Dark Orange
    { 210, 105,  30 }, // Chocolate (Wood/Dark hair)
    { 160,  82,  45 }, // Sienna

    // --- 5. The Reds ---
    { 220,  20,  60 }, // Crimson (Lips/Fabric)
    { 128,   0,   0 }, // Maroon (Deep red shadow)
    { 255, 105, 180 }, // Hot Pink (Flowers)
    { 219, 112, 147 }, // Pale Violet Red (Muted rose)

    // --- 6. The Depth (Blues) ---
    {  65, 105, 225 }, // Royal Blue
    {   0,   0, 205 }, // Medium Blue
    {  25,  25, 112 }, // Midnight Blue (Replaces black)
    { 240, 248, 255 }  // Alice Blue (Cool white highlight)
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
    { F32(Rysselberghe_u8[15].r), F32(Rysselberghe_u8[15].g), F32(Rysselberghe_u8[15].b) },
    { F32(Rysselberghe_u8[16].r), F32(Rysselberghe_u8[16].g), F32(Rysselberghe_u8[16].b) },
    { F32(Rysselberghe_u8[17].r), F32(Rysselberghe_u8[17].g), F32(Rysselberghe_u8[17].b) },
    { F32(Rysselberghe_u8[18].r), F32(Rysselberghe_u8[18].g), F32(Rysselberghe_u8[18].b) },
    { F32(Rysselberghe_u8[19].r), F32(Rysselberghe_u8[19].g), F32(Rysselberghe_u8[19].b) },
    { F32(Rysselberghe_u8[20].r), F32(Rysselberghe_u8[20].g), F32(Rysselberghe_u8[20].b) },
    { F32(Rysselberghe_u8[21].r), F32(Rysselberghe_u8[21].g), F32(Rysselberghe_u8[21].b) },
    { F32(Rysselberghe_u8[22].r), F32(Rysselberghe_u8[22].g), F32(Rysselberghe_u8[22].b) },
    { F32(Rysselberghe_u8[23].r), F32(Rysselberghe_u8[23].g), F32(Rysselberghe_u8[23].b) }
}};


#endif // __IMAGE_LAB_ART_POINTILISM_PAINTER_RYSSELBERGHE_TECHNICS__