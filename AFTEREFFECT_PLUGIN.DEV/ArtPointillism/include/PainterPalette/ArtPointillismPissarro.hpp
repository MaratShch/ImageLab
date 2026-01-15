#ifndef __IMAGE_LAB_ART_POINTILISM_PAINTER_PISSARO_TECHNICS__
#define __IMAGE_LAB_ART_POINTILISM_PAINTER_PISSARO_TECHNICS__

#include <array>
#include "ArtPointillismPalette.hpp"
#include "Common.hpp"

using Pissarro_Palette_u8  = std::array<PEntry<uint8_t>, 24>;
using Pissarro_Palette_f32 = std::array<PEntry<float>, 24>;

CACHE_ALIGN constexpr Pissarro_Palette_u8 Pissarro_u8 =
{{
    // --- 1. The Sky & Atmosphere (Soft Tints) ---
    { 255, 250, 240 }, // Floral White (Sunlight base)
    { 230, 230, 250 }, // Lavender (Cloud shadows)
    { 176, 224, 230 }, // Powder Blue (Horizon haze)
    { 245, 245, 220 }, // Beige (Warm light)

    // --- 2. The Fields (Greens - Spring/Summer) ---
    { 154, 205,  50 }, // Yellow Green (Fresh grass)
    { 107, 142,  35 }, // Olive Drab (Hay shadow)
    {  85, 107,  47 }, // Olive Green (Deep foliage)
    {  34, 139,  34 }, // Forest Green (Trees)

    // --- 3. The Harvest (Earths & Yellows) ---
    { 255, 215,   0 }, // Gold (Ripe wheat)
    { 218, 165,  32 }, // Goldenrod (Shadowed wheat)
    { 205, 133,  63 }, // Peru (Dry earth)
    { 139,  69,  19 }, // Saddle Brown (Soil/Wood)

    // --- 4. The Village (Reds & Warmth) ---
    { 255, 160, 122 }, // Light Salmon (Brick highlights/Roofs)
    { 205,  92,  92 }, // Indian Red (Aged brick)
    { 178,  34,  34 }, // Firebrick (Deep red accents)
    { 255, 192, 203 }, // Pink (Flowers/Skin)

    // --- 5. The Distance (Blues) ---
    { 100, 149, 237 }, // Cornflower Blue (Mid-sky)
    {  70, 130, 180 }, // Steel Blue (Distant hills)
    {  65, 105, 225 }, // Royal Blue (Water/Shadows)
    {  25,  25, 112 }, // Midnight Blue (Deepest darks)

    // --- 6. The Bridges (Neutralizers) ---
    { 143, 188, 143 }, // Dark Sea Green (Cool shadow bridge)
    { 119, 136, 153 }, // Light Slate Gray (Overcast sky)
    { 148,   0, 211 }, // Dark Violet (Strong shadow contrast)
    { 105, 105, 105 }  // Dim Gray (Neutral dark)
}};

CACHE_ALIGN constexpr Pissarro_Palette_f32 Pissarro_f32 =
{{
    { F32(Pissarro_u8[ 0].r), F32(Pissarro_u8[ 0].g), F32(Pissarro_u8[ 0].b) },
    { F32(Pissarro_u8[ 1].r), F32(Pissarro_u8[ 1].g), F32(Pissarro_u8[ 1].b) },
    { F32(Pissarro_u8[ 2].r), F32(Pissarro_u8[ 2].g), F32(Pissarro_u8[ 2].b) },
    { F32(Pissarro_u8[ 3].r), F32(Pissarro_u8[ 3].g), F32(Pissarro_u8[ 3].b) },
    { F32(Pissarro_u8[ 4].r), F32(Pissarro_u8[ 4].g), F32(Pissarro_u8[ 4].b) },
    { F32(Pissarro_u8[ 5].r), F32(Pissarro_u8[ 5].g), F32(Pissarro_u8[ 5].b) },
    { F32(Pissarro_u8[ 6].r), F32(Pissarro_u8[ 6].g), F32(Pissarro_u8[ 6].b) },
    { F32(Pissarro_u8[ 7].r), F32(Pissarro_u8[ 7].g), F32(Pissarro_u8[ 7].b) },
    { F32(Pissarro_u8[ 8].r), F32(Pissarro_u8[ 8].g), F32(Pissarro_u8[ 8].b) },
    { F32(Pissarro_u8[ 9].r), F32(Pissarro_u8[ 9].g), F32(Pissarro_u8[ 9].b) },
    { F32(Pissarro_u8[10].r), F32(Pissarro_u8[10].g), F32(Pissarro_u8[10].b) },
    { F32(Pissarro_u8[11].r), F32(Pissarro_u8[11].g), F32(Pissarro_u8[11].b) },
    { F32(Pissarro_u8[12].r), F32(Pissarro_u8[12].g), F32(Pissarro_u8[12].b) },
    { F32(Pissarro_u8[13].r), F32(Pissarro_u8[13].g), F32(Pissarro_u8[13].b) },
    { F32(Pissarro_u8[14].r), F32(Pissarro_u8[14].g), F32(Pissarro_u8[14].b) },
    { F32(Pissarro_u8[15].r), F32(Pissarro_u8[15].g), F32(Pissarro_u8[15].b) },
    { F32(Pissarro_u8[16].r), F32(Pissarro_u8[16].g), F32(Pissarro_u8[16].b) },
    { F32(Pissarro_u8[17].r), F32(Pissarro_u8[17].g), F32(Pissarro_u8[17].b) },
    { F32(Pissarro_u8[18].r), F32(Pissarro_u8[18].g), F32(Pissarro_u8[18].b) },
    { F32(Pissarro_u8[19].r), F32(Pissarro_u8[19].g), F32(Pissarro_u8[19].b) },
    { F32(Pissarro_u8[20].r), F32(Pissarro_u8[20].g), F32(Pissarro_u8[20].b) },
    { F32(Pissarro_u8[21].r), F32(Pissarro_u8[21].g), F32(Pissarro_u8[21].b) },
    { F32(Pissarro_u8[22].r), F32(Pissarro_u8[22].g), F32(Pissarro_u8[22].b) },
    { F32(Pissarro_u8[23].r), F32(Pissarro_u8[23].g), F32(Pissarro_u8[23].b) }
}};

#endif // __IMAGE_LAB_ART_POINTILISM_PAINTER_PISSARO_TECHNICS__