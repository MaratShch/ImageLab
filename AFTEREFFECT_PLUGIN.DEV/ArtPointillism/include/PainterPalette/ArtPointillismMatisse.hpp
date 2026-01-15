#ifndef __IMAGE_LAB_ART_POINTILISM_PAINTER_MATISSE_TECHNICS__
#define __IMAGE_LAB_ART_POINTILISM_PAINTER_MATISSE_TECHNICS__

#include <array>
#include "ArtPointillismPalette.hpp"
#include "Common.hpp"
    
using Matisse_Palette_u8  = std::array<PEntry<uint8_t>, 24>;
using Matisse_Palette_f32 = std::array<PEntry<float>, 24>;

CACHE_ALIGN constexpr Matisse_Palette_u8 Matisse_u8 =
{{
    // --- 1. The Pinks & Magentas (The "Red Studio") ---
    // Matisse used these where others used brown or gray.
    { 255,  20, 147 }, // Deep Pink / Rose Madder (Vibrant)
    { 255, 105, 180 }, // Hot Pink (Light)
    { 199,  21, 133 }, // Medium Violet Red (Shadow)
    { 255, 192, 203 }, // Light Pink (Highlight bridge) - NEW

    // --- 2. The Fauve Reds & Oranges ---
    { 255,  69,   0 }, // Cadmium Red (Pure energy)
    { 220,  20,  60 }, // Crimson Lake
    { 255, 140,   0 }, // Cadmium Orange
    { 255, 160, 122 }, // Light Salmon (Skin bridge) - NEW

    // --- 3. The Mediterranean Blues ---
    {  30, 144, 255 }, // Dodger Blue / Cobalt
    {   0,   0, 205 }, // Medium Blue / Ultramarine
    {  64, 224, 208 }, // Turquoise (The view from the window)
    { 176, 224, 230 }, // Powder Blue (Lightest tint) - NEW

    // --- 4. The Emeralds & Teals ---
    {   0, 168, 107 }, // Emerald Green (The "Green Stripe")
    {  50, 205,  50 }, // Lime Green (Pigment based)
    {  32, 178, 170 }, // Light Sea Green
    {   0, 128, 128 }, // Teal (Dark contrast)

    // --- 5. The Violets (Structure) ---
    { 138,  43, 226 }, // Blue Violet
    { 148,   0, 211 }, // Dark Violet
    { 147, 112, 219 }, // Medium Purple (Lavender)

    // --- 6. The Bridges (No Gray!) ---
    // Matisse didn't use gray. He used colored neutrals.
    { 255, 255,  51 }, // Lemon Yellow (Light)
    { 255, 228, 181 }, // Moccasin (Warm Neutral)
    {  25,  25, 112 }, // Midnight Blue (The "Black" replacement)
    { 240, 255, 240 }  // Honeydew (Canvas White)
}};

CACHE_ALIGN constexpr Matisse_Palette_f32 Matisse_f32 =
{{
    { F32(Matisse_u8[ 0].r), F32(Matisse_u8[ 0].g), F32(Matisse_u8[ 0].b) },
    { F32(Matisse_u8[ 1].r), F32(Matisse_u8[ 1].g), F32(Matisse_u8[ 1].b) },
    { F32(Matisse_u8[ 2].r), F32(Matisse_u8[ 2].g), F32(Matisse_u8[ 2].b) },
    { F32(Matisse_u8[ 3].r), F32(Matisse_u8[ 3].g), F32(Matisse_u8[ 3].b) },
    { F32(Matisse_u8[ 4].r), F32(Matisse_u8[ 4].g), F32(Matisse_u8[ 4].b) },
    { F32(Matisse_u8[ 5].r), F32(Matisse_u8[ 5].g), F32(Matisse_u8[ 5].b) },
    { F32(Matisse_u8[ 6].r), F32(Matisse_u8[ 6].g), F32(Matisse_u8[ 6].b) },
    { F32(Matisse_u8[ 7].r), F32(Matisse_u8[ 7].g), F32(Matisse_u8[ 7].b) },
    { F32(Matisse_u8[ 8].r), F32(Matisse_u8[ 8].g), F32(Matisse_u8[ 8].b) },
    { F32(Matisse_u8[ 9].r), F32(Matisse_u8[ 9].g), F32(Matisse_u8[ 9].b) },
    { F32(Matisse_u8[10].r), F32(Matisse_u8[10].g), F32(Matisse_u8[10].b) },
    { F32(Matisse_u8[11].r), F32(Matisse_u8[11].g), F32(Matisse_u8[11].b) },
    { F32(Matisse_u8[12].r), F32(Matisse_u8[12].g), F32(Matisse_u8[12].b) },
    { F32(Matisse_u8[13].r), F32(Matisse_u8[13].g), F32(Matisse_u8[13].b) },
    { F32(Matisse_u8[14].r), F32(Matisse_u8[14].g), F32(Matisse_u8[14].b) },
    { F32(Matisse_u8[15].r), F32(Matisse_u8[15].g), F32(Matisse_u8[15].b) },
    { F32(Matisse_u8[16].r), F32(Matisse_u8[16].g), F32(Matisse_u8[16].b) },
    { F32(Matisse_u8[17].r), F32(Matisse_u8[17].g), F32(Matisse_u8[17].b) },
    { F32(Matisse_u8[18].r), F32(Matisse_u8[18].g), F32(Matisse_u8[18].b) },
    { F32(Matisse_u8[19].r), F32(Matisse_u8[19].g), F32(Matisse_u8[19].b) },
    { F32(Matisse_u8[20].r), F32(Matisse_u8[20].g), F32(Matisse_u8[20].b) },
    { F32(Matisse_u8[21].r), F32(Matisse_u8[21].g), F32(Matisse_u8[21].b) },
    { F32(Matisse_u8[22].r), F32(Matisse_u8[22].g), F32(Matisse_u8[22].b) },
    { F32(Matisse_u8[23].r), F32(Matisse_u8[23].g), F32(Matisse_u8[23].b) }
}};


#endif // __IMAGE_LAB_ART_POINTILISM_PAINTER_MATISSE_TECHNICS__