#ifndef __IMAGE_LAB_ART_POINTILISM_PAINTER_SIGNAC_TECHNICS__
#define __IMAGE_LAB_ART_POINTILISM_PAINTER_SIGNAC_TECHNICS__

#include <array>
#include "ArtPointillismPalette.hpp"
#include "Common.hpp"

using Signac_Palette_u8  = std::array<PEntry<uint8_t>, 24>;
using Signac_Palette_f32 = std::array<PEntry<float>, 24>;

CACHE_ALIGN constexpr Signac_Palette_u8 Signac_u8 =
{{
    // --- 1. The Mediterranean Light (High Key) ---
    { 255, 250, 205 }, // Lemon Chiffon (Sunlight)
    { 255, 228, 181 }, // Moccasin (Warm Light)
    { 255, 215,   0 }, // Gold (Reflection on water)
    { 255, 160, 122 }, // Light Salmon (Sunset clouds) - NEW

    // --- 2. The Port Colors (Vibrant) ---
    { 255,  69,   0 }, // Red Orange (Sails/Buoys)
    { 255, 140,   0 }, // Dark Orange
    { 255,  20, 147 }, // Deep Pink (Signac loved pink houses) - NEW
    { 199,  21, 133 }, // Medium Violet Red

    // --- 3. The Water (Aquas & Blues) ---
    // Signac needs more blues than anyone else.
    {  64, 224, 208 }, // Turquoise (Shallow water)
    {   0, 191, 255 }, // Deep Sky Blue
    {  30, 144, 255 }, // Dodger Blue
    {   0,   0, 205 }, // Medium Blue (Deep water)

    // --- 4. The Shadows (Violet/Mauve) ---
    // Signac used purple shadows almost exclusively.
    { 138,  43, 226 }, // Blue Violet
    { 147, 112, 219 }, // Medium Purple (Bridge color)
    {  75,   0, 130 }, // Indigo (Darkest shadow)
    {  25,  25, 112 }, // Midnight Blue (Structural dark)

    // --- 5. The Vegetation (Vibrant Greens) ---
    {  50, 205,  50 }, // Lime Green
    {  34, 139,  34 }, // Forest Green
    { 154, 205,  50 }, // Yellow Green (Sunlit leaves)
    {  46, 139,  87 }, // Sea Green (Shadow leaves)

    // --- 6. The Bridges (Warmth) ---
    { 210, 105,  30 }, // Chocolate (Wood/Masts)
    { 244, 164,  96 }, // Sandy Brown (Beach)
    { 188, 143, 143 }, // Rosy Brown (Distant buildings) - NEW
    { 255, 255, 255 }  // Pure White (Sparkles on water)
}};

CACHE_ALIGN constexpr Signac_Palette_f32 Signac_f32 =
{{
    { F32(Signac_u8[ 0].r), F32(Signac_u8[ 0].g), F32(Signac_u8[ 0].b) },
    { F32(Signac_u8[ 1].r), F32(Signac_u8[ 1].g), F32(Signac_u8[ 1].b) },
    { F32(Signac_u8[ 2].r), F32(Signac_u8[ 2].g), F32(Signac_u8[ 2].b) },
    { F32(Signac_u8[ 3].r), F32(Signac_u8[ 3].g), F32(Signac_u8[ 3].b) },
    { F32(Signac_u8[ 4].r), F32(Signac_u8[ 4].g), F32(Signac_u8[ 4].b) },
    { F32(Signac_u8[ 5].r), F32(Signac_u8[ 5].g), F32(Signac_u8[ 5].b) },
    { F32(Signac_u8[ 6].r), F32(Signac_u8[ 6].g), F32(Signac_u8[ 6].b) },
    { F32(Signac_u8[ 7].r), F32(Signac_u8[ 7].g), F32(Signac_u8[ 7].b) },
    { F32(Signac_u8[ 8].r), F32(Signac_u8[ 8].g), F32(Signac_u8[ 8].b) },
    { F32(Signac_u8[ 9].r), F32(Signac_u8[ 9].g), F32(Signac_u8[ 9].b) },
    { F32(Signac_u8[10].r), F32(Signac_u8[10].g), F32(Signac_u8[10].b) },
    { F32(Signac_u8[11].r), F32(Signac_u8[11].g), F32(Signac_u8[11].b) },
    { F32(Signac_u8[12].r), F32(Signac_u8[12].g), F32(Signac_u8[12].b) },
    { F32(Signac_u8[13].r), F32(Signac_u8[13].g), F32(Signac_u8[13].b) },
    { F32(Signac_u8[14].r), F32(Signac_u8[14].g), F32(Signac_u8[14].b) },
    { F32(Signac_u8[15].r), F32(Signac_u8[15].g), F32(Signac_u8[15].b) },
    { F32(Signac_u8[16].r), F32(Signac_u8[16].g), F32(Signac_u8[16].b) },
    { F32(Signac_u8[17].r), F32(Signac_u8[17].g), F32(Signac_u8[17].b) },
    { F32(Signac_u8[18].r), F32(Signac_u8[18].g), F32(Signac_u8[18].b) },
    { F32(Signac_u8[19].r), F32(Signac_u8[19].g), F32(Signac_u8[19].b) },
    { F32(Signac_u8[20].r), F32(Signac_u8[20].g), F32(Signac_u8[20].b) },
    { F32(Signac_u8[21].r), F32(Signac_u8[21].g), F32(Signac_u8[21].b) },
    { F32(Signac_u8[22].r), F32(Signac_u8[22].g), F32(Signac_u8[22].b) },
    { F32(Signac_u8[23].r), F32(Signac_u8[23].g), F32(Signac_u8[23].b) }
}};


#endif // __IMAGE_LAB_ART_POINTILISM_PAINTER_SIGNAC_TECHNICS__