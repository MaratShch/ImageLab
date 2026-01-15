#ifndef __IMAGE_LAB_ART_POINTILISM_PAINTER_SEURAT_TECHNICS__
#define __IMAGE_LAB_ART_POINTILISM_PAINTER_SEURAT_TECHNICS__

#include <array>
#include "ArtPointillismPalette.hpp"
#include "Common.hpp"

using Seurat_Palette_u8  = std::array<PEntry<uint8_t>, 24>;
using Seurat_Palette_f32 = std::array<PEntry<float>, 24>;

CACHE_ALIGN constexpr Seurat_Palette_u8 Seurat_u8 =
{{
    // --- 1. The Luminous Ground (Atmosphere) ---
    { 255, 248, 231 }, // Canvas / Cream (The base)
    { 255, 255, 240 }, // Ivory (High light)
    { 230, 230, 250 }, // Lavender (Atmospheric shadow bridge) - NEW
    { 176, 224, 230 }, // Powder Blue (Sky haze bridge) - NEW

    // --- 2. The Primary Pigments (Scientific) ---
    { 227,  66,  52 }, // Vermilion (Red)
    { 255, 165,   0 }, // Orange
    { 255, 215,   0 }, // Chrome Yellow
    {  34, 139,  34 }, // Viridian / Forest Green

    // --- 3. The Complementaries (Contrast) ---
    {  65, 105, 225 }, // Royal Blue (French Ultramarine)
    {   0,   0, 139 }, // Dark Blue (Deep shadow)
    { 148,   0, 211 }, // Dark Violet (Contrast to Yellow)
    { 128,   0,   0 }, // Maroon (Contrast to Green) - NEW

    // --- 4. The Earths (Seurat's Foundation) ---
    // Critical for his "Grande Jatte" park scenes
    { 205, 133,  63 }, // Peru / Raw Sienna
    { 218, 165,  32 }, // Goldenrod
    { 160,  82,  45 }, // Burnt Sienna
    { 101,  67,  33 }, // Dark Brown / Van Dyke Brown - NEW

    // --- 5. The "Dusty" Bridges (Anti-Neon) ---
    // These desaturated colors prevent the "Neon" look.
    { 107, 142,  35 }, // Olive Drab (Grass shadow)
    { 189, 183, 107 }, // Dark Khaki (Sunlit dust)
    { 119, 136, 153 }, // Light Slate Gray (Neutralizer)
    {  47,  79,  79 }, // Dark Slate Gray (The "Black" replacement)

    // --- 6. The Flesh & Warmth ---
    { 255, 192, 203 }, // Pink (Flesh highlight)
    { 233, 150, 122 }, // Dark Salmon (Flesh shadow)
    { 210, 105,  30 }, // Chocolate (Warm Dark) - NEW
    {  70, 130, 180 }  // Steel Blue (Cool Dark) - NEW
}};

CACHE_ALIGN constexpr Seurat_Palette_f32 Seurat_f32 =
{{
    { F32(Seurat_u8[ 0].r), F32(Seurat_u8[ 0].g), F32(Seurat_u8[ 0].b) },
    { F32(Seurat_u8[ 1].r), F32(Seurat_u8[ 1].g), F32(Seurat_u8[ 1].b) },
    { F32(Seurat_u8[ 2].r), F32(Seurat_u8[ 2].g), F32(Seurat_u8[ 2].b) },
    { F32(Seurat_u8[ 3].r), F32(Seurat_u8[ 3].g), F32(Seurat_u8[ 3].b) },
    { F32(Seurat_u8[ 4].r), F32(Seurat_u8[ 4].g), F32(Seurat_u8[ 4].b) },
    { F32(Seurat_u8[ 5].r), F32(Seurat_u8[ 5].g), F32(Seurat_u8[ 5].b) },
    { F32(Seurat_u8[ 6].r), F32(Seurat_u8[ 6].g), F32(Seurat_u8[ 6].b) },
    { F32(Seurat_u8[ 7].r), F32(Seurat_u8[ 7].g), F32(Seurat_u8[ 7].b) },
    { F32(Seurat_u8[ 8].r), F32(Seurat_u8[ 8].g), F32(Seurat_u8[ 8].b) },
    { F32(Seurat_u8[ 9].r), F32(Seurat_u8[ 9].g), F32(Seurat_u8[ 9].b) },
    { F32(Seurat_u8[10].r), F32(Seurat_u8[10].g), F32(Seurat_u8[10].b) },
    { F32(Seurat_u8[11].r), F32(Seurat_u8[11].g), F32(Seurat_u8[11].b) },
    { F32(Seurat_u8[12].r), F32(Seurat_u8[12].g), F32(Seurat_u8[12].b) },
    { F32(Seurat_u8[13].r), F32(Seurat_u8[13].g), F32(Seurat_u8[13].b) },
    { F32(Seurat_u8[14].r), F32(Seurat_u8[14].g), F32(Seurat_u8[14].b) },
    { F32(Seurat_u8[15].r), F32(Seurat_u8[15].g), F32(Seurat_u8[15].b) },
    { F32(Seurat_u8[16].r), F32(Seurat_u8[16].g), F32(Seurat_u8[16].b) },
    { F32(Seurat_u8[17].r), F32(Seurat_u8[17].g), F32(Seurat_u8[17].b) },
    { F32(Seurat_u8[18].r), F32(Seurat_u8[18].g), F32(Seurat_u8[18].b) },
    { F32(Seurat_u8[19].r), F32(Seurat_u8[19].g), F32(Seurat_u8[19].b) },
    { F32(Seurat_u8[20].r), F32(Seurat_u8[20].g), F32(Seurat_u8[20].b) },
    { F32(Seurat_u8[21].r), F32(Seurat_u8[21].g), F32(Seurat_u8[21].b) },
    { F32(Seurat_u8[22].r), F32(Seurat_u8[22].g), F32(Seurat_u8[22].b) },
    { F32(Seurat_u8[23].r), F32(Seurat_u8[23].g), F32(Seurat_u8[23].b) }
}};


#endif // __IMAGE_LAB_ART_POINTILISM_PAINTER_SEURAT_TECHNICS__