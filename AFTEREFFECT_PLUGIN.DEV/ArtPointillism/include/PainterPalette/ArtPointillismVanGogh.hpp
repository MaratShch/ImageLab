#ifndef __IMAGE_LAB_ART_POINTILISM_PAINTER_VAN_GOGH_TECHNICS__
#define __IMAGE_LAB_ART_POINTILISM_PAINTER_VAN_GOGH_TECHNICS__


#include <array>
#include "ArtPointillismPalette.hpp"
#include "Common.hpp"

using VanGogh_Palette_u8  = std::array<PEntry<uint8_t>, 17>;
using VanGogh_Palette_f32 = std::array<PEntry<float>, 17>;

CACHE_ALIGN constexpr VanGogh_Palette_u8 VanGogh_u8 =
{{
    // --- The Yellows/Oranges (Sunflowers & Wheat) ---
    { 245, 223,  77 }, // Chrome Yellow Lemon (Less digital yellow)
    { 255, 172,   0 }, // Chrome Yellow Deep
    { 227, 130,   0 }, // Cadmium Orange
    { 218, 165,  32 }, // Goldenrod / Sunflower Yellow

    // --- The Earths (CRITICAL for "Glue") ---
    { 204, 119,  34 }, // Yellow Ochre (Essential for skin/fur)
    { 139,  69,  19 }, // Burnt Sienna (Warm brown shadows)

    // --- The Reds ---
    { 227,  66,  52 }, // Vermilion (Orangey-Red)
    { 178,  34,  34 }, // Carmine / Madder Lake (Cool Red)

    // --- The Blues (Starry Night) ---
    {  30,  55, 153 }, // Prussian Blue (Deep dark)
    {  65, 105, 225 }, // Cobalt Blue (The signature blue)
    { 135, 206, 235 }, // Cerulean Blue (Sky)

    // --- The Greens (Cypresses & Olives) ---
    {  64, 130, 109 }, // Viridian (Blue-Green, deep)
    {  85, 107,  47 }, // Olive Green (Earthy green)
    { 154, 205,  50 }, // Yellow Green (Light grass, less neon)

    // --- The Violets (Irises) ---
    { 102,  51, 153 }, // Cobalt Violet (Deep purple, NOT Magenta)
    
    // --- Neutrals ---
    { 245, 245, 220 }, // Beige/Cream (Canvas color replacement for pure white)
    {  47,  79,  79 }  // Dark Slate (Van Gogh rarely used pure black)
}};


CACHE_ALIGN constexpr VanGogh_Palette_f32 VanGogh_f32 =
{{
    { F32(VanGogh_u8[ 0].r), F32(VanGogh_u8[ 0].g), F32(VanGogh_u8[ 0].b) },
    { F32(VanGogh_u8[ 1].r), F32(VanGogh_u8[ 1].g), F32(VanGogh_u8[ 1].b) },
    { F32(VanGogh_u8[ 2].r), F32(VanGogh_u8[ 2].g), F32(VanGogh_u8[ 2].b) },
    { F32(VanGogh_u8[ 3].r), F32(VanGogh_u8[ 3].g), F32(VanGogh_u8[ 3].b) },
    { F32(VanGogh_u8[ 4].r), F32(VanGogh_u8[ 4].g), F32(VanGogh_u8[ 4].b) },
    { F32(VanGogh_u8[ 5].r), F32(VanGogh_u8[ 5].g), F32(VanGogh_u8[ 5].b) },
    { F32(VanGogh_u8[ 6].r), F32(VanGogh_u8[ 6].g), F32(VanGogh_u8[ 6].b) },
    { F32(VanGogh_u8[ 7].r), F32(VanGogh_u8[ 7].g), F32(VanGogh_u8[ 7].b) },
    { F32(VanGogh_u8[ 8].r), F32(VanGogh_u8[ 8].g), F32(VanGogh_u8[ 8].b) },
    { F32(VanGogh_u8[ 9].r), F32(VanGogh_u8[ 9].g), F32(VanGogh_u8[ 9].b) },
    { F32(VanGogh_u8[10].r), F32(VanGogh_u8[10].g), F32(VanGogh_u8[10].b) },
    { F32(VanGogh_u8[11].r), F32(VanGogh_u8[11].g), F32(VanGogh_u8[11].b) },
    { F32(VanGogh_u8[12].r), F32(VanGogh_u8[12].g), F32(VanGogh_u8[12].b) },
    { F32(VanGogh_u8[13].r), F32(VanGogh_u8[13].g), F32(VanGogh_u8[13].b) },
    { F32(VanGogh_u8[14].r), F32(VanGogh_u8[14].g), F32(VanGogh_u8[14].b) },
    { F32(VanGogh_u8[15].r), F32(VanGogh_u8[15].g), F32(VanGogh_u8[15].b) }
}};


#endif // __IMAGE_LAB_ART_POINTILISM_PAINTER_VAN_GOGH_TECHNICS__