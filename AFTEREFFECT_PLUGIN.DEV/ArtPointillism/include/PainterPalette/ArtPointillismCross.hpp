#ifndef __IMAGE_LAB_ART_POINTILISM_PAINTER_CROSS_TECHNICS__
#define __IMAGE_LAB_ART_POINTILISM_PAINTER_CROSS_TECHNICS__

#include <array>
#include "ArtPointillismPalette.hpp"
#include "Common.hpp"

using Cross_Palette_u8  = std::array<PEntry<uint8_t>, 24>;
using Cross_Palette_f32 = std::array<PEntry<float>, 24>;

CACHE_ALIGN constexpr Cross_Palette_u8 Cross_u8 =
{{
    // --- 1. The Light & Atmosphere (High Luma) ---
    { 255, 245, 200 }, // Cream / Warm White (Sunlight base)
    { 240, 230, 140 }, // Naples Yellow (Sand/High-key earth) - NEW
    { 255, 236,  50 }, // Lemon Yellow (Brightest pure chroma)
    { 216, 191, 216 }, // Thistle (Pale Violet Haze - Anti-Gray Bridge) - NEW

    // --- 2. The Sun & Heat (Yellows/Oranges) ---
    { 255, 195,  11 }, // Chrome Yellow Deep
    { 255, 165,   0 }, // Orange
    { 255, 140,   0 }, // Dark Orange / Cadmium
    { 255, 127,  80 }, // Coral (Pinkish-Orange Bridge) - NEW

    // --- 3. The Earths (Structure/Skin) ---
    { 218, 165,  32 }, // Goldenrod (Sunlit Earth)
    { 204, 119,  34 }, // Yellow Ochre (Essential for Skin)
    { 210, 105,  30 }, // Chocolate / Raw Sienna (Warm Earth) - NEW
    { 160,  82,  45 }, // Burnt Sienna (Dark Earth)

    // --- 4. The Pinks & Violets (Shadows on Sand) ---
    { 255, 105, 180 }, // Hot Pink / Rose Madder
    { 216, 112, 147 }, // Pale Violet Red (Muted Pink)
    { 205,  92,  92 }, // Indian Red (Brick/Clay)
    { 148,   0, 211 }, // Dark Violet (Deep Shadows)

    // --- 5. The Sea (Blues/Teals) ---
    {  64, 224, 208 }, // Turquoise (Signature Riviera water)
    {   0, 128, 128 }, // Teal (Deep water)
    {  70, 130, 180 }, // Steel Blue (Sky)
    {   0,  47, 167 }, // French Ultramarine (True Blue)

    // --- 6. The Vegetation & Depth (Greens/Darks) ---
    {  50, 205,  50 }, // Lime Green (Sunlit foliage)
    {  34, 139,  34 }, // Forest Green (Shadow foliage)
    {  47,  79,  79 }, // Dark Slate Gray (Atmospheric Dark)
    {  25,  25, 112 }  // Midnight Blue (The "Black" Replacement) - NEW
}};


CACHE_ALIGN constexpr Cross_Palette_f32 Cross_f32 =
{{
    { F32(Cross_u8[ 0].r), F32(Cross_u8[ 0].g), F32(Cross_u8[ 0].b) },
    { F32(Cross_u8[ 1].r), F32(Cross_u8[ 1].g), F32(Cross_u8[ 1].b) },
    { F32(Cross_u8[ 2].r), F32(Cross_u8[ 2].g), F32(Cross_u8[ 2].b) },
    { F32(Cross_u8[ 3].r), F32(Cross_u8[ 3].g), F32(Cross_u8[ 3].b) },
    { F32(Cross_u8[ 4].r), F32(Cross_u8[ 4].g), F32(Cross_u8[ 4].b) },
    { F32(Cross_u8[ 5].r), F32(Cross_u8[ 5].g), F32(Cross_u8[ 5].b) },
    { F32(Cross_u8[ 6].r), F32(Cross_u8[ 6].g), F32(Cross_u8[ 6].b) },
    { F32(Cross_u8[ 7].r), F32(Cross_u8[ 7].g), F32(Cross_u8[ 7].b) },
    { F32(Cross_u8[ 8].r), F32(Cross_u8[ 8].g), F32(Cross_u8[ 8].b) },
    { F32(Cross_u8[ 9].r), F32(Cross_u8[ 9].g), F32(Cross_u8[ 9].b) },
    { F32(Cross_u8[10].r), F32(Cross_u8[10].g), F32(Cross_u8[10].b) },
    { F32(Cross_u8[11].r), F32(Cross_u8[11].g), F32(Cross_u8[11].b) },
    { F32(Cross_u8[12].r), F32(Cross_u8[12].g), F32(Cross_u8[12].b) },
    { F32(Cross_u8[13].r), F32(Cross_u8[13].g), F32(Cross_u8[13].b) },
    { F32(Cross_u8[14].r), F32(Cross_u8[14].g), F32(Cross_u8[14].b) },
    { F32(Cross_u8[15].r), F32(Cross_u8[15].g), F32(Cross_u8[15].b) },
    { F32(Cross_u8[16].r), F32(Cross_u8[16].g), F32(Cross_u8[16].b) },
    { F32(Cross_u8[17].r), F32(Cross_u8[17].g), F32(Cross_u8[17].b) },
    { F32(Cross_u8[18].r), F32(Cross_u8[18].g), F32(Cross_u8[18].b) },
    { F32(Cross_u8[19].r), F32(Cross_u8[19].g), F32(Cross_u8[19].b) },
    { F32(Cross_u8[20].r), F32(Cross_u8[20].g), F32(Cross_u8[20].b) },
    { F32(Cross_u8[21].r), F32(Cross_u8[21].g), F32(Cross_u8[21].b) },
    { F32(Cross_u8[22].r), F32(Cross_u8[22].g), F32(Cross_u8[22].b) },
    { F32(Cross_u8[23].r), F32(Cross_u8[23].g), F32(Cross_u8[23].b) }
}};

#endif // __IMAGE_LAB_ART_POINTILISM_PAINTER_CROSS_TECHNICS__