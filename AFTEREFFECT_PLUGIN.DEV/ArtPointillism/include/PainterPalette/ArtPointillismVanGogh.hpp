#ifndef __IMAGE_LAB_ART_POINTILISM_PAINTER_VAN_GOGH_TECHNICS__
#define __IMAGE_LAB_ART_POINTILISM_PAINTER_VAN_GOGH_TECHNICS__


#include <array>
#include "ArtPointillismPalette.hpp"
#include "Common.hpp"

using VanGogh_Palette_u8  = std::array<PEntry<uint8_t>, 24>;
using VanGogh_Palette_f32 = std::array<PEntry<float>, 24>;

CACHE_ALIGN constexpr VanGogh_Palette_u8 VanGogh_u8 =
{{
    // --- 1. The Sun & Stars (Luminous Yellows) ---
    { 245, 223,  77 }, // Chrome Yellow Lemon (Stars/Light)
    { 255, 215,   0 }, // Chrome Yellow Medium (Sunflowers)
    { 255, 170,  50 }, // Chrome Yellow Deep / Orange
    { 240, 230, 140 }, // Straw Yellow (High-key wheat bridge) - NEW

    // --- 2. The Earths (The "Potato Eaters" Foundation) ---
    // Essential for faces, wood, and ground. Prevents gray skin.
    { 204, 119,  34 }, // Yellow Ochre (Van Gogh's staple)
    { 210, 105,  30 }, // Raw Sienna
    { 139,  69,  19 }, // Burnt Sienna (Reddish earth)
    { 222, 184, 135 }, // Burlywood (Canvas/Beige bridge) - NEW

    // --- 3. The Reds (Contrast) ---
    { 227,  66,  52 }, // Vermilion (The Bedroom floor)
    { 178,  34,  34 }, // Geranium Lake / Carmine (Cool Red)
    { 165,  42,  42 }, // Red Ochre / Brick

    // --- 4. The Greens (Cypresses & Olives) ---
    {  64, 130, 109 }, // Viridian (Deep cool green)
    {  85, 107,  47 }, // Olive Green (Warm muddy green)
    { 107, 142,  35 }, // Olive Drab (Light olive)
    { 154, 205,  50 }, // Veronese Green (Vibrant highlights)

    // --- 5. The Blues (Starry Night) ---
    {  65, 105, 225 }, // Cobalt Blue (Signature sky color)
    {  30,  55, 153 }, // Prussian Blue (Deepest darks)
    { 100, 149, 237 }, // Cornflower Blue (Mid-tone sky)
    {  70, 130, 180 }, // Steel Blue (Shadow bridge)

    // --- 6. The Violets (Shadows) ---
    { 102,  51, 153 }, // Cobalt Violet (Shadows on gold)
    {  75,   0, 130 }, // Indigo (Dark outlines)
    {  47,  79,  79 }, // Dark Slate Gray (Atmospheric Dark)
    { 230, 230, 250 }  // Lavender (Lightest shadow tint) - NEW
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
    { F32(VanGogh_u8[15].r), F32(VanGogh_u8[15].g), F32(VanGogh_u8[15].b) },
    { F32(VanGogh_u8[16].r), F32(VanGogh_u8[16].g), F32(VanGogh_u8[16].b) },
    { F32(VanGogh_u8[17].r), F32(VanGogh_u8[17].g), F32(VanGogh_u8[17].b) },
    { F32(VanGogh_u8[18].r), F32(VanGogh_u8[18].g), F32(VanGogh_u8[18].b) },
    { F32(VanGogh_u8[19].r), F32(VanGogh_u8[19].g), F32(VanGogh_u8[19].b) },
    { F32(VanGogh_u8[20].r), F32(VanGogh_u8[20].g), F32(VanGogh_u8[20].b) },
    { F32(VanGogh_u8[21].r), F32(VanGogh_u8[21].g), F32(VanGogh_u8[21].b) },
    { F32(VanGogh_u8[22].r), F32(VanGogh_u8[22].g), F32(VanGogh_u8[22].b) },
    { F32(VanGogh_u8[23].r), F32(VanGogh_u8[23].g), F32(VanGogh_u8[23].b) }
}};


#endif // __IMAGE_LAB_ART_POINTILISM_PAINTER_VAN_GOGH_TECHNICS__