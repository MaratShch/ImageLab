#ifndef __IMAGE_LAB_ART_POINTILISM_PAINTER_VAN_GOGH_TECHNICS__
#define __IMAGE_LAB_ART_POINTILISM_PAINTER_VAN_GOGH_TECHNICS__


#include <array>
#include "ArtPointilismPalette.hpp"
#include "Common.hpp"

using VanGogh_Palette_u8  = std::array<PEntry<uint8_t>, 16>;
using VanGogh_Palette_f32 = std::array<PEntry<float>, 16>;

CACHE_ALIGN constexpr VanGogh_Palette_u8 VanGogh_u8 =
{{
    { 255, 221,  51 },
    { 255, 187,   0 },
    { 255, 102,   0 },
    { 204,  51,   0 },
    {   0,  51, 153 },
    {  51, 102, 204 },
    {   0, 153, 153 },
    { 153, 204,  51 },
    { 102, 153,  51 },
    { 255,   0, 255 },
    { 204,   0, 204 },
    { 255, 255, 255 },
    { 0,     0,   0 },
    { 128, 128, 128 },
    { 255,  51,  51 },
    {  51,   0, 201 }
}};

CACHE_ALIGN constexpr VanGogh_Palette_f32 VanGogh_u8 =
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