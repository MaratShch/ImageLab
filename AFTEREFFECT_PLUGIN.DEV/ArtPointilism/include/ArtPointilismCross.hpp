#ifndef __IMAGE_LAB_ART_POINTILISM_PAINTER_CROSS_TECHNICS__
#define __IMAGE_LAB_ART_POINTILISM_PAINTER_CROSS_TECHNICS__

#include <array>
#include "ArtPointilismPalette.hpp"
#include "Common.hpp"

using Cross_Palette_u8  = std::array<PEntry<uint8_t>, 14>;
using Cross_Palette_f32 = std::array<PEntry<float>, 14>;

CACHE_ALIGN constexpr Cross_Palette_u8 Cross_u8 =
{{
    { 255, 238, 102 },
    { 255, 204,  51 },
    { 255, 153,   0 },
    { 204, 102,   0 },
    {  51, 153,   0 },
    {   0, 102,  51 },
    {   0,  51, 153 },
    {  51, 102, 204 },
    { 153, 204, 255 },
    { 255, 255, 255 },
    {   0,   0,   0 },
    { 128, 128, 128 },
    { 255,  51, 153 },
    { 204, 102, 153 },
}};

CACHE_ALIGN constexpr Cross_Palette_f32 Cross_f32 =
{{
    { F32(Cross_u8[ 0].r), F32(Cross_u8[ 0].g), F32(Cross_u8[ 0].b) },
    { F32(Cross_u8[ 1].r), F32(Cross_u8[ 1].g), F32(Cross_u8[ 1].b) },
    { F32(Cross_u8[ 2].r), F32(Cross_u8[ 2].g), F32(Cross_u8[ 2].b) },
    { F32(Cross_u8[ 3].r), F32(Cross_u8[ 3].g), F32(Cross_u8[ 3].b) },
    { F32(Cross_u8[ 4].r), F32(Cross_u8[ 4].g), F32(Cross_u8[ 4].b) },
    { F32(Cross_u8[ 5].r), F32(Cross_u8[ 5].g), F32(Cross_u8[ 5].b) },
    { F32(Cross_u8[ 5].r), F32(Cross_u8[ 5].g), F32(Cross_u8[ 5].b) },
    { F32(Cross_u8[ 6].r), F32(Cross_u8[ 6].g), F32(Cross_u8[ 6].b) },
    { F32(Cross_u8[ 7].r), F32(Cross_u8[ 7].g), F32(Cross_u8[ 7].b) },
    { F32(Cross_u8[ 8].r), F32(Cross_u8[ 8].g), F32(Cross_u8[ 8].b) },
    { F32(Cross_u8[ 9].r), F32(Cross_u8[ 9].g), F32(Cross_u8[ 9].b) },
    { F32(Cross_u8[10].r), F32(Cross_u8[10].g), F32(Cross_u8[10].b) },
    { F32(Cross_u8[11].r), F32(Cross_u8[11].g), F32(Cross_u8[11].b) },
    { F32(Cross_u8[12].r), F32(Cross_u8[12].g), F32(Cross_u8[12].b) },
    { F32(Cross_u8[13].r), F32(Cross_u8[13].g), F32(Cross_u8[13].b) }
}};

#endif // __IMAGE_LAB_ART_POINTILISM_PAINTER_CROSS_TECHNICS__