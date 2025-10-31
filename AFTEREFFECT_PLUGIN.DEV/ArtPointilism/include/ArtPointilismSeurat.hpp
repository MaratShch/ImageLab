#ifndef __IMAGE_LAB_ART_POINTILISM_PAINTER_SEURAT_TECHNICS__
#define __IMAGE_LAB_ART_POINTILISM_PAINTER_SEURAT_TECHNICS__

#include <array>
#include "ArtPointilismPalette.hpp"
#include "Common.hpp"

using Seurat_Palette_u8  = std::array<PEntry<uint8_t>, 12>;
using Seurat_Palette_f32 = std::array<PEntry<float>, 12>;

CACHE_ALIGN constexpr Seurat_Palette_u8 Seurat_u8 =
{{
    { 255,   0,   0 },
    { 255,  64,   0 },
    { 255, 255,   0 },
    { 255, 255, 128 },
    {   0, 128,   0 },
    {   0, 255, 128 },
    {   0,   0, 255 },
    { 128,   0, 255 },
    { 255,   0, 255 },
    { 255, 255, 255 },
    {   0,   0,   0 },
    { 128, 128, 128 }
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
    { F32(Seurat_u8[11].r), F32(Seurat_u8[11].g), F32(Seurat_u8[11].b) }
}};


#endif // __IMAGE_LAB_ART_POINTILISM_PAINTER_SEURAT_TECHNICS__