#ifndef __IMAGE_LAB_RETRO_VISION_PALETTE_EGA_VALUES__
#define __IMAGE_LAB_RETRO_VISION_PALETTE_EGA_VALUES__

#include "PaletteEntry.hpp"
#include <array>

constexpr int32_t EGA_width  = 640;
constexpr int32_t EGA_height = 350;

// EGA standard palette
constexpr std::array<PEntry<uint8_t>, 16> EGA_Standard_u8 =
{ {
    {   0,   0,   0 },
    {   0,   0, 170 },
    {   0, 170,   0 },
    {   0, 170, 170 },
    { 170,   0,   0 },
    { 170,   0, 170 },
    { 170,  85,   0 },
    { 170, 170, 170 },
    {  85,  85,  85 },
    {  85,  85, 255 },
    {  85, 255,  85 },
    {  85, 255, 255 },
    { 255,  85,  85 },
    { 255,  85, 255 },
    { 255, 255,  85 },
    { 255, 255, 255 }
}};


#endif // __IMAGE_LAB_RETRO_VISION_PALETTE_EGA_VALUES__