#ifndef __IMAGE_LAB_RETRO_VISION_FILTER_GUI__
#define __IMAGE_LAB_RETRO_VISION_FILTER_GUI__

#include <array>
#include <windows.h>
#include "RetroVisionEnum.hpp"
#include "RetroVisionResource.hpp"

constexpr int32_t guiBarWidth = 48;
constexpr int32_t guiBarHeight = 48;

constexpr size_t BitmapSize = guiBarWidth * guiBarHeight;
constexpr size_t BitmapMemSize = BitmapSize * 4;
constexpr size_t totalBitmaps = 16;
using Logo = std::array<uint8_t, BitmapMemSize>;

HMODULE GetResourceLibHandler(void);
bool LoadBitmaps (void);

void SetBitmapIdx(int32_t idx);
int32_t GetBitmapIdx(void);
const Logo& getBitmap(void);

#endif // __IMAGE_LAB_RETRO_VISION_FILTER_GUI__