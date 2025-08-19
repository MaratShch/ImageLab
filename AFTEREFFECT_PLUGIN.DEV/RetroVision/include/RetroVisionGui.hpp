#ifndef __IMAGE_LAB_RETRO_VISION_FILTER_GUI__
#define __IMAGE_LAB_RETRO_VISION_FILTER_GUI__

#include <windows.h>
#include "RetroVisionEnum.hpp"


BITMAP LoadBitmap(const RetroBitmap& bitmap, HBITMAP& hndl);
void CloseBitmap(HBITMAP hndl);

#endif // __IMAGE_LAB_RETRO_VISION_FILTER_GUI__