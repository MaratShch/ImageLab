#include <array>
#include "RetroVisionGui.hpp"
#include "resource.h"

constexpr size_t BitmapSize    = guiBarWidth * guiBarHeight;
constexpr size_t BitmapMemSize = BitmapSize * 4;
using Logo = std::array<uint8_t, BitmapMemSize>;

static const std::array<LPSTR, 6> bitmapId =
{
    MAKEINTRESOURCE(IDB_BITMAP_CGA1),   // CGA1 no intencity 
    MAKEINTRESOURCE(IDB_BITMAP_CGA1I),  // CGA1 intencity
    MAKEINTRESOURCE(IDB_BITMAP_CGA2),   // CGA2 no intencity
    MAKEINTRESOURCE(IDB_BITMAP_CGA2I),  // CGA2 intencity
//    MAKEINTRESOURCE(IDB_BITMAP_VGA), // VGA logo - VGA256
    MAKEINTRESOURCE(IDB_BITMAP_HERCULES)  // HERCULES logo
}; 


bool LoadBitmaps (void)
{
}


void CloseBitmap (HBITMAP hndl)
{
    if (hndl)
        DeleteObject(hndl);

    hndl = 0;
    return;
}