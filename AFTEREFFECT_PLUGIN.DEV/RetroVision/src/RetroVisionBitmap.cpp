#include <array>
#include "RetroVisionGui.hpp"
#include "resource.h"


static const std::array<LPSTR, 6> bitmapId =
{
    MAKEINTRESOURCE(IDB_BITMAP_CGA), // CGA logo - CGA1
    MAKEINTRESOURCE(IDB_BITMAP_CGA), // CGA logo - CGA2
    MAKEINTRESOURCE(IDB_BITMAP_EGA), // EGA logo - EGA
    MAKEINTRESOURCE(IDB_BITMAP_VGA), // VGA logo - VGA16
    MAKEINTRESOURCE(IDB_BITMAP_VGA), // VGA logo - VGA256
    MAKEINTRESOURCE(IDB_BITMAP_HER)  // HERCULES logo
}; 


BITMAP LoadBitmap (const RetroBitmap& bitmap, HBITMAP& hndl)
{
    BITMAP bmp{};

    // Load bitmap from resource
    hndl = static_cast<HBITMAP>(
            LoadImage(GetModuleHandle(NULL), bitmapId[UnderlyingType(bitmap)], IMAGE_BITMAP, 0, 0, LR_CREATEDIBSECTION)
        );

    if (hndl)
        GetObject (hndl, sizeof(BITMAP), &bmp);

    return bmp;
}


void CloseBitmap (HBITMAP hndl)
{
    if (hndl)
        DeleteObject(hndl);

    hndl = 0;
    return;
}