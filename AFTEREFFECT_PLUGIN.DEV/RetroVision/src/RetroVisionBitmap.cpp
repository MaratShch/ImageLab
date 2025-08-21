#include <array>
#include "RetroVisionGui.hpp"
#include "resource.h"


static const std::array<LPSTR, 6> bitmapId =
{
    MAKEINTRESOURCE(IDB_BITMAP_CGA1),   // CGA1 no intencity 
    MAKEINTRESOURCE(IDB_BITMAP_CGA1I),  // CGA1 intencity
    MAKEINTRESOURCE(IDB_BITMAP_CGA2),   // CGA2 no intencity
    MAKEINTRESOURCE(IDB_BITMAP_CGA2I),  // CGA2 intencity
//    MAKEINTRESOURCE(IDB_BITMAP_VGA), // VGA logo - VGA256
    MAKEINTRESOURCE(IDB_BITMAP_HERCULES)  // HERCULES logo
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