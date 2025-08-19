#include <array>
#include "RetroVisionGui.hpp"
#include "resource.h"


static const std::array<LPSTR, 4> bitmapId =
{
    MAKEINTRESOURCE(IDB_BITMAP_CGA),
    MAKEINTRESOURCE(IDB_BITMAP_EGA),
    MAKEINTRESOURCE(IDB_BITMAP_VGA),
    MAKEINTRESOURCE(IDB_BITMAP_HER)
};


BITMAP LoadBitmap (const RetroBitmap& bitmap, HBITMAP& hndl)
{
    BITMAP bmp{};

    // Load bitmap from resource
    hndl = static_cast<HBITMAP>(
            LoadImage(GetModuleHandle(NULL), bitmapId[underlying(bitmap) - underlying(RetroBitmap::eRETRO_BITMAP_CGA)], IMAGE_BITMAP, 0, 0, LR_CREATEDIBSECTION)
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