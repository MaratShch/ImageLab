#include <atomic>
#include "RetroVisionGui.hpp"

// BGRA order:
// index 0: Blue
// index 1 : Green
// index 2 : Red
// index 3 : Alpha

std::atomic<int32_t> bitmapIdx{ 0 };

void SetBitmapIdx (int32_t idx)
{
    if (idx >= 0 && idx < 15)
        bitmapIdx = idx;
    return;
}

int32_t GetBitmapIdx (void)
{
    const int32_t idx = bitmapIdx;
    return idx;
}


static std::array<Logo, totalBitmaps> bitmapsData{};

static constexpr std::array<int32_t, totalBitmaps> bitmapId =
{
    IDB_BITMAP_ATARI,              // MAIN logo 
    IDB_BITMAP_CGA1,               // CGA1 no intencity 
    IDB_BITMAP_CGA1I,              // CGA1 intencity
    IDB_BITMAP_CGA2,               // CGA2 no intencity
    IDB_BITMAP_CGA2I,              // CGA2 intencity
    IDB_BITMAP_EGA_STANDARD,       // EGA Standard
    IDB_BITMAP_EGA_KQ3,            // EGA King Quest 3
    IDB_BITMAP_EGA_KYRANDIA,       // EGA Legend Of Kyrandia
    IDB_BITMAP_EGA_THEXDER,        // EGA Thexder
    IDB_BITMAP_EGA_DUNE,           // EGA Dune
    IDB_BITMAP_EGA_DOOM,           // EGA Doom
    IDB_BITMAP_EGA_METAL,          // EGA Metal Mutant
    IDB_BITMAP_EGA_WOLFENSTEIN,    // EGA Wolfenstein 3D
    IDB_BITMAP_VGA,                // VGA
    IDB_BITMAP_HERCULES            // HERCULES logo
}; 

const Logo& getBitmap(void)
{
    return bitmapsData[bitmapIdx];
}


bool LoadBitmaps (void)
{
    bool bRet = false;
    const int32_t bitmapsSize = static_cast<int32_t>(bitmapId.size());
    size_t cnt = 0;

    HMODULE resourceDll = GetResourceLibHandler();

    for (int32_t i = 0; i < bitmapsSize; i++)
    {
        // Load bitmap from resource
        HBITMAP hBmp = static_cast<HBITMAP>(LoadImage(resourceDll, MAKEINTRESOURCE(bitmapId[i]), IMAGE_BITMAP, 0, 0, LR_CREATEDIBSECTION));
        if (nullptr == hBmp)
            break;

        BITMAP bmp{};
        GetObject(hBmp, sizeof(BITMAP), &bmp);

        if (bmp.bmBits)
        {
            std::memcpy(bitmapsData[i].data(), bmp.bmBits, bitmapsData[i].size());
        } // if (bmp.bmBits)

        DeleteObject(hBmp);
        hBmp = nullptr;
        cnt++;
    } // for (int32_t i = 0; i < bitmapsSize; i++)

    if (cnt == totalBitmaps)
        bRet = true;

    return bRet;
}
