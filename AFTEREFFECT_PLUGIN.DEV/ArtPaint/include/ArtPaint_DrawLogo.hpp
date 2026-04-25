#ifndef __IMAGE_LAB_ART_PAINT_DRAW_LOGO_MODULE__
#define __IMAGE_LAB_ART_PAINT_DRAW_LOGO_MODULE__

#include <array>
#include "CommonAdobeAE.hpp"

constexpr size_t logoSize = 100 * 30;
using Logo = std::array<int32_t, logoSize>;

bool LoadResourceDll(PF_InData* in_data);
void FreeResourceDll(void);
bool LoadLogo(void);
const Logo getBitmap(void);

#endif // __IMAGE_LAB_ART_PAINT_DRAW_LOGO_MODULE__