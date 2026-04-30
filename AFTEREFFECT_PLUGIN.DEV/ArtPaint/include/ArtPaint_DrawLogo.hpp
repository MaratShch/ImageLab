#ifndef __IMAGE_LAB_ART_PAINT_DRAW_LOGO_MODULE__
#define __IMAGE_LAB_ART_PAINT_DRAW_LOGO_MODULE__

#include <array>
#include "CommonAdobeAE.hpp"

constexpr size_t logoWidth = 100;
constexpr size_t logoHeight = 30;

constexpr size_t logoSize = logoWidth * logoHeight;
constexpr size_t logoMemSize = logoSize * 4;

using Logo = std::array<uint8_t, logoMemSize>;

bool LoadResourceDll(PF_InData* in_data);
void FreeResourceDll(void);
bool LoadLogo(void);
const Logo& getBitmap(void);


PF_Err DrawEvent
(
    PF_InData		*in_data,
    PF_OutData		*out_data,
    PF_ParamDef		*params[],
    PF_LayerDef		*output,
    PF_EventExtra	*event_extra
);

#endif // __IMAGE_LAB_ART_PAINT_DRAW_LOGO_MODULE__