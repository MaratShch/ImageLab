#include <vector>
#include <cmath>
#include "CommonDebugUtils.hpp"
#include "RetroVisionPalette.hpp"
#include "RetroVisionAlgorithm.hpp"
#include "RetroVisionEnum.hpp"
#include "RetroVisionControls.hpp"


void RetroResolution_Simulation
(
    const fRGB* __restrict input,
          fRGB* __restrict output,
    A_long sizeX,
    A_long sizeY,
    const RVControls& controlParams
)
{
    // Simulate Retro-Monitor view
    switch (controlParams.monitor)
    {
        case RetroMonitor::eRETRO_BITMAP_CGA:
        {
            const CGA_PaletteF32& palette = (PaletteCGA::eRETRO_PALETTE_CGA1 == controlParams.cga_palette ?
                (0 == controlParams.cga_intencity_bit ? CGA0_f32 : CGA0i_f32) :
                    (0 == controlParams.cga_intencity_bit ? CGA1_f32 : CGA1i_f32));

            CGA_Simulation (input, output, sizeX, sizeY, palette);
        }
        break;

        case RetroMonitor::eRETRO_BITMAP_EGA:
        {
            const EGA_PaletteF32& palette = getEgaPalette(controlParams.ega_palette);
            EGA_Simulation (input, output, sizeX, sizeY, palette);
        }
        break;

        case RetroMonitor::eRETRO_BITMAP_VGA:
        {
            if (PaletteVGA::eRETRO_PALETTE_VGA_16_BITS == controlParams.vga_palette)
                Vga_Simulation (input, output, sizeX, sizeY, VGA_Standard16_f32);
            else
                Vga_Simulation (input, output, sizeX, sizeY, VGA_Standard256_f32);
        }
        break;

        case RetroMonitor::eRETRO_BITMAP_HERCULES:
        default:
        {
            const PEntry<float> whiteLevel = HERCULES_White_ColorF32[UnderlyingType(controlParams.white_color_hercules)];
            const float threshold = static_cast<float>(controlParams.hercules_threshold) / 255.f;
            Hercules_Simulation (input, output, sizeX, sizeY, threshold, whiteLevel);
        }
        break;
    }

    // Simulate CRT Artifacts
    if (0 != controlParams.scan_lines_enable)
    {
        // Simulate Scan lines
    }

    if (0 != controlParams.phosphor_glow_enable)
    {
        // PhosphorGlow effect enable
    }

    return;
}