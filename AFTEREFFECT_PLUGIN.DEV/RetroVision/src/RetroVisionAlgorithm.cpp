#include "RetroVisionAlgorithm.hpp"
#include "RetroVisionEnum.hpp"


void CGA_Simulation
(
    const fRGB* __restrict input,
          fRGB* __restrict output,
    A_long sizeX,
    A_long sizeY,
    const CGA_Palette& palette
)
{
    return;
}


void EGA_Simulation
(
    const fRGB* __restrict input,
          fRGB* __restrict output,
    A_long sizeX,
    A_long sizeY,
    const EGA_Palette& palette
)
{
    return;
}


void VGA16_Simulation
(
    const fRGB* __restrict input,
          fRGB* __restrict output,
    A_long sizeX,
    A_long sizeY,
    const VGA_Palette16& palette
)
{
    return;
}


void VGA256_Simulation
(
    const fRGB* __restrict input,
          fRGB* __restrict output,
    A_long sizeX,
    A_long sizeY,
    const VGA_Palette256& palette
)
{
    return;
}


void Hercules_Simulation
(
    const fRGB* __restrict input,
    fRGB* __restrict output,
    A_long sizeX,
    A_long sizeY,
    float threshold
)
{
    return;
}


void RetroResolution_Simulation
(
    const fRGB* __restrict input,
          fRGB* __restrict output,
    A_long sizeX,
    A_long sizeY,
    PF_ParamDef* __restrict params[]
)
{
    // check moitor type
    const RetroMonitor monitor = static_cast<RetroMonitor>(params[UnderlyingType(RetroVision::eRETRO_VISION_DISPLAY)]->u.bd.value - 1);
    switch (monitor)
    {
        case RetroMonitor::eRETRO_BITMAP_CGA:
        {
            const PaletteCGA palette = static_cast<PaletteCGA>(params[UnderlyingType(RetroVision::eRETRO_VISION_CGA_PALETTE)]->u.bd.value - 1);
            const A_long intencity = params[UnderlyingType(RetroVision::eRETRO_VISION_CGA_INTTENCITY_BIT)]->u.bd.value;
            switch (palette)
            {
                case PaletteCGA::eRETRO_PALETTE_CGA1:
                {
                    const CGA_Palette& CgaValues = (0 == intencity ? CGA0_u8 : CGA0i_u8);
                    CGA_Simulation(input, output, sizeX, sizeY, CgaValues);
                }
                break;
                case PaletteCGA::eRETRO_PALETTE_CGA2:
                {
                    const CGA_Palette& CgaValues = (0 == intencity ? CGA1_u8 : CGA1i_u8);
                    CGA_Simulation (input, output, sizeX, sizeY, CgaValues);
                }
                break;
            }
        }
        break;

        case RetroMonitor::eRETRO_BITMAP_EGA:
        {
            const PaletteEGA palette = static_cast<PaletteEGA>(params[UnderlyingType(RetroVision::eRETRO_VISION_EGA_PALETTE)]->u.bd.value - 1);
            switch (palette)
            {
                case PaletteEGA::eRETRO_PALETTE_EGA_STANDARD:
                    EGA_Simulation (input, output, sizeX, sizeY, EGA_Standard_u8);
                break;

                case PaletteEGA::eRETRO_PALETTE_EGA_KING_QUESTS:
                    EGA_Simulation (input, output, sizeX, sizeY, EGA_KQ3_u8);
                break;

                case PaletteEGA::eRETRO_PALETTE_EGA_KYRANDIA:
                    EGA_Simulation (input, output, sizeX, sizeY, EGA_Kyrandia_u8);
                break;

                case PaletteEGA::eRETRO_PALETTE_EGA_THEXDER:
                    EGA_Simulation (input, output, sizeX, sizeY, EGA_Thexder_u8);
                break;

                case PaletteEGA::eRETRO_PALETTE_EGA_DUNE:
                    EGA_Simulation (input, output, sizeX, sizeY, EGA_Dune_u8);
                break;

                case PaletteEGA::eRETRO_PALETTE_EGA_DOOM:
                    EGA_Simulation (input, output, sizeX, sizeY, EGA_Doom_u8);
                break;

                case PaletteEGA::eRETRO_PALETTE_EGA_METAL_MUTANT:
                    EGA_Simulation (input, output, sizeX, sizeY, EGA_MetalMutant_u8);
                break;

                case PaletteEGA::eRETRO_PALETTE_EGA_WOLFENSTEIN:
                default:
                    EGA_Simulation (input, output, sizeX, sizeY, EGA_Wolfenstein_u8);
                break;
            }
        }
        break;

        case RetroMonitor::eRETRO_BITMAP_VGA:
        {
            const PaletteVGA palette = static_cast<PaletteVGA>(params[UnderlyingType(RetroVision::eRETRO_VISION_VGA_PALETTE)]->u.bd.value - 1);
            if (PaletteVGA::eRETRO_PALETTE_VGA_16_BITS == palette)
                VGA16_Simulation  (input, output, sizeX, sizeY, VGA_Standard16_u8);
            else
                VGA256_Simulation (input, output, sizeX, sizeY, VGA_Standard256_u8);
        }
        break;

        case RetroMonitor::eRETRO_BITMAP_HERCULES:
        default:
        {
            const float binThreshold = 0.5f;
            Hercules_Simulation(input, output, sizeX, sizeY, binThreshold);
        }
        break;
    }
    return;
}