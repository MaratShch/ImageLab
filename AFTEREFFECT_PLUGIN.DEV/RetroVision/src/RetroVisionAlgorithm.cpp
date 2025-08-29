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
    CACHE_ALIGN const CGA_PaletteF32 p = {{
        { static_cast<float>(palette[0].r) / 255.f, static_cast<float>(palette[0].g) / 255.f, static_cast<float>(palette[0].b) / 255.f },
        { static_cast<float>(palette[1].r) / 255.f, static_cast<float>(palette[1].g) / 255.f, static_cast<float>(palette[1].b) / 255.f },
        { static_cast<float>(palette[2].r) / 255.f, static_cast<float>(palette[2].g) / 255.f, static_cast<float>(palette[2].b) / 255.f },
        { static_cast<float>(palette[3].r) / 255.f, static_cast<float>(palette[3].g) / 255.f, static_cast<float>(palette[3].b) / 255.f }
    }};

    const A_long hBlocks = (sizeX <= CGA_width  ? 1 : sizeX / CGA_width);
    const A_long vBlocks = (sizeY <= CGA_height ? 1 : sizeY / CGA_height);

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
    CACHE_ALIGN const EGA_PaletteF32 p = {{
        { static_cast<float>(palette[0].r ) / 255.f, static_cast<float>(palette[0].g ) / 255.f, static_cast<float>(palette[0].b ) / 255.f },
        { static_cast<float>(palette[1].r ) / 255.f, static_cast<float>(palette[1].g ) / 255.f, static_cast<float>(palette[1].b ) / 255.f },
        { static_cast<float>(palette[2].r ) / 255.f, static_cast<float>(palette[2].g ) / 255.f, static_cast<float>(palette[2].b ) / 255.f },
        { static_cast<float>(palette[3].r ) / 255.f, static_cast<float>(palette[3].g ) / 255.f, static_cast<float>(palette[3].b ) / 255.f },
        { static_cast<float>(palette[4].r ) / 255.f, static_cast<float>(palette[4].g ) / 255.f, static_cast<float>(palette[4].b ) / 255.f },
        { static_cast<float>(palette[5].r ) / 255.f, static_cast<float>(palette[5].g ) / 255.f, static_cast<float>(palette[5].b ) / 255.f },
        { static_cast<float>(palette[6].r ) / 255.f, static_cast<float>(palette[6].g ) / 255.f, static_cast<float>(palette[6].b ) / 255.f },
        { static_cast<float>(palette[7].r ) / 255.f, static_cast<float>(palette[7].g ) / 255.f, static_cast<float>(palette[7].b ) / 255.f },
        { static_cast<float>(palette[8].r ) / 255.f, static_cast<float>(palette[8].g ) / 255.f, static_cast<float>(palette[8].b ) / 255.f },
        { static_cast<float>(palette[9].r ) / 255.f, static_cast<float>(palette[9].g ) / 255.f, static_cast<float>(palette[9].b ) / 255.f },
        { static_cast<float>(palette[10].r) / 255.f, static_cast<float>(palette[10].g) / 255.f, static_cast<float>(palette[10].b) / 255.f },
        { static_cast<float>(palette[11].r) / 255.f, static_cast<float>(palette[11].g) / 255.f, static_cast<float>(palette[11].b) / 255.f },
        { static_cast<float>(palette[12].r) / 255.f, static_cast<float>(palette[12].g) / 255.f, static_cast<float>(palette[12].b) / 255.f },
        { static_cast<float>(palette[13].r) / 255.f, static_cast<float>(palette[13].g) / 255.f, static_cast<float>(palette[13].b) / 255.f },
        { static_cast<float>(palette[14].r) / 255.f, static_cast<float>(palette[14].g) / 255.f, static_cast<float>(palette[14].b) / 255.f },
        { static_cast<float>(palette[15].r) / 255.f, static_cast<float>(palette[15].g) / 255.f, static_cast<float>(palette[15].b) / 255.f }
    }};

    const A_long hBlocks = (sizeX <= EGA_width ? 1  : sizeX / EGA_width);
    const A_long vBlocks = (sizeY <= EGA_height ? 1 : sizeY / EGA_height);

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