#include <vector>
#include <cmath>
#include "CommonDebugUtils.hpp"
#include "RetroVisionPalette.hpp"
#include "RetroVisionAlgorithm.hpp"
#include "RetroVisionEnum.hpp"
#include "RetroVisionControls.hpp"


inline void ScanLines_SimulationHelper
(
    const fRGB* __restrict in,
          fRGB* __restrict out,
    const A_long sizeX,
    const A_long sizeY,
    const A_long interval,
    const A_long smooth,
    const float  darkness
) noexcept
{
    float darkenFactor = 0.f;

    for (A_long j = 0; j < sizeY; j++)
    {
        const A_long nearest_center = std::round(j / interval) * interval;
        const A_long distance = FastCompute::Abs(j - nearest_center);

        const fRGB* __restrict srcLine = in  + j * sizeX;
              fRGB* __restrict dstLine = out + j * sizeX;

        if (0 == distance)
        {
            darkenFactor = 1.f - darkness;
        }
        else if (distance <= smooth)
        {
           const float weight = FastCompute::Max(0.f, 1.f - (static_cast<float>(distance) / static_cast<float>(smooth + 1)));
           const float darken_center = 1.f - darkness;
           darkenFactor = (1.f - weight) + weight * darken_center;
        }
        else
        {
           darkenFactor = 1.f;
           std::memcpy (dstLine, srcLine, sizeX * sizeof(fRGB));
           continue;
        }

        for (A_long i = 0; i < sizeX; i++)
        {
            dstLine[i].R * srcLine[i].R * darkenFactor;
            dstLine[i].G * srcLine[i].G * darkenFactor;
            dstLine[i].B * srcLine[i].B * darkenFactor;
        }

    } // for (A_long j = 0; j < sizeY; j++)

    return;
}


inline void PhosphorGlow_SimulationHelper
(
    const fRGB* __restrict in,
          fRGB* __restrict out,
    const A_long sizeX,
    const A_long sizeY,
    const float strength,
    const float opacity
) noexcept
{
    return;
}


inline void AppertureGrill_SimulationHelper
(
    const fRGB* __restrict in,
          fRGB* __restrict out,
    const A_long sizeX,
    const A_long sizeY,
    const AppertureGtrill type,
    const int32_t interval,
    const float darkness,
    const int32_t color
) noexcept
{
    return;
}


void ScanLines_Simulation
(
    const fRGB** input,
          fRGB** output,
    A_long sizeX,
    A_long sizeY,
    const RVControls& controlParams
)
{
    if (0 != controlParams.scan_lines_enable)
    {
        const fRGB* __restrict pInput  = *input;
              fRGB* __restrict pOutput = *output;

        const A_long interval = static_cast<A_long>(controlParams.scan_lines_interval);
        const A_long smooth   = static_cast<A_long>(controlParams.scan_lines_smooth);
        const float  darkness = controlParams.scan_lines_darkness;

        ScanLines_SimulationHelper (pInput, pOutput, sizeX, sizeY, interval, smooth, darkness);
    }
    else
        *output = const_cast<fRGB*>(*input); // nothing to do - pass by
    
    return;
}


void PhosphorGlow_Simulation
(
    const fRGB** input,
          fRGB** output,
    A_long sizeX,
    A_long sizeY,
    const RVControls& controlParams
)
{
    if (0 != controlParams.phosphor_glow_enable)
    {
        const fRGB* __restrict pInput = *input;
              fRGB* __restrict pOutput = *output;

        const float strength = controlParams.phosphor_glow_strength;
        const float opacity  = controlParams.phosphor_glow_opacity;

        PhosphorGlow_SimulationHelper (pInput, pOutput, sizeX, sizeY, strength, opacity);
    }
    else
        *output = const_cast<fRGB*>(*input); // nothing to do - pass by

    return;

}


void AppertureGrill_Simulation
(
    const fRGB** input,
          fRGB** output,
    A_long sizeX,
    A_long sizeY,
    const RVControls& controlParams
)
{
    if (0 != controlParams.apperture_grill_enable)
    {
        const fRGB* __restrict pInput = *input;
              fRGB* __restrict pOutput = *output;

        const AppertureGtrill type = controlParams.mask_type;
        const int32_t interval = controlParams.mask_interval;
        const float   darkness = controlParams.mask_darkness;
        const int32_t color = controlParams.mask_color;

        AppertureGrill_SimulationHelper (pInput, pOutput, sizeX, sizeY, type, interval, darkness, color);
    }
    else
        *output = const_cast<fRGB*>(*input); // nothing to do - pass by

    return;

}


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
                Vga_Simulation16 (input, output, sizeX, sizeY, VGA_Standard16_f32);
            else
                Vga_Simulation256 (input, output, sizeX, sizeY, VGA_Standard256_f32);
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

#if defined(__INTEL_COMPILER) || defined(__INTEL_LLVM_COMPILER)
    #pragma warning(push)
    #pragma warning(disable:2308)
#endif

    // Scan Lines CRT Artifacts
    ScanLines_Simulation (&input, &output, sizeX, sizeY, controlParams);

    // PhosphorGlow (a.k.a. CRT Bloom) CRT Artifacts
    PhosphorGlow_Simulation (&input, &output, sizeX, sizeY, controlParams);

    // Apperture Grill CRT Artifacts
    AppertureGrill_Simulation (&input, &output, sizeX, sizeY, controlParams);

#if defined(__INTEL_COMPILER) || defined(__INTEL_LLVM_COMPILER)
    #pragma warning(pop)
#endif

    return;
}