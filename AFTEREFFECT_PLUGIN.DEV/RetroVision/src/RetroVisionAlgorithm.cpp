#define FAST_COMPUTE_EXTRA_PRECISION

#include <vector>
#include <array>
#include <cmath>
#include "CommonDebugUtils.hpp"
#include "RetroVisionPalette.hpp"
#include "RetroVisionAlgorithm.hpp"
#include "RetroVisionEnum.hpp"
#include "RetroVisionControls.hpp"


inline float compute_gaussian (float sigma, float x) noexcept
{
    return (1.f / (FastCompute::Sqrt (FastCompute::PIx2) * sigma)) * FastCompute::Exp(-(x * x) / (2.f * sigma * sigma));
}


std::vector<float> compute_kernel (float glow_strength, float sigma = 20.0f)
{
    // Ensure odd size, larger for more glow
    const int32_t kernel_size = 0x1 | static_cast<int32_t>(std::floor(glow_strength * 2.f) * 2.f + 1.f);
    const int32_t radius = kernel_size >> 1;
    float cumSum = 0.f;

    std::vector<float> v (kernel_size);
    std::vector<float> kernel (kernel_size);

    // compute kernel coefficients
    for (int32_t i = -radius, idx = 0; i <= radius; i++, idx++)
    {
        v[idx] = compute_gaussian (sigma, i);
        cumSum += v[idx];
    }

    // normalize kernel coefficients
    for (int32_t i = 0; i < kernel_size; i++)
        kernel[i] = v[i] / cumSum;

    return kernel;
}

void ScanLines_SimulationHelper
(
    fRGB* __restrict in,
    fRGB* __restrict out,
    const A_long sizeX,
    const A_long sizeY,
    const A_long interval,
    const A_long smooth,
    const float  darkness
)
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
           darkenFactor = 1.f; // no changes in this image line  - just copy from source to destination
           std::memcpy (dstLine, srcLine, sizeX * sizeof(fRGB));
           continue;
        }

        for (A_long i = 0; i < sizeX; i++)
        {
            dstLine[i].R = srcLine[i].R * darkenFactor;
            dstLine[i].G = srcLine[i].G * darkenFactor;
            dstLine[i].B = srcLine[i].B * darkenFactor;
        }

    } // for (A_long j = 0; j < sizeY; j++)

    return;
}

inline fRGB HorizontalGaussianBlur
(
    const fRGB* current,
    A_long left,
    A_long right,
    std::vector<float> kernel
)
{
    fRGB out{};
    for (A_long i = -left, idx = 0; i <= right; i++, idx++)
    {
        out.R += (current[i].R * kernel[idx]);
        out.G += (current[i].G * kernel[idx]);
        out.B += (current[i].B * kernel[idx]);
    }
    return out;
}

inline fRGB VerticalGaussianBlur
(
    const fRGB* current,
    A_long top,
    A_long down,
    A_long pitch,
    std::vector<float> kernel
)
{
    fRGB out{};
    for (A_long i = -top, idx = 0; i <= down; i++, idx++)
    {
        const A_long j = i * pitch;
        out.R += (current[j].R * kernel[idx]);
        out.G += (current[j].G * kernel[idx]);
        out.B += (current[j].B * kernel[idx]);
    }
    return out;
}

void PhosphorGlow_SimulationHelper
(
    fRGB* __restrict in,
    fRGB* __restrict out,
    const A_long sizeX,
    const A_long sizeY,
    const float strength,
    const float opacity
)
{
    const std::vector<float> kernel = compute_kernel (strength);
    const A_long halfKernel = static_cast<A_long>(kernel.size()) >> 1;

    for (A_long j = 0; j < sizeY; j++)
    {
        const A_long currLineIdx = j * sizeX;
        const A_long top  = FastCompute::Min(halfKernel, FastCompute::Max(0, j - halfKernel));
        const A_long down = FastCompute::Min(halfKernel, (sizeY - 1) - j);

        for (A_long i = 0; i < sizeX; i++)
        {
            const fRGB* current = in + currLineIdx + i;

            // compute Horizontal bluring result
            const A_long left  = FastCompute::Min(halfKernel, FastCompute::Max(0, i - halfKernel));
            const A_long right = FastCompute::Min(halfKernel, (sizeX - 1) - i);

            // horizontal blur result  
            const fRGB hBlur = HorizontalGaussianBlur (current, left, right, kernel);

            // vertical blur result
            const fRGB vBlur = VerticalGaussianBlur (current, top, down, sizeX, kernel);

            const float glowR = (hBlur.R + vBlur.R) * 0.5f;
            const float glowG = (hBlur.G + vBlur.G) * 0.5f;
            const float glowB = (hBlur.B + vBlur.B) * 0.5f;

            const float glow_r_contrib = FastCompute::Max(0.f, glowR - current->R);
            const float glow_g_contrib = FastCompute::Max(0.f, glowG - current->G);
            const float glow_b_contrib = FastCompute::Max(0.f, glowB - current->B);

            // Final color in normalized float values
            const float finalR = current->R + glow_r_contrib * opacity;
            const float finalG = current->G + glow_g_contrib * opacity;
            const float finalB = current->B + glow_b_contrib * opacity;

            out[currLineIdx + i].R = CLAMP_VALUE(finalR, 0.f, 1.f);
            out[currLineIdx + i].G = CLAMP_VALUE(finalG, 0.f, 1.f);
            out[currLineIdx + i].B = CLAMP_VALUE(finalB, 0.f, 1.f);
        }
    }
    return;
}


void ScanLines_Simulation
(
    fRGB* input,
    fRGB* output,
    A_long sizeX,
    A_long sizeY,
    const RVControls& controlParams
)
{
    const A_long interval = static_cast<A_long>(controlParams.scan_lines_interval);
    const A_long smooth = static_cast<A_long>(controlParams.scan_lines_smooth);
    const float  darkness = controlParams.scan_lines_darkness;

    ScanLines_SimulationHelper (input, output, sizeX, sizeY, interval, smooth, darkness);
    return;
}


void PhosphorGlow_Simulation
(
    fRGB* input,
    fRGB* output,
    A_long sizeX,
    A_long sizeY,
    const RVControls& controlParams
)
{
    const float strength = controlParams.phosphor_glow_strength;
    const float opacity = controlParams.phosphor_glow_opacity;

    PhosphorGlow_SimulationHelper (input, output, sizeX, sizeY, strength, opacity);
    return;
}


void AppertureGrill_SimulationApperture
(
    fRGB* __restrict in,
    fRGB* __restrict out,
    const A_long sizeX,
    const A_long sizeY,
    const int32_t interval,
    const float darkness,
    const int32_t color
)
{
    constexpr A_long phaseSize = 3;
    const float wR[phaseSize] = { 1.f, 1.f - darkness, 1.f - darkness };
    const float wG[phaseSize] = { 1.f - darkness, 1.f, 1.f - darkness };
    const float wB[phaseSize] = { 1.f - darkness, 1.f - darkness, 1.f };

    for (A_long j = 0; j < sizeY; j++)
    {
        const A_long lineIdx = j * sizeX;

        for (A_long i = 0; i < sizeX; i++)
        {
            const A_long idx = lineIdx + i;
            const A_long phase = (i / interval) % phaseSize;

            out[idx].R = in[idx].R * wR[phase];
            out[idx].G = in[idx].G * wG[phase];
            out[idx].B = in[idx].B * wB[phase];
        }
    }

    return;
}


void AppertureGrill_SimulationShadowMask
(
    fRGB* __restrict in,
    fRGB* __restrict out,
    const A_long sizeX,
    const A_long sizeY,
    const int32_t interval,
    const float darkness,
    const int32_t color
)
{
    constexpr A_long phaseSize = 3;
    const float overall = 1.0f - 0.5f * darkness;

    const float wR[phaseSize] = { 1.0f * overall, 0.7f * overall, 0.7f * overall };
    const float wG[phaseSize] = { 0.7f * overall, 1.0f * overall, 0.7f * overall };
    const float wB[phaseSize] = { 0.7f * overall, 0.7f * overall, 1.0f * overall };

    for (A_long j = 0; j < sizeY; j++)
    {
        const A_long lineIdx = j * sizeX;

        for (A_long i = 0; i < sizeX; i++)
        {
            const A_long idx = lineIdx + i;
            const A_long phase = (i / interval) % phaseSize;

            out[idx].R = in[idx].R * wR[phase];
            out[idx].G = in[idx].G * wG[phase];
            out[idx].B = in[idx].B * wB[phase];
        }
    }

    return;
}



void AppertureGrill_Simulation
(
    fRGB* __restrict input,
    fRGB* __restrict output,
    A_long sizeX,
    A_long sizeY,
    const RVControls& controlParams
)
{
    const AppertureGtrill type = controlParams.mask_type;
    const int32_t interval = controlParams.mask_interval;
    const float   darkness = controlParams.mask_darkness;
    const int32_t color = controlParams.mask_color;

    if (AppertureGtrill::eRETRO_APPERTURE_SHADOW_MASK == controlParams.mask_type)
        AppertureGrill_SimulationApperture (input, output, sizeX, sizeY, interval, darkness, color);
    else
        AppertureGrill_SimulationShadowMask (input, output, sizeX, sizeY, interval, darkness, color);

    return;
}


fRGB* RetroResolution_Simulation
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

    fRGB* scanLinesIn  = nullptr;
    fRGB* scanLinesOut = nullptr;
    fRGB* phosphorGlowIn  = nullptr;
    fRGB* phosphorGlowOut = nullptr;
    fRGB* appertureLinesIn = nullptr;
    fRGB* appertureLinesOut = nullptr;

    // Scan Lines CRT Artifacts
    if (0 != controlParams.scan_lines_enable)
    {
        scanLinesIn = output;
        scanLinesOut = const_cast<fRGB*>(input);
        ScanLines_Simulation(scanLinesIn, scanLinesOut, sizeX, sizeY, controlParams);
    }
    else
    {
        scanLinesOut = output;
        scanLinesIn = const_cast<fRGB*>(input);
    }

    // PhosphorGlow (a.k.a. CRT Bloom) CRT Artifacts
    if (0 != controlParams.phosphor_glow_enable)
    {
        phosphorGlowIn  = scanLinesOut;
        phosphorGlowOut = scanLinesIn;
        PhosphorGlow_Simulation (phosphorGlowIn, phosphorGlowOut, sizeX, sizeY, controlParams);
    }
    else
    {
        phosphorGlowOut = scanLinesOut;
        phosphorGlowIn  = scanLinesIn;
    }

    // Apperture Grill CRT Artifacts
    if (0 != controlParams.apperture_grill_enable)
    {
        appertureLinesIn  = phosphorGlowOut;
        appertureLinesOut = phosphorGlowIn;
        AppertureGrill_Simulation (appertureLinesIn, appertureLinesOut, sizeX, sizeY, controlParams);
    }
    else
    {
        appertureLinesOut = phosphorGlowOut;
    }

#if defined(__INTEL_COMPILER) || defined(__INTEL_LLVM_COMPILER)
    #pragma warning(pop)
#endif

    return appertureLinesOut;
}