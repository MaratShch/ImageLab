#ifndef __IMAGE_LAB_ART_POINTILISM_ARTISTICS_RENDERING__
#define __IMAGE_LAB_ART_POINTILISM_ARTISTICS_RENDERING__

#include <algorithm>
#include <cstdint>
#include <cmath>
#include "Common.hpp"

// Return type for the decomposition logic
struct ColorMix
{
    int32_t index_p1; // Index of Primary Palette Color
    int32_t index_p2; // Index of Secondary Palette Color
    float ratio;      // 0.0 to 1.0 (Probability of picking P1)
};

// --- GENERIC RENDER CONTROL PARAMETERS ---
// These are passed from the UI sliders to Phase 4.
// They apply to Seurat, Signac, Matisse, etc.
struct PointillismRenderParams
{
    int32_t DotSize;        // 0-100 (Controls overlap/radius)
    int32_t Vibrancy;       // 0-100 (Saturation boost)
    int32_t BackgroundMode; // 0=Canvas, 1=White, 2=Transparent, 3=Source
    int32_t RandomSeed;     // For deterministic jitter
    int32_t Opacity;        // 0-100 (Blend with original - Phase 5)
};


void Init_Canvas
(
    float* RESTRICT canvas_lab,
    const float* RESTRICT source_lab,
    int32_t width, 
    int32_t height, 
    int32_t mode
);


#endif //__IMAGE_LAB_ART_POINTILISM_ARTISTICS_RENDERING__