#ifndef __IMAGE_LAB_ART_POINTILISM_ARTISTICS_RENDERING__
#define __IMAGE_LAB_ART_POINTILISM_ARTISTICS_RENDERING__

#include <algorithm>
#include <cstdint>
#include <cmath>
#include "ArtPointillismControl.hpp"
#include "Common.hpp"
#include "CommonAuxPixFormat.hpp"
#include "IPainter.hpp"
#include "AlgoJFA.hpp"
#include "PainterFactory.hpp"

// --- LOCAL RNG (For deterministic rendering per frame) ---
struct LCG_RNG
{
    uint32_t state;
    LCG_RNG(int seed) { state = (uint32_t)seed; }
    
    float next_float()
    { // Returns 0.0 - 1.0
        state = state * 1664525u + 1013904223u;
        return (float)state / (float)0xFFFFFFFFu;
    }
    
    float next_range (float min, float max)
    {
        return min + (next_float() * (max - min));
    }
};


// Return type for the decomposition logic
struct DecomposedColor
{
    int idx_p1;
    int idx_p2;
    float ratio; // 0.0 to 1.0 (Amount of P1)
};


struct RenderScratchMemory
{
    float*      acc_L;      // Accumulator for L channel
    float*      acc_a;      // Accumulator for a channel
    float*      acc_b;      // Accumulator for b channel
    int32_t*    acc_count;  // Counter (how many pixels belong to this dot)
    fCIELabPix* avg_colors; // The final averaged color per dot
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
    int width,
    int height,
    const BackgroundArt bg_mode
);

void Init_Canvas
(
    float* RESTRICT canvas_lab,      // Output: Interleaved (L,a,b, L,a,b...)
    const float* RESTRICT src_L,     // Input: Planar L (W*H)
    const float* RESTRICT src_ab,    // Input: Interleaved ab (W*H*2) [a,b, a,b...]
    int width,
    int height,
    const BackgroundArt bg_mode
);


void Integrate_Colors
(
    const int32_t* RESTRICT jfa_map_indices, // From Phase 3 (Seed ID per pixel)
    const float* RESTRICT source_lab,
    int width, int height,
    int num_dots,
    // Scratch buffers (size = num_dots)
    float* RESTRICT acc_L,
    float* RESTRICT acc_a,
    float* RESTRICT acc_b,
    int32_t* RESTRICT acc_count,
    // Output
    fCIELabPix* RESTRICT out_dot_colors
);

void Integrate_Colors
(
    const JFAPixel* RESTRICT jfa_map, // Struct pointer
    const float* RESTRICT src_L,      // Planar L
    const float* RESTRICT src_ab,     // Interleaved ab
    int width, int height,
    int num_dots,
    // Scratch buffers
    float* RESTRICT acc_L,
    float* RESTRICT acc_a,
    float* RESTRICT acc_b,
    int32_t* RESTRICT acc_count,
    // Output
    fCIELabPix* RESTRICT out_dot_colors
);

fCIELabPix Apply_Color_Mode
(
    fCIELabPix& input, 
    int color_mode,      // 0=Scientific, 1=Expressive (from IPainter)
    float user_vibrancy  // -100 to +100 (from User UI)
);

DecomposedColor Decompose
(
    const fCIELabPix& target,
    const fCIELabPix* palette,
    int palette_size
);

void RenderKernel_Cluster
(
    const Point2D& pt,
    const fCIELabPix& target_color,
    const RenderContext& ctx,
    const PointillismRenderParams& params,
    float* RESTRICT canvas,
    int width, int height,
    LCG_RNG& rng
);

void RenderKernel_Mosaic
(
    const Point2D& pt,
    const fCIELabPix& target_color,
    const RenderContext& ctx,
    const PointillismRenderParams& params,
    float* RESTRICT canvas,
    int width, int height,
    LCG_RNG& rng
);

void RenderKernel_Flow
(
    const Point2D& pt,
    const fCIELabPix& target_color,
    const RenderContext& ctx,
    const PointillismRenderParams& params,
    const float* RESTRICT density_map, // Needed for flow calculation
    float* RESTRICT canvas,
    int width, int height,
    LCG_RNG& rng
);


void ArtisticRendering
(
    const Point2D* RESTRICT points, 
    int num_points,
    const int32_t* RESTRICT voronoi_map,
    const float* RESTRICT source_lab,
    const float* RESTRICT density_map, 
    int width, int height,
    const PontillismControls& user_params,
    const RenderScratchMemory& scratch,
    float* RESTRICT canvas_lab
);

void ArtisticRendering
(
    const Point2D* RESTRICT points, 
    int num_points,
    const JFAPixel* RESTRICT voronoi_map,
    const float* RESTRICT src_L,
    const float* RESTRICT src_ab,
    const float* RESTRICT density_map, 
    int width, 
    int height,
    const PontillismControls& user_params,
    const RenderScratchMemory& scratch,
    float* RESTRICT canvas_lab
);


RenderScratchMemory AllocScratchMemory
(
    const int width, 
    const int height
);

RenderScratchMemory AllocScratchMemory
(
    const int32_t maxDots 
);

void FreeScratchMemory
(
    RenderScratchMemory& scratchHandler
);


#endif //__IMAGE_LAB_ART_POINTILISM_ARTISTICS_RENDERING__