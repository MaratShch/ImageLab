#include <iostream>
#include "FastAriphmetics.hpp"
#include "AlgoDot2D.hpp"
#include "AlgoArtisticsRendering.hpp"
#include "ArtPointillismEnums.hpp"
#include "PainterFactory.hpp" // Assuming this is where GetPainter lives



// --- BLENDING HELPER (Inline for speed) ---
// Blends a source color onto the canvas using alpha.
inline void Blend_Lab_Pixel
(
    float* RESTRICT pixel_ptr, 
    const fCIELabPix& color, 
    float alpha
) noexcept
{
    float inv_alpha = 1.0f - alpha;
    pixel_ptr[0] = (color.L * alpha) + (pixel_ptr[0] * inv_alpha);
    pixel_ptr[1] = (color.a * alpha) + (pixel_ptr[1] * inv_alpha);
    pixel_ptr[2] = (color.b * alpha) + (pixel_ptr[2] * inv_alpha);
}



/**
 * Pre-process the target color based on Painter Mode.
 * Handles "Vibrancy" and "Expressive" shifts.
 */
fCIELabPix Apply_Color_Mode
(
    fCIELabPix& input, 
    int color_mode,      // 0=Scientific, 1=Expressive (from IPainter)
    float user_vibrancy  // -100 to +100 (from User UI)
)
{
    // 1. Calculate Chroma (Saturation in Lab)
//    float chroma = std::sqrt(input.a * input.a + input.b * input.b);
    float chroma = FastCompute::Sqrt(input.a * input.a + input.b * input.b);
    
    // 2. Determine Saturation Boost Factor
    float boost = 1.0f;

    // Base user adjustment (-1.0 to +1.0)
    float user_factor = user_vibrancy / 100.0f; 
    boost += user_factor;

    // Painter Specific Logic
    if (color_mode == 1)
    { // MODE_EXPRESSIVE (Matisse/Van Gogh)
        // Expressive painters exaggerated color.
        // We implicitly boost saturation to force the decomposition 
        // to pick more vivid palette colors.
        boost *= 1.3f; 
    }

    // 3. Apply Boost
    if (chroma > 0.001f && boost != 1.0f)
    {
        input.a *= boost;
        input.b *= boost;
    }

    return input;
}


/**
 * Pre-process the target color based on Painter Mode.
 * UPDATED: Adds aggressive Chroma Expansion for Expressive modes to prevent "Gray Soup".
 */
fCIELabPix Apply_Color_Mode_Boost
(
    fCIELabPix& input, 
    int color_mode,      // 0=Scientific, 1=Expressive
    float user_vibrancy  // -100 to +100 (from User UI)
)
{
    // 1. Calculate current Chroma (Saturation intensity)
    float chroma = FastCompute::Sqrt(input.a * input.a + input.b * input.b);
    
    // 2. Base Boost from User Slider
    // Map -100..100 to factor 0.0..3.0
    float boost = 1.0f + (user_vibrancy / 50.0f);
    if (boost < 0.0f) boost = 0.0f;

    // 3. EXPRESSIVE MODE LOGIC (The Van Gogh Fix)
    if (color_mode == 1) // MODE_EXPRESSIVE
    {
        // A. Intrinsic Boost
        // Van Gogh is naturally more vibrant.
        boost *= 1.5f; 

        // B. The "Gray Killer" (Chroma Floor)
        // If a color is weak (grayish) but not pure black/white, 
        // we artificially inflate its color purity so it snaps to a colorful palette entry 
        // instead of a gray one.
        
        // Threshold: 5.0 is a subtle gray. 
        if (chroma < 10.0f && chroma > 0.5f) 
        {
            // If it's a weak color, pretend it's a strong color.
            // This forces the "Decompose" function to find a Blue/Yellow match 
            // instead of a Gray match.
            float fake_chroma = 20.0f + (user_vibrancy * 0.2f); 
            float scale = fake_chroma / chroma;
            
            input.a *= scale;
            input.b *= scale;
            
            // Recalculate chroma for the next step
            chroma = fake_chroma; 
        }
    }

    // 4. Apply Final Boost
    if (chroma > 0.001f && boost != 1.0f)
    {
        input.a *= boost;
        input.b *= boost;
    }

    return input;
}


/**
 * Find the 2 closest colors in the palette and their mixing ratio.
 */
DecomposedColor Decompose
(
    const fCIELabPix& target,
    const fCIELabPix* palette,
    int palette_size
)
{
    int p1 = 0; float dist1 = 1e9f;
    int p2 = 0; float dist2 = 1e9f;

    // Linear Search (Palette is small, ~10 colors, so this is fast)
    for (int i = 0; i < palette_size; ++i)
    {
        float dL = target.L - palette[i].L;
        float da = target.a - palette[i].a;
        float db = target.b - palette[i].b;
        
        // Squared Euclidean Distance
        float d2 = dL*dL + da*da + db*db;

        if (d2 < dist1)
        {
            dist2 = dist1; p2 = p1;
            dist1 = d2;    p1 = i;
        } else if (d2 < dist2)
        {
            dist2 = d2;    p2 = i;
        }
    }

    // Calculate Ratio using Inverse Distance Weighting
    float d1_sqrt = FastCompute::Sqrt(dist1);
    float d2_sqrt = FastCompute::Sqrt(dist2);
    float sum = d1_sqrt + d2_sqrt;

    DecomposedColor result;
    result.idx_p1 = p1;
    result.idx_p2 = p2;
    
    if (sum < 0.001f)
    {
        result.ratio = 1.0f; // Exact match
    } else
    {
        // Ratio is probability of Primary. 
        // If dist1 is 0, ratio should be 1.0.
        result.ratio = d2_sqrt / sum;
    }

    return result;
}


// Seurat / Pissarro (The Cluster)
// Technique: Draws multiple small, scattered circles to simulate "dusting."
void RenderKernel_Cluster
(
    const Point2D& pt,
    const fCIELabPix& target_color,
    const RenderContext& ctx,
    const PointillismRenderParams& params,
    float* RESTRICT canvas,
    int width, int height,
    LCG_RNG& rng
)
{
    // 1. Color Logic
    fCIELabPix processed_color = const_cast<fCIELabPix&>(target_color); // Cast for local mod
    processed_color = Apply_Color_Mode (processed_color, (int)ctx.color_mode, (float)params.Vibrancy);
    
    DecomposedColor mix = Decompose(processed_color, ctx.palette_buffer, ctx.palette_size);

    // 2. Geometry Setup
    // Seurat uses many small dots.
    int sub_dots = 5; 
    float radius = (float)params.DotSize * 0.1f; // Scale slider to pixels (heuristic)
    if (radius < 1.0f) radius = 1.0f;
    float opacity = 0.85f; // Slight transparency for layering

    float r_sq = radius * radius;

    // 3. Sub-Dot Loop
    for (int k = 0; k < sub_dots; ++k)
    {
        // Pick Color (Probabilistic)
        int color_idx = (rng.next_float() < mix.ratio) ? mix.idx_p1 : mix.idx_p2;
        fCIELabPix draw_color = ctx.palette_buffer[color_idx];

        // Jitter Position (Cluster effect)
        float scatter_range = radius * 1.5f;
        float cx = pt.x + rng.next_range(-scatter_range, scatter_range);
        float cy = pt.y + rng.next_range(-scatter_range, scatter_range);

        // Bounding Box
        int min_x = std::max(0, (int)(cx - radius));
        int max_x = std::min(width, (int)(cx + radius) + 1);
        int min_y = std::max(0, (int)(cy - radius));
        int max_y = std::min(height, (int)(cy + radius) + 1);

        // Rasterize Circle
        for (int y = min_y; y < max_y; ++y)
        {
            float dy = (float)y - cy;
            int row_offset = y * width;
            for (int x = min_x; x < max_x; ++x)
            {
                float dx = (float)x - cx;
                if ((dx*dx + dy*dy) <= r_sq)
                {
                    Blend_Lab_Pixel(&canvas[(row_offset + x) * 3], draw_color, opacity);
                }
            }
        }
    }
    
    return;
}

// Signac (The Mosaic)
// Technique: Draws rotated squares/rectangles (Tesserae).
void RenderKernel_Mosaic
(
    const Point2D& pt,
    const fCIELabPix& target_color,
    const RenderContext& ctx,
    const PointillismRenderParams& params,
    float* RESTRICT canvas,
    int width, int height,
    LCG_RNG& rng
)
{
    // 1. Color Logic (Signac is usually more solid/opaque)
    fCIELabPix processed_color = const_cast<fCIELabPix&>(target_color);
    processed_color = Apply_Color_Mode (processed_color, (int)ctx.color_mode, (float)params.Vibrancy);
    DecomposedColor mix = Decompose(processed_color, ctx.palette_buffer, ctx.palette_size);

    // 2. Geometry Setup
    // Signac uses fewer, larger, solid blocks.
    int sub_dots = 1; 
    float size = (float)params.DotSize * 0.15f; 
    if (size < 1.5f) size = 1.5f;
    float opacity = 0.95f; // Solid paint

    // Random Rotation (-15 to +15 degrees)
    constexpr float pi2rad = FastCompute::PI / 180.0f;
    float angle_deg = rng.next_range(-15.0f, 15.0f);
    float angle_rad = angle_deg * pi2rad;
    
    float cos_a, sin_a;
    FastCompute::SinCos (angle_rad, cos_a, sin_a);

    for (int k = 0; k < sub_dots; ++k)
    {
        int color_idx = (rng.next_float() < mix.ratio) ? mix.idx_p1 : mix.idx_p2;
        fCIELabPix draw_color = ctx.palette_buffer[color_idx];

        // Scan bounding box based on rotation radius (sqrt(2) * half_size)
        float half_size = size * 0.5f;
        float bb_radius = half_size * 1.414f; 

        int min_x = std::max(0, (int)(pt.x - bb_radius));
        int max_x = std::min(width, (int)(pt.x + bb_radius) + 1);
        int min_y = std::max(0, (int)(pt.y - bb_radius));
        int max_y = std::min(height, (int)(pt.y + bb_radius) + 1);

        // Rasterize Rotated Rectangle
        for (int y = min_y; y < max_y; ++y)
        {
            float dy = (float)y - pt.y;
            int row_offset = y * width;
            
            for (int x = min_x; x < max_x; ++x)
            {
                float dx = (float)x - pt.x;

                // Rotate point into local rectangle space
                float local_x = dx * cos_a - dy * sin_a;
                float local_y = dx * sin_a + dy * cos_a;

                // Check Axis-Aligned Bounds in local space
                if (std::abs(local_x) <= half_size && std::abs(local_y) <= half_size)
                {
                    Blend_Lab_Pixel(&canvas[(row_offset + x) * 3], draw_color, opacity);
                }
            }
        }
    }
    
    return;
}

// Matisse (The Flow)
// Technique: Draws oriented ellipses based on image gradients. Note: This requires the Density_Map to calculate the gradient on the fly.
void RenderKernel_Flow
(
    const Point2D& pt,
    const fCIELabPix& target_color,
    const RenderContext& ctx,
    const PointillismRenderParams& params,
    const float* RESTRICT density_map, // Needed for flow calculation
    float* RESTRICT canvas,
    int width, int height,
    LCG_RNG& rng,
    bool bLongStroke = false
)
{
    // 1. Color Logic
    fCIELabPix processed_color = const_cast<fCIELabPix&>(target_color);
    processed_color = Apply_Color_Mode_Boost (processed_color, (int)ctx.color_mode, (float)params.Vibrancy);
    DecomposedColor mix = Decompose(processed_color, ctx.palette_buffer, ctx.palette_size);

    // 2. Calculate Gradient Flow (Sobel on the fly at dot position)
    int px = std::min(std::max(1, (int)pt.x), width - 2);
    int py = std::min(std::max(1, (int)pt.y), height - 2);
    
    // Simple 3x3 Sobel approx
    float g_x = density_map[py * width + (px + 1)] - density_map[py * width + (px - 1)];
    float g_y = density_map[(py + 1) * width + px] - density_map[(py - 1) * width + px];
    
    // Angle perpendicular to gradient (along the edge)
    float angle = std::atan2(g_y, g_x) + (3.14159f * 0.5f); 
    
    float len_a, len_b;
    
    // 3. Geometry Setup
    if (true == bLongStroke)
    {
        len_a = (float)params.DotSize * 0.3f; // Long axis
        len_b = len_a * 0.2f;                 // Short axis (Aspect ratio ~1:3)
        if (len_a < 3.0f) len_a = 3.0f;
        if (len_b < 0.75) len_b = 0.75f;
    }
    else
    {
        len_a = (float)params.DotSize * 0.2f; // Long axis
        len_b = len_a * 0.3f;                 // Short axis (Aspect ratio ~1:3)
        if (len_a < 2.0f) len_a = 2.0f;
        if (len_b < 1.0f) len_b = 1.0f;
    }
    

    float opacity = 0.9f;
    float cos_a, sin_a;
    FastCompute::SinCos (angle, cos_a, sin_a);

    // 4. Draw Oriented Ellipse
    int color_idx = (rng.next_float() < mix.ratio) ? mix.idx_p1 : mix.idx_p2;
    fCIELabPix draw_color = ctx.palette_buffer[color_idx];

    float bb_radius = std::max(len_a, len_b);
    int min_x = std::max(0, static_cast<int>(pt.x - bb_radius));
    int max_x = std::min(width, static_cast<int>(pt.x + bb_radius) + 1);
    int min_y = std::max(0, static_cast<int>(pt.y - bb_radius));
    int max_y = std::min(height, static_cast<int>(pt.y + bb_radius) + 1);

    float a_sq = len_a * len_a;
    float b_sq = len_b * len_b;

    for (int y = min_y; y < max_y; ++y)
    {
        const float dy = static_cast<float>(y) - pt.y;
        int row_offset = y * width;
        
        for (int x = min_x; x < max_x; ++x)
        {
            const float dx = static_cast<float>(x) - pt.x;

            // Rotate into local space
            float u = dx * cos_a + dy * sin_a;
            float v = -dx * sin_a + dy * cos_a;

            // Ellipse Equation: (u^2 / a^2) + (v^2 / b^2) <= 1
            if (((u*u)/a_sq + (v*v)/b_sq) <= 1.0f)
            {
                Blend_Lab_Pixel(&canvas[(row_offset + x) * 3], draw_color, opacity);
            }
        }
    }
    
    return;
}




/**
 * PHASE 4 ORCHESTRATOR: ARTISTIC RENDERING
 * Adapted for Semi-Planar Source Input (L + ab).
 */
void ArtisticRendering
(
    const Point2D* RESTRICT points, 
    int32_t num_points,
    const JFAPixel* RESTRICT voronoi_map,
    const float* RESTRICT src_L,
    const float* RESTRICT src_ab,
    const float* RESTRICT density_map, 
    int32_t width, 
    int32_t height,
    const PontillismControls& user_params,
    const RenderScratchMemory& scratch,
    float* RESTRICT canvas_lab
)
{
    // 1. Initialize RNG
    LCG_RNG rng(user_params.RandomSeed);

    // 2. Initialize Canvas
    // Pass split buffers
    Init_Canvas(canvas_lab, src_L, src_ab, width, height, user_params.Background);

    // 3. Integrate Colors
    // Pass split buffers
    Integrate_Colors
    (
        voronoi_map, 
        src_L,
        src_ab, 
        width,
        height, 
        num_points, 
        scratch.acc_L, 
        scratch.acc_a, 
        scratch.acc_b, 
        scratch.acc_count, 
        scratch.avg_colors 
    );

    // 4. Retrieve Painter Strategy
    IPainter* painter = GetPainterRegistry(user_params.PainterStyle);
    
    RenderContext ctx{};
    painter->SetupContext(ctx);

    // Prepare Render Params
    PointillismRenderParams render_params;
    render_params.DotSize = user_params.DotSize;
    render_params.Vibrancy = user_params.Vibrancy;
    render_params.RandomSeed = user_params.RandomSeed;
    

    // ---------------------------------------------------------
    // OPTIMIZATION: SPATIAL SORT (INDIRECT)
    // ---------------------------------------------------------
    
    // A. Prepare Index Buffer
    // We reuse 'scratch.acc_count' (int32) because it is dead memory now.
    int32_t* draw_order = scratch.acc_count;
    
    // Fill with 0, 1, 2, ... num_points - 1
    for (int i = 0; i < num_points; ++i) draw_order[i] = i;

    // B. Sort the Indices based on Y-Coordinate of points
    std::sort(draw_order, draw_order + num_points, 
        [&points](const int32_t a, const int32_t b) {
            return points[a].y < points[b].y;
        }
    );

    // ---------------------------------------------------------
    // 3. MAIN RENDERING LOOP (Sorted)
    // ---------------------------------------------------------    
    
    switch (user_params.PainterStyle)
    {
        // --- GROUP A: THE CLUSTER (Circles) ---
        // Seurat, Pissarro: Tiny atomized dots.
        case ArtPointillismPainter::ART_POINTILLISM_PAINTER_SEURAT:
        case ArtPointillismPainter::ART_POINTILLISM_PAINTER_PISSARRO:
        {
            for (int32_t i = 0; i < num_points; ++i)
            {
                int32_t k = draw_order[i]; 

                RenderKernel_Cluster
                (
                    points[k], scratch.avg_colors[k], 
                    ctx, render_params, 
                    canvas_lab, width, height, rng
                );
            }
        }
        break;

        // --- GROUP B: THE MOSAIC (Squares) ---
        // Signac, Cross, Luce: Distinct blocks/tesserae.
        case ArtPointillismPainter::ART_POINTILLISM_PAINTER_SIGNAC:
        case ArtPointillismPainter::ART_POINTILLISM_PAINTER_CROSS:
        case ArtPointillismPainter::ART_POINTILLISM_PAINTER_RYSSELBERGHE:
        case ArtPointillismPainter::ART_POINTILLISM_PAINTER_LUCE:
        {
            for (int32_t i = 0; i < num_points; ++i)
            {
                int32_t k = draw_order[i]; 

                RenderKernel_Mosaic
                (
                    points[k], scratch.avg_colors[k], 
                    ctx, render_params, 
                    canvas_lab, width, height, rng
                );
            }
        }
        break;

        // --- GROUP C: THE FLOW (Oriented Ellipses) ---
        // Van Gogh, Matisse: Directional strokes following the Density Gradient.
        case ArtPointillismPainter::ART_POINTILLISM_PAINTER_VAN_GOGH:
        case ArtPointillismPainter::ART_POINTILLISM_PAINTER_MATISSE:
        {
            for (int32_t i = 0; i < num_points; ++i)
            {
                int32_t k = draw_order[i]; 

                RenderKernel_Flow
                (
                    points[k], scratch.avg_colors[k], 
                    ctx, render_params, 
                    density_map, // <--- Passing the Map for Orientation Calculation
                    canvas_lab, width, height, rng,
                    user_params.PainterStyle == ArtPointillismPainter::ART_POINTILLISM_PAINTER_VAN_GOGH
                );
            }
        }
        break;
    }
        
    return;
}
