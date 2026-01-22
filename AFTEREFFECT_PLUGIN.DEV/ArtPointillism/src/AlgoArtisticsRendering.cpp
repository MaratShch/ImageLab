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
    const float inv_alpha = 1.0f - alpha;
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
//    float chroma = std::sqrt(input.a * input.a + input.b * input.b);
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
 * AVX2 Optimized Color Decompose.
 * Uses Unaligned Loads (_mm256_loadu_ps) to handle standard class memory layouts.
 */
DecomposedColor Decompose
(
    const fCIELabPix& target, // Passed by const ref for speed
    const RenderContext& ctx
)
{
    // 1. Broadcast Target to Registers
    __m256 tL = _mm256_set1_ps(target.L);
    __m256 ta = _mm256_set1_ps(target.a);
    __m256 tb = _mm256_set1_ps(target.b);

    // Array to store calculated squared distances
    // This local array IS aligned because it's on the stack and we can control it.
    CACHE_ALIGN float dists[32]; 

    int count = ctx.palette_size;
    
    // 2. Loop in steps of 8
    // We assume internal buffers are padded to 32 floats, so reading past 'count' is safe
    // up to the 32 boundary.
    
    // Force loop to go up to 32 to avoid scalar tail handling? 
    // Since we padded with Infinity, handling 32 is always safe and branchless.
    for (int i = 0; i < 32; i += 8)
    {
        
        // --- CHANGE: LOAD UNALIGNED ---
        // Pointers from RenderContext might not be on 32-byte boundaries.
        __m256 pL = _mm256_loadu_ps(ctx.pal_L + i);
        __m256 pa = _mm256_loadu_ps(ctx.pal_a + i);
        __m256 pb = _mm256_loadu_ps(ctx.pal_b + i);

        // Calc Diff
        __m256 dL = _mm256_sub_ps(tL, pL);
        __m256 da = _mm256_sub_ps(ta, pa);
        __m256 db = _mm256_sub_ps(tb, pb);

        // Calc DistSq: (dL*dL + da*da + db*db)
        __m256 d2 = _mm256_mul_ps(dL, dL);
        d2 = _mm256_fmadd_ps(da, da, d2);
        d2 = _mm256_fmadd_ps(db, db, d2);

        // Store to stack (Aligned store is fine here)
        _mm256_store_ps(dists + i, d2);
    }

    // 3. Find Min1 and Min2 (Scalar scan of the 32 floats)
    // Fast scalar iteration over L1-cached stack memory
    int p1 = 0; float min1 = 1e9f;
    int p2 = 0; float min2 = 1e9f;

    // Iterate up to actual size (ignore padding)
    for (int k = 0; k < count; ++k)
    {
        float d = dists[k];
        if (d < min1) {
            min2 = min1; p2 = p1;
            min1 = d;    p1 = k;
        } else if (d < min2) {
            min2 = d;    p2 = k;
        }
    }

    // 4. Ratio Calculation
    float d1_sqrt = FastCompute::Sqrt(min1);
    float d2_sqrt = FastCompute::Sqrt(min2);
    float sum = d1_sqrt + d2_sqrt;

    DecomposedColor result;
    result.idx_p1 = p1;
    result.idx_p2 = p2;
    // Avoid div by zero
    result.ratio = (sum < 0.001f) ? 1.0f : (d2_sqrt / sum);
    
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
    int32_t width,
    int32_t height,
    LCG_RNG& rng
)
{
    // 1. Color Logic
    fCIELabPix processed_color = const_cast<fCIELabPix&>(target_color);
    processed_color = Apply_Color_Mode_Boost(processed_color, (int)ctx.color_mode, (float)params.Vibrancy);
    
    // FIX: Call the AVX2 function, passing the Context (which holds planar pointers)
    DecomposedColor mix = Decompose(processed_color, ctx);

    // 2. Geometry Setup
    int sub_dots = 5; 
    float radius = (float)params.DotSize * 0.1f; 
    if (radius < 1.0f) radius = 1.0f;
    float opacity = 0.85f;
    float r_sq = radius * radius;

    // 3. Sub-Dot Loop
    for (int k = 0; k < sub_dots; ++k) {
        int color_idx = (rng.next_float() < mix.ratio) ? mix.idx_p1 : mix.idx_p2;
        
        // FIX: Reconstruct color from Planar Arrays [L, a, b]
        fCIELabPix draw_color;
        draw_color.L = ctx.pal_L[color_idx];
        draw_color.a = ctx.pal_a[color_idx];
        draw_color.b = ctx.pal_b[color_idx];

        // Jitter & Draw
        float scatter_range = radius * 1.5f;
        float cx = pt.x + rng.next_range(-scatter_range, scatter_range);
        float cy = pt.y + rng.next_range(-scatter_range, scatter_range);

        // ... Bounding Box & Pixel Loop (Same as before) ...
        int min_x = std::max(0, (int)(cx - radius));
        int max_x = std::min(width, (int)(cx + radius) + 1);
        int min_y = std::max(0, (int)(cy - radius));
        int max_y = std::min(height, (int)(cy + radius) + 1);

        for (int y = min_y; y < max_y; ++y) {
            float dy = (float)y - cy;
            int row_offset = y * width;
            for (int x = min_x; x < max_x; ++x) {
                float dx = (float)x - cx;
                if ((dx*dx + dy*dy) <= r_sq) {
                    Blend_Lab_Pixel(&canvas[(row_offset + x) * 3], draw_color, opacity);
                }
            }
        }
    }
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
    int32_t width,
    int32_t height,
    LCG_RNG& rng
)
{
    fCIELabPix processed_color = const_cast<fCIELabPix&>(target_color);
    processed_color = Apply_Color_Mode_Boost(processed_color, (int)ctx.color_mode, (float)params.Vibrancy);
    
    // FIX: AVX2 Decompose
    DecomposedColor mix = Decompose(processed_color, ctx);

    int sub_dots = 1; 
    float size = (float)params.DotSize * 0.15f; 
    if (size < 1.5f) size = 1.5f;
    float opacity = 0.95f; 

    float angle_deg = rng.next_range(-15.0f, 15.0f);
    float angle_rad = angle_deg * 3.14159f / 180.0f;
    float cos_a, sin_a;

    FastCompute::SinCos(angle_rad, sin_a, cos_a);

    for (int k = 0; k < sub_dots; ++k) {
        int color_idx = (rng.next_float() < mix.ratio) ? mix.idx_p1 : mix.idx_p2;
        
        // FIX: Planar Lookup
        fCIELabPix draw_color;
        draw_color.L = ctx.pal_L[color_idx];
        draw_color.a = ctx.pal_a[color_idx];
        draw_color.b = ctx.pal_b[color_idx];

        // ... Drawing Logic (Same as before) ...
        float half_size = size * 0.5f;
        float bb_radius = half_size * 1.414f; 
        int min_x = std::max(0, (int)(pt.x - bb_radius));
        int max_x = std::min(width, (int)(pt.x + bb_radius) + 1);
        int min_y = std::max(0, (int)(pt.y - bb_radius));
        int max_y = std::min(height, (int)(pt.y + bb_radius) + 1);

        for (int y = min_y; y < max_y; ++y) {
            float dy = (float)y - pt.y;
            int row_offset = y * width;
            for (int x = min_x; x < max_x; ++x) {
                float dx = (float)x - pt.x;
                float local_x = dx * cos_a - dy * sin_a;
                float local_y = dx * sin_a + dy * cos_a;
                if (std::abs(local_x) <= half_size && std::abs(local_y) <= half_size) {
                    Blend_Lab_Pixel(&canvas[(row_offset + x) * 3], draw_color, opacity);
                }
            }
        }
    }
}

// Matisse (The Flow)
// Technique: Draws oriented ellipses based on image gradients. Note: This requires the Density_Map to calculate the gradient on the fly.
void RenderKernel_Flow
(
    const Point2D& pt,
    const fCIELabPix& target_color,
    const RenderContext& ctx,
    const PointillismRenderParams& params,
    const float* RESTRICT density_map, 
    float* RESTRICT canvas,
    int width, int height,
    LCG_RNG& rng,
    bool bLongStroke = false
)
{
    fCIELabPix processed_color = const_cast<fCIELabPix&>(target_color);
    processed_color = Apply_Color_Mode_Boost(processed_color, (int)ctx.color_mode, (float)params.Vibrancy);
    
    // FIX: AVX2 Decompose
    DecomposedColor mix = Decompose(processed_color, ctx);

    // ... Gradient Calculation (Same) ...
    int px = std::min(std::max(1, (int)pt.x), width - 2);
    int py = std::min(std::max(1, (int)pt.y), height - 2);
    float g_x = density_map[py * width + (px + 1)] - density_map[py * width + (px - 1)];
    float g_y = density_map[(py + 1) * width + px] - density_map[(py - 1) * width + px];
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
    
    const float opacity = (processed_color.L < 20.0f) ? 0.98f : 0.90f;
    float cos_a, sin_a;

    FastCompute::SinCos(angle, sin_a, cos_a);

    int color_idx = (rng.next_float() < mix.ratio) ? mix.idx_p1 : mix.idx_p2;
    
    // FIX: Planar Lookup
    fCIELabPix draw_color;
    draw_color.L = ctx.pal_L[color_idx];
    draw_color.a = ctx.pal_a[color_idx];
    draw_color.b = ctx.pal_b[color_idx];

    // ... Drawing Logic (Same) ...
    float bb_radius = std::max(len_a, len_b);
    int min_x = std::max(0, (int)(pt.x - bb_radius));
    int max_x = std::min(width, (int)(pt.x + bb_radius) + 1);
    int min_y = std::max(0, (int)(pt.y - bb_radius));
    int max_y = std::min(height, (int)(pt.y + bb_radius) + 1);

    float a_sq = len_a * len_a;
    float b_sq = len_b * len_b;

    for (int y = min_y; y < max_y; ++y)
    {
        float dy = (float)y - pt.y;
        int row_offset = y * width;
        for (int x = min_x; x < max_x; ++x) {
            float dx = (float)x - pt.x;
            float u = dx * cos_a + dy * sin_a;
            float v = -dx * sin_a + dy * cos_a;
            if (((u*u)/a_sq + (v*v)/b_sq) <= 1.0f) {
                Blend_Lab_Pixel(&canvas[(row_offset + x) * 3], draw_color, opacity);
            }
        }
    }
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

    // 2. Integrate Colors
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

    // 3. Initialize Canvas
    // Pass split buffers
    Init_Canvas(canvas_lab, src_L, src_ab, width, height, user_params.Background);

    // 4. Retrieve Painter Strategy
    IPainter* painter = GetPainterRegistry(user_params.PainterStyle);
    
    RenderContext ctx{};
    painter->SetupContext(ctx);

    // Prepare Render Params
    PointillismRenderParams render_params;
    render_params.DotSize = user_params.DotSize;
    render_params.Vibrancy = user_params.Vibrancy;
    render_params.BackgroundMode = UnderlyingType(user_params.Background);
    render_params.RandomSeed = user_params.RandomSeed;
    render_params.Opacity = user_params.Opacity;


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

RenderScratchMemory AllocScratchMemory
(
    const int32_t width, 
    const int32_t height
)
{
    constexpr float max_density_factor = (250.0f / 10000.0f) * 7.0f;
    const int32_t max_dots = static_cast<int32_t>((width * height) * max_density_factor * 1.15f + 0.5f);

    return AllocScratchMemory(max_dots);
 }


RenderScratchMemory AllocScratchMemory
(
    const int32_t max_dots 
)
{
    RenderScratchMemory scratch{};

    const size_t size_float = max_dots * sizeof(float);
    const size_t size_int   = max_dots * sizeof(int32_t);
    const size_t size_color = max_dots * sizeof(fCIELabPix);

    const size_t total_bytes_needed = (size_float * 3) + size_int + size_color;

    std::cout << "Scratch memory required size: " << total_bytes_needed << " bytes for " << max_dots << " MAX dots." << std::endl;
    
    // --- 2. Allocate Raw Memory (ONCE, reuse per frame) ---
    // Use your preferred allocator (new, malloc, or custom aligned_alloc)
    uint8_t* raw_buffer = new uint8_t[total_bytes_needed];

    if (nullptr != raw_buffer)
    {
        // --- 3. Map the Struct (The "Binding" step) ---
        // Pointer Arithmetic to slice the big block
        scratch.acc_L     = (float*)(raw_buffer);
        scratch.acc_a     = (float*)(raw_buffer + size_float);
        scratch.acc_b     = (float*)(raw_buffer + size_float * 2);
        scratch.acc_count = (int32_t*)(raw_buffer + size_float * 3);
        scratch.avg_colors= (fCIELabPix*)(raw_buffer + size_float * 3 + size_int);
    }
    
    std::cout << "Scratch memory ptr = " << reinterpret_cast<uint64_t>(raw_buffer) << std::endl;

    return scratch;
}


void FreeScratchMemory (RenderScratchMemory& scratchHandler)
{
    if (nullptr != scratchHandler.acc_L)
    {
        uint8_t* ptr = reinterpret_cast<uint8_t*>(scratchHandler.acc_L);
        delete [] ptr;
        memset(&scratchHandler, 0, sizeof(scratchHandler));
    }
    
    return;
}
