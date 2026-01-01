#include "AlgoJFA.hpp"

/**
 * Sub-Stage 3.1: Initialize JFA Buffer
 * 
 * Clears the grid to "Infinity" and writes the initial seed positions.
 * 
 * @param points_in     [Input]  Array of current dot positions.
 * @param num_points    [Input]  Number of dots in the array.
 * @param width         [Input]  Image width.
 * @param height        [Input]  Image height.
 * @param grid_out      [Output] The JFA Buffer (W * H) to initialize.
 */
void JFA_Init_Buffer
(
    const Point2D* RESTRICT points_in,
    int32_t num_points,
    int32_t width,
    int32_t height,
    JFAPixel* RESTRICT grid_out
) noexcept
{
    // 1. Clear Grid to "Empty/Infinity"
    // We iterate every pixel linearly.
    const int32_t total_pixels = width * height;
    for (int32_t i = 0; i < total_pixels; ++i)
    {
        grid_out[i].seed_index = -1;       // No seed assigned
        grid_out[i].seed_x     = FLT_MAX;  // Infinite distance
        grid_out[i].seed_y     = FLT_MAX;
    }

    // 2. Splat Seeds
    // We iterate through the dots and place them on the grid.
    for (int32_t i = 0; i < num_points; ++i)
    {
        const float px = points_in[i].x;
        const float py = points_in[i].y;

        // Convert float coordinate to integer pixel index
        const int32_t ix = static_cast<int32_t>(px + 0.5f); // Round to nearest
        const int32_t iy = static_cast<int32_t>(py + 0.5f);

        // Bounds Check (Crucial for safety)
        if (ix >= 0 && ix < width && iy >= 0 && iy < height)
        {
            const int32_t pixel_idx = iy * width + ix;
            
            // Write the seed info into this pixel
            grid_out[pixel_idx].seed_index = i;
            grid_out[pixel_idx].seed_x     = px;
            grid_out[pixel_idx].seed_y     = py;
        }
    }
    
    return;
}

/**
 * Sub-Stage 3.2: JFA Step (Kernel)
 * 
 * Propagates seed information across the grid using a specific step size.
 * 
 * @param grid_src      [Input]  Source JFA Buffer (Read-Only).
 * @param grid_dst      [Output] Destination JFA Buffer (Write-Only).
 * @param width         [Input]  Image Width.
 * @param height        [Input]  Image Height.
 * @param step_len      [Input]  The jump distance for this pass (e.g., 512, 256, 128...).
 */
void JFA_Step
(
    const JFAPixel* RESTRICT grid_src,
    JFAPixel* RESTRICT grid_dst,
    int32_t width,
    int32_t height,
    int32_t step_len
)
{
    // Iterate over every pixel in the image (Destination)
    for (int32_t y = 0; y < height; ++y)
	{
        for (int32_t x = 0; x < width; ++x)
		{
            // 1. Read the current best seed for this pixel from Source
            // (We assume the pixel at (x,y) corresponds to itself in the previous pass)
            const int32_t center_idx = y * width + x;
            JFAPixel best_pixel = grid_src[center_idx];

            // Calculate current best squared distance
            // If ID is -1 (empty), distance is Infinite
            float best_dist_sq = FLT_MAX;
            if (best_pixel.seed_index != -1)
			{
                float dx = static_cast<float>(x) - best_pixel.seed_x;
                float dy = static_cast<float>(y) - best_pixel.seed_y;
                best_dist_sq = dx*dx + dy*dy;
            }

            // 2. Check 8 Neighbors (and center, effectively 9 samples)
            // We loop from -1 to +1 in both X and Y directions
            for (int32_t dy = -1; dy <= 1; ++dy)
			{
                for (int32_t dx = -1; dx <= 1; ++dx)
				{
                    
                    // Optimization: Skip center (0,0) as we already loaded it above
                    if (dx == 0 && dy == 0) continue;

                    // Calculate Neighbor Coordinate
                    const int32_t ny = y + (dy * step_len);
                    const int32_t nx = x + (dx * step_len);

                    // Bounds Check: Is the neighbor inside the image?
                    if (nx >= 0 && nx < width && ny >= 0 && ny < height)
					{
                        
                        // 3. Read Neighbor's Seed Info
                        const int32_t neighbor_idx = ny * width + nx;
                        const JFAPixel* neighbor_pixel = &grid_src[neighbor_idx];

                        // Does the neighbor have a valid seed?
                        if (neighbor_pixel->seed_index != -1)
						{
                            
                            // 4. Calculate Distance:
                            // From ME (Current Pixel x,y) to THE NEIGHBOR'S SEED (seed_x, seed_y)
                            const float dist_x = static_cast<float>(x) - neighbor_pixel->seed_x;
                            const float dist_y = static_cast<float>(y) - neighbor_pixel->seed_y;
                            const float dist_sq = dist_x*dist_x + dist_y*dist_y;

                            // 5. Compare and Swap
                            if (dist_sq < best_dist_sq)
							{
                                best_dist_sq = dist_sq;
                                // Adopt the neighbor's seed info
                                best_pixel.seed_index = neighbor_pixel->seed_index;
                                best_pixel.seed_x     = neighbor_pixel->seed_x;
                                best_pixel.seed_y     = neighbor_pixel->seed_y;
                            }
                        }
                    }
                }
            }

            // 6. Write the winner to the Destination Buffer
            grid_dst[center_idx] = best_pixel;
        }
    }
}


/**
 * Sub-Stage 3.3: Update Points (Centroid)
 * 
 * Accumulates weighted positions from the grid and moves points to centroids.
 * 
 * @param grid_in       [Input]  Final JFA Buffer (Voronoi Map).
 * @param density_map   [Input]  Density Map (0.0-1.0).
 * @param width, height [Input]  Dimensions.
 * @param points_in_out [In/Out] The Point List to update.
 * @param num_points    [Input]  Number of points.
 * @param accum_x       [Scratch] Buffer of size num_points (Zeroed).
 * @param accum_y       [Scratch] Buffer of size num_points (Zeroed).
 * @param accum_w       [Scratch] Buffer of size num_points (Zeroed).
 */
void JFA_Update_Points
(
    const JFAPixel* RESTRICT grid_in,
    const float* RESTRICT density_map,
    int32_t width,
    int32_t height,
    Point2D* RESTRICT points_in_out,
    int32_t num_points,
    float* RESTRICT accum_x,
    float* RESTRICT accum_y,
    float* RESTRICT accum_w
)
{
    // 1. Clear Accumulators (Reset to 0)
    // We iterate linearly through the point array size.
    for (int32_t i = 0; i < num_points; ++i)
	{
        accum_x[i] = 0.0f;
        accum_y[i] = 0.0f;
        accum_w[i] = 0.0f;
    }

    // 2. Accumulate (Integration Pass)
    // Iterate over every pixel in the image.
    const int32_t total_pixels = width * height;
    for (int32_t i = 0; i < total_pixels; ++i)
	{
        
        // Which dot owns this pixel?
        const int32_t seed_id = grid_in[i].seed_index;

        // Only process if a valid seed owns this pixel (Safety check)
        if (seed_id >= 0 && seed_id < num_points)
		{
            
            // Recover (x, y) from index 'i'
            const int32_t y = i / width;
            const int32_t x = i % width;
            
            // Read Density (Weight)
            const float weight = density_map[i];

            // Add to the owner's accumulator
            accum_x[seed_id] += static_cast<float>(x) * weight;
            accum_y[seed_id] += static_cast<float>(y) * weight;
            accum_w[seed_id] += weight;
        }
    }

    // 3. Move Points (Normalization Pass)
    // Iterate over dots and divide accumulated position by total weight.
    for (int32_t i = 0; i < num_points; ++i)
	{
        const float total_weight = accum_w[i];

        // If the dot owns pixels with density (normal case)
        if (total_weight > 0.0001f)
		{
            const float new_x = accum_x[i] / total_weight;
            const float new_y = accum_y[i] / total_weight;

            // Update the point position
            points_in_out[i].x = new_x;
            points_in_out[i].y = new_y;
        } 
        // Edge Case: If a dot was squeezed out or owns only empty space,
        // we leave it where it is (or we could cull it, but keeping it is safer).
    }
	
	return;
}


void Dot_Refinement
(
    Point2D* RESTRICT points, 
    int32_t num_points,
    const float* RESTRICT density_map,
    int32_t width,
	int32_t height,
    int32_t relaxation_iterations, // e.g., 4 to 8
    JFAPixel* RESTRICT jfa_buffer_A,    // Pre-allocated W*H
    JFAPixel* RESTRICT jfa_buffer_B,    // Pre-allocated W*H
    float* RESTRICT acc_x,              // Pre-allocated Scratch
    float* RESTRICT acc_y,              // Pre-allocated Scratch
    float* RESTRICT acc_w               // Pre-allocated Scratch
)
{
    // 1. Calculate Initial JFA Step Size (Power of 2)
    const int32_t max_dim = (width > height) ? width : height;
    int32_t start_step = 1;
    
	while (start_step < max_dim)
	{
        start_step <<= 1;
    }
	
    start_step >>= 1; // Start at half the max dimension (e.g., 1024 for 1080p)

    // --- OUTER LOOP: LLOYD'S RELAXATION (e.g., 8 times) ---
    for (int32_t iter = 0; iter < relaxation_iterations; ++iter)
	{
        // A. INITIALIZE JFA GRID
        // Clear grid and splat current point positions onto Buffer A
        JFA_Init_Buffer (points, num_points, width, height, jfa_buffer_A);

        // Setup Ping-Pong Pointers
        JFAPixel* RESTRICT src = jfa_buffer_A;
        JFAPixel* RESTRICT dst = jfa_buffer_B;

        // B. INNER LOOP: JUMP FLOODING PASSES
        // Reduces step size: 1024 -> 512 -> ... -> 1
        int32_t current_step = start_step;
        
        while (current_step >= 1)
		{
            // Execute One Pass
            JFA_Step (src, dst, width, height, current_step);

            // Ping-Pong: Swap pointers
            // The destination of this pass becomes the source of the next
            std::swap(src, dst);

            // Halve the jump distance
            current_step >>= 1; 
        }

        // After the JFA loop finishes, 'src' holds the final valid Voronoi Map.
        // (Because we swapped at the end of the last loop iteration).
        const JFAPixel* final_voronoi_grid = src;

        // C. COMPUTE CENTROIDS & UPDATE POINTS
        // Calculate new positions based on the Voronoi map we just built
        JFA_Update_Points (final_voronoi_grid, density_map, width, height, points, num_points, acc_x, acc_y, acc_w);
    }
	
	return;
}