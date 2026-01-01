#include "AlgoDotEngine.hpp"

// --- DETERMINISTIC RNG HELPER (LCG) ---
// Simple, fast, thread-safe.
struct LCG_RNG
{
    uint32_t state;
    
    // Initialize with User's Seed
    LCG_RNG(int seed) { state = (uint32_t)seed; }
    
    // Returns float [0.0, 1.0]
    float next_float()
	{
        state = state * 1664525u + 1013904223u; // numerical recipes constant
        return (float)state / (float)0xFFFFFFFFu;
    }
};


/**
 * PHASE 2: CONTENT-AWARE SEEDING
 * 
 * Generates an initial distribution of points based on the Density Map.
 * Uses a non-recursive Quadtree to cluster points in high-density areas.
 * 
 * @param density_map       [Input] Float buffer (H * W). Values 0.0 to 1.0.
 * @param width             [Input] Image width.
 * @param height            [Input] Image height.
 * @param params            [Input] User parameters (Density, Seed).
 * @param node_pool         [Scratch] Pre-allocated buffer for QuadNodes.
 *                          Recommended Size: ~ (W*H) / 64.
 * @param max_nodes         [Input] Size of node_pool.
 * @param point_out         [Output] Pre-allocated buffer for Point2D.
 *                          Recommended Size: Max expected dots + padding.
 * @param max_points        [Input] Size of point_out buffer.
 * @param out_generated     [Output] Actual number of points written.
 */
void Run_Seeding
(
    const float* RESTRICT density_map, 
    int32_t width, 
	int32_t height,
    int32_t DotDensity,
    int32_t RandomSeed,
    FlatQuadNode* RESTRICT node_pool, 
	int32_t max_nodes,
    Point2D* RESTRICT point_out, 
	int32_t max_points,
    int32_t* RESTRICT out_generated
) 
{
    // 1. Initialize RNG
    LCG_RNG rng(RandomSeed);

    // 2. Calculate Target Budget
    int32_t target_dot_count = CalculateTargetDotCount(width, height, DotDensity);
    
    // Safety check for empty buffers
    if (max_nodes < 1 || max_points < 1) { *out_generated = 0; return; }

    // --- STEP A: BUILD QUADTREE (Iterative) ---
    
    // Stack for traversal (Depth 64 is enough for any reasonable image resolution)
    CACHE_ALIGN int32_t stack[64];
    int32_t stack_top = 0;
    
    // Initialize Pool Allocator
    int32_t pool_count = 0;

    // Push Root Node
    FlatQuadNode* root = &node_pool[pool_count++];
    root->x = 0.0f; root->y = 0.0f; 
    root->w = (float)width; root->h = (float)height;
    root->leaf_type = 1; // Assume leaf initially
    
    stack[stack_top++] = 0; // Push root index

    float global_total_mass = 0.0f;

    while (stack_top > 0)
	{
        // Pop current node index
        int32_t curr_idx = stack[--stack_top];
        FlatQuadNode* curr = &node_pool[curr_idx];

        // A.1 Calculate Mass (Sum density within bounds)
        // Note: For extreme optimization, use an Integral Image (Summed Area Table).
        // Here we use standard loops for code clarity and lower memory usage.
        float mass = 0.0f;
        const int32_t y_start = static_cast<int32_t>(curr->y);
        const int32_t x_start = static_cast<int32_t>(curr->x);
        const int32_t y_end   = std::min(static_cast<int32_t>(curr->y + curr->h), height);
        const int32_t x_end   = std::min(static_cast<int32_t>(curr->x + curr->w), width);

        for (int32_t y = y_start; y < y_end; ++y)
		{
            const float* row = density_map + (y * width);
            for (int32_t x = x_start; x < x_end; ++x)
			{
                mass += row[x];
            }
        }
        curr->total_mass = mass;

        // A.2 Split Logic
        // Thresholds:
        // - Mass > 5.0: The node contains enough "density" to warrant detailed attention.
        // - Size > 4.0: Don't split pixels too small (diminishing returns).
        // - Pool check: Ensure we don't overrun the scratch buffer.
        const bool should_split = (mass > 5.0f) && 
                            (curr->w > 4.0f) && (curr->h > 4.0f) && 
                            ((pool_count + 4) < max_nodes);

        if (should_split)
		{
            curr->leaf_type = 0; // Convert to Branch

            float hw = curr->w * 0.5f;
            float hh = curr->h * 0.5f;

            // Push 4 children to pool and stack
            // We use a loop or unroll manually. Manual is clearer for 4.
            
            // Child 1 (Top Left)
            const int32_t c1 = pool_count++;
            node_pool[c1] = {curr->x, curr->y, hw, hh, 0.0f, 1};
            stack[stack_top++] = c1;

            // Child 2 (Top Right)
            const int32_t c2 = pool_count++;
            node_pool[c2] = {curr->x + hw, curr->y, hw, hh, 0.0f, 1};
            stack[stack_top++] = c2;

            // Child 3 (Bottom Left)
            const int32_t c3 = pool_count++;
            node_pool[c3] = {curr->x, curr->y + hh, hw, hh, 0.0f, 1};
            stack[stack_top++] = c3;

            // Child 4 (Bottom Right)
            const int32_t c4 = pool_count++;
            node_pool[c4] = {curr->x + hw, curr->y + hh, hw, hh, 0.0f, 1};
            stack[stack_top++] = c4;

        }
		else
		{
            // It remains a LEAF. Add to global sum.
            global_total_mass += mass;
        }
    }

    // --- STEP B: GENERATE POINTS (Linear) ---
    
    int32_t points_generated = 0;
    
    // Prevent division by zero if image is purely black
    if (global_total_mass < 0.0001f) global_total_mass = 1.0f;

    // Iterate linearly through the node pool. 
    // This is cache-friendly compared to traversing pointers.
    for (int32_t i = 0; i < pool_count; ++i)
	{
        FlatQuadNode* node = &node_pool[i];

        // Only process LEAF nodes
        if (1 == node->leaf_type)
		{
            
            // Calculate dots for this node based on its mass contribution
            const float exact_count = static_cast<float>(target_dot_count) * (node->total_mass / global_total_mass);
            int32_t count = static_cast<int32_t>(exact_count);
            
            // Stochastic Rounding:
            // If exact_count is 0.4, we have a 40% chance of spawning 1 dot.
            // This prevents banding in gradients.
            const float remainder = exact_count - static_cast<float>(count);
            if (rng.next_float() < remainder)
			{
                count++;
            }

            // Scatter dots within node bounds
            for (int32_t k = 0; k < count; ++k)
			{
                if (points_generated >= max_points) break;

                // Random position relative to node
                const float px = node->x + (rng.next_float() * node->w);
                const float py = node->y + (rng.next_float() * node->h);

                point_out[points_generated].x = px;
                point_out[points_generated].y = py;
                points_generated++;
            }
        }
        
        if (points_generated >= max_points)
            break;
    }

    *out_generated = points_generated;
	
	return;
}