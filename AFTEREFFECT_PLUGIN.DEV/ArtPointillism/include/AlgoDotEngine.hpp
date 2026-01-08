#ifndef __IMAGE_LAB_ART_POINTILISM_DOT_ENGINE__
#define __IMAGE_LAB_ART_POINTILISM_DOT_ENGINE__

#include <algorithm> // For min/max
#include <cstdint>
#include <cmath>  // For sqrt, fabs
#include <iostream>
#include <fstream>
#include "Common.hpp"
#include "AlgoDot2D.hpp"

// --- CONSTANTS ---
// 250 dots per 100x100 block is our new High Detail baseline.
constexpr float BASE_DOTS_PER_BLOCK = 250.0f; 

// What is the MAX multiplier the slider can ever produce?
// If slider goes to 125, logic might be: 1.0 + ((125-50)/50)*2.0 = 4.0x
constexpr float MAX_SLIDER_MULTIPLIER = 4.0f;

// Calculate factor: (250 dots / 10000 pixels) * MaxMultiplier(4.0)
// Factor = 0.025 * 4.0 = 0.10 dots per pixel.
constexpr float max_dots_per_pixel = (BASE_DOTS_PER_BLOCK / 10000.0f) * MAX_SLIDER_MULTIPLIER;


// --- INTERNAL WORKING STRUCTURE ---
// Represents a node in the Quadtree. 
// passed in via an external "scratch" buffer (node_pool).
struct FlatQuadNode
{
    float x, y, w, h;       // Bounding box
    float total_mass;       // Sum of density within this node
    int32_t leaf_type;      // 0 = Branch (Internal), 1 = Leaf
};


inline void FlatQuadNode_dbg_print (const FlatQuadNode* RESTRICT pNodeStr, const int32_t size)
{
    if (nullptr != pNodeStr && size > 0)
    {
        std::ofstream outputFile("FlatQuadNode.txt");
        if (outputFile.is_open())
        {
            outputFile << "FlatQuadNode buffer content:" << std::endl;
            
            for (int32_t i = 0; i < size; i++)
            {
                const float x = pNodeStr[i].x;
                const float y = pNodeStr[i].y;
                const float w = pNodeStr[i].w;
                const float h = pNodeStr[i].h;
                
                if (0.f == x && 0.f == y && 0.f == w && 0.f == h)
                    continue;
                
                outputFile << "=== [" << i << "] ===" << std::endl;
                outputFile << "x = " << x << "; y = " << y << "; w = " << w << "; h = " << h << std::endl;
                outputFile << "total_mass = " << pNodeStr[i].total_mass << "; leaf_type = " << pNodeStr[i].leaf_type << std::endl;
                outputFile << "===============" << std::endl;
            }
            
            outputFile.flush();
            outputFile.close();
        }
    }
    
    return;
}

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
);


inline int32_t CalculateTargetDotCount (int32_t sizeX, int32_t sizeY, int32_t user_density) noexcept
{
    // 1. Resolution Baseline (e.g., 100 dots per 100x100 block)
    float area_factor = static_cast<float>(sizeX * sizeY) / 10000.0f;
    float base_dots = 100.0f; 

    // 2. Map Slider (0-100) to a Multiplier (0.1x to 3.0x)
    // Non-linear mapping feels better for users
    float multiplier = 1.0f;
    if (user_density < 50)
	{
        // 0..50 -> 0.1 .. 1.0
        multiplier = 0.1f + (user_density / 50.0f) * 0.9f;
    } else
	{
        // 50..100 -> 1.0 .. 3.0
        multiplier = 1.0f + ((user_density - 50.0f) / 50.0f) * 2.0f;
    }

    return static_cast<int32_t>(area_factor * base_dots * multiplier + 0.5f);
}


inline int32_t CalculateTargetDotHighCount (int32_t width, int32_t height, int32_t density_slider) noexcept
{
    float area_factor = static_cast<float>(width * height) / 10000.0f;
    
    // INCREASED BASELINE:
    // Was 100.0f. Now 250.0f ensures we capture fine details like eyes/leaves 
    // even at the default slider position (50).
    float base_dots_per_block = 250.0f; 

    // Mapping Logic (Kept the same)
    float multiplier = 1.0f;
    if (density_slider < 50)
    {
        // Slider 0..50 -> 0.2x to 1.0x
        multiplier = 0.2f + (static_cast<float>(density_slider) / 50.0f) * 0.8f;
    }
    else
    {
        // Slider 50..100 -> 1.0x to 3.0x
        // At max slider, we get 250 * 3 = 750 dots per block. Ultra High Detail.
        multiplier = 1.0f + ((static_cast<float>(density_slider) - 50.0f) / 50.0f) * 3.0f;
    }

    return static_cast<int>(area_factor * base_dots_per_block * multiplier + 0.5f);
}



inline int32_t CalculateSubDotRadius(int width, int32_t height, int32_t dot_count, int32_t size_slider) noexcept
{
    // 1. Calculate Average Spacing (The grid size)
    const float avg_pixels_per_dot = static_cast<float>(width * height) / static_cast<float>(dot_count);
    const float spacing = std::sqrt(avg_pixels_per_dot);

    // 2. Map Slider to Overlap Factor
    // Old: 0.65 default.
    // New: Allow going down to 0.3 (very sparse/crisp) up to 1.0 (touching).
    // Slider 0   -> 0.3 (Tiny dots, lots of background)
    // Slider 50  -> 0.6 (Standard Seurat)
    // Slider 100 -> 1.2 (Thick Impasto, heavy overlap)
    const float overlap_factor = 0.3f + (static_cast<float>(size_slider) / 100.0f) * 0.9f;

    // 3. Calculate Radius
    float radius = spacing * overlap_factor;

    // 4. Clamp
    // Even at high detail, never go below 1.0 pixels or we see nothing.
    if (radius < 1.0f) radius = 1.0f;
    
    return static_cast<int>(radius);
}


#endif // __IMAGE_LAB_ART_POINTILISM_DOT_ENGINE__
