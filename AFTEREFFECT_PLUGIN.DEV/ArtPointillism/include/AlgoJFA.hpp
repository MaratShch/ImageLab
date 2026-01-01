#ifndef __IMAGE_LAB_ART_POINTILISM_JFA_ALGO_DEFINITIONS__
#define __IMAGE_LAB_ART_POINTILISM_JFA_ALGO_DEFINITIONS__
 
#include <cstdint>
#include <cfloat> // For FLT_MAX
#include <cmath>  // For sqrt, fabs
#include <algorithm> // For min/max
#include "Common.hpp"
#include "AlgoDot2D.hpp"


// Represents a pixel in the Voronoi Grid (JFA Buffer).
// It stores information about the CLOSEST seed found so far.
struct JFAPixel
{
    int32_t seed_index; // The ID of the closest dot (-1 if empty/unknown)
    float   seed_x;     // The X coordinate of that closest dot
    float   seed_y;     // The Y coordinate of that closest dot
};

void JFA_Init_Buffer
(
   const Point2D* RESTRICT points_in,
   int32_t num_points,
   int32_t width,
   int32_t height,
   JFAPixel* RESTRICT grid_out
) noexcept;

void JFA_Step
(
    const JFAPixel* RESTRICT grid_src,
    JFAPixel* RESTRICT grid_dst,
    int32_t width,
    int32_t height,
    int32_t step_len
);

void Update_Points
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
);


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
);


#endif // __IMAGE_LAB_ART_POINTILISM_JFA_ALGO_DEFINITIONS__