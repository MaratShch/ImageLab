#include "AlgoArtisticsRendering.hpp"

void Init_Canvas
(
    float* RESTRICT canvas_lab,
    const float* RESTRICT source_lab,
    int32_t width, 
    int32_t height, 
    int32_t mode
)
{
    const int32_t total_pixels = width * height;
    
    // Default: Cream/Canvas (L=96, a=2, b=8 approx)
    float fill_L = 96.0f;
    float fill_a = 2.0f;
    float fill_b = 8.0f;

    if (mode == 1)
	{ // Pure White
        fill_L = 100.0f; fill_a = 0.0f; fill_b = 0.0f;
    }
	else if (mode == 2)
	{ // Transparent / Black init
        fill_L = 0.0f; fill_a = 0.0f; fill_b = 0.0f;
    }
	
    if (mode == 3)
	{
        // Copy Source
        for (int32_t i = 0; i < total_pixels * 3; ++i)
		{
            canvas_lab[i] = source_lab[i];
        }
    }
	else
	{
        // Fill Uniform
        for (int32_t i = 0; i < total_pixels; ++i)
		{
            canvas_lab[i * 3 + 0] = fill_L;
            canvas_lab[i * 3 + 1] = fill_a;
            canvas_lab[i * 3 + 2] = fill_b;
        }
    }
	
	return;
}