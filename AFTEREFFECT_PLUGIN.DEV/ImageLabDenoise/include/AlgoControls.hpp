#pragma once

#include <cstdint>

enum class ProcAccuracy : int32_t
{
	AccDraft = 0,	// Draft (Stride 5) - Fastest for scrubbing the timeline.
	AccStandard,	// Standard (Stride 3) - Good balance.
	AccHigh,		// High (Stride 2) - Standard high-quality.
	AccMaster	 	// Master (Stride 1) - Slowest, best for final render.
};

struct AlgoControls
{
    float denoise_amount;
    float luma_strength;
    float chroma_strength;

    float detail_preservation;
    float match_sensitivity;
    
    int32_t search_radius;
    int32_t stride;
    
    float low_freq_mult;
    float high_freq_mult;
    
    int32_t block_size;
    
    float noise_curve_a; 
    float noise_curve_b;  
};

AlgoControls getAlgoControlsDefault (void);