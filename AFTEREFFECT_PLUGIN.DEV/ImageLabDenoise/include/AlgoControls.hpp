#pragma once

#include <cstdint>
#include "Common.hpp"
#include "AE_Effect.h"

enum class ProcAccuracy : int32_t
{
    AccDraft = 0,   // Draft (Stride 4) - Fastest for scrubbing the timeline.
    AccStandard,    // Standard (Stride 2) - Good balance (3.9s baseline).
    AccHigh         // High (Stride 1) - Overlapping patches at every pixel.
};

struct AlgoControls
{
    // --- PERFORMANCE ---
    ProcAccuracy accuracy;

    // --- GLOBAL STRENGTH ---
    float master_denoise_amount; 

    // --- CHANNEL SEPARATION ---
    float luma_strength;   
    float chroma_strength; 

    // --- FREQUENCY / SCALE TUNING ---
    float fine_detail_preservation; 
    float coarse_noise_reduction;   
};

constexpr size_t AlgoControlsSize = sizeof(AlgoControls);


AlgoControls GetControlParametersStruct(PF_ParamDef* RESTRICT params[]);
AlgoControls getAlgoControlsDefault(void);