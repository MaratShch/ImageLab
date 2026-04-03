#pragma once

#include <cstdint>
#include "Common.hpp"


enum class StrokeBias : int32_t
{
    DarkBias_Open = 0,  // Standard Opening (Shrinks brights, expands darks)
    LightBias_Close,    // Standard Closing (Shrinks darks, expands brights)
    Balanced_ASF,       // Alternating Sequential Filter (Opening + Closing)
    TotalStrokeBias
};

constexpr float sigmaMin = 1.0f;
constexpr float sigmaMax = 25.f;
constexpr float sigmaDef = 5.0f;

constexpr float angularMin = 1.0f;
constexpr float angularMax = 90.f;
constexpr float angularDef = 9.0f;

constexpr float angleMin = 1.0f;
constexpr float angleMax = 90.f;
constexpr float angleDef = 30.0f;

constexpr int32_t iterMin = 1;
constexpr int32_t iterMax = 20;
constexpr int32_t iterDef = 5;

struct AlgoControls
{
    float sigma;      // e.g., 5.0f (Brush Size / Tensor smoothing)
    float angular;    // e.g., 9.0f (Stroke flow tolerance)
    float angle;      // e.g., 30.0f (Stroke conic tolerance)
    int32_t iter;     // e.g., 5 (Stroke length / thickness)
    StrokeBias bias;  // Toggle for Open, Close, or Balanced ASF
    
    // Default constructor to your baseline values
    constexpr AlgoControls() noexcept : sigma(sigmaDef), angular(angularDef), angle(angleDef), iter(iterDef), bias(StrokeBias::DarkBias_Open) {} 
};