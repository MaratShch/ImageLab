#ifndef __IMAGE_LAB_ADAPTIVE_FREQUENCY_MEDIAN_FILTER_CONTROLS__
#define __IMAGE_LAB_ADAPTIVE_FREQUENCY_MEDIAN_FILTER_CONTROLS__

#include <cstdint>
#include <algorithm>
#include "AFMedianFilterEnum.hpp"

struct AlgoControls
{
    // ==========================================
    // Control Values
    // ==========================================
    // Renamed slightly to match the Algorithm_Main variables we wrote
    int32_t radius;
    float   tolerance;
    int32_t iterations;

    // ==========================================
    // Constructor
    // ==========================================
    constexpr AlgoControls(void) 
        : radius(kernelRadiusMin)
        , tolerance(noiseToleranceMin)
        , iterations(iterPassMin)
    {}

    // ==========================================
    // Defensive Sanitizer
    // ==========================================
    // Call this immediately after pulling values from the Adobe UI, 
    // before passing the struct to Algorithm_Main.
    void Sanitize() 
    {
        radius     = std::max(kernelRadiusMin, std::min(radius, kernelRadiusMax));
        tolerance  = std::max(noiseToleranceMin, std::min(tolerance, noiseToleranceMax));
        iterations = std::max(iterPassMin, std::min(iterations, iterPassMax));
    }
};

AlgoControls getAlgoControlsDefault (void);


#endif // __IMAGE_LAB_ADAPTIVE_FREQUENCY_MEDIAN_FILTER_CONTROLS__