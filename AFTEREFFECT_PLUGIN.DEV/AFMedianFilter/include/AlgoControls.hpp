#ifndef __IMAGE_LAB_ADAPTIVE_FREQUENCY_MEDIAN_FILTER_CONTROLS__
#define __IMAGE_LAB_ADAPTIVE_FREQUENCY_MEDIAN_FILTER_CONTROLS__

#include <cstdint>
#include <algorithm>
#include "AFMedianFilterEnum.hpp"

struct AlgoControls
{
    // ==========================================
    // UI Boundary Constants (Single Source of Truth)
    // ==========================================
    static constexpr int32_t RADIUS_MIN = kernelRadiusMin;
    static constexpr int32_t RADIUS_MAX = kernelRadiusMax; // The Allocator uses this for Max Padding!

    static constexpr float   TOLERANCE_MIN = noiseToleranceMin / 10.f; // Assuming 0.0f
    static constexpr float   TOLERANCE_MAX = noiseToleranceMax / 10.f; // Assuming 1.0f
    static constexpr float   TOLERANCE_DEF = noiseToleranceDef / 10.f; // Default starting value = 0.10f

    static constexpr int32_t ITER_MIN = iterPassMin;
    static constexpr int32_t ITER_MAX = iterPassMax;

    // Default output mode = the denoised image itself
    static constexpr AFMF_Input  INPUT_DEF  = AFMF_Input ::AFMF_INPUT_LUMINANCE;
    static constexpr AFMF_Output OUTPUT_DEF = AFMF_Output::AFMF_OUTPUT_IMAGE;

    // ==========================================
    // Control Values
    // ==========================================
    int32_t     radius;
    float       tolerance;
    int32_t     iterations;
    AFMF_Input  inputType;
    AFMF_Output outputType;

    // ==========================================
    // Constructor
    // ==========================================
    constexpr AlgoControls(void) 
        : radius(RADIUS_MIN)
        , tolerance(TOLERANCE_DEF)
        , iterations(ITER_MIN)
        , inputType(INPUT_DEF)
        , outputType(OUTPUT_DEF)
    {}

    // ==========================================
    // Defensive Sanitizer
    // ==========================================
    // Call this immediately after pulling values from the Adobe UI, 
    // before passing the struct to Algorithm_Main.
    void Sanitize() 
    {
        radius     = std::max(RADIUS_MIN,    std::min(radius,     RADIUS_MAX));
        tolerance  = std::max(TOLERANCE_MIN, std::min(tolerance,  TOLERANCE_MAX));
        iterations = std::max(ITER_MIN,      std::min(iterations, ITER_MAX));

        // Validate enum: anything not in the known set falls back to the default.
        if (outputType != AFMF_Output::AFMF_OUTPUT_IMAGE &&
            outputType != AFMF_Output::AFMF_OUTPUT_NOISE_MAP)
        {
            outputType = OUTPUT_DEF;
        }
        // Validate enum: anything not in the known set falls back to the default.
        if (inputType != AFMF_Input::AFMF_INPUT_LUMINANCE &&
            inputType != AFMF_Input::AFMF_INPUT_ALL_RGB)
        {
            inputType = INPUT_DEF;
        }
    }
};

AlgoControls getAlgoControlsDefault (void);

//const AlgoControls getAlgoControls (PF_ParamDef* params[]);


#endif // __IMAGE_LAB_ADAPTIVE_FREQUENCY_MEDIAN_FILTER_CONTROLS__