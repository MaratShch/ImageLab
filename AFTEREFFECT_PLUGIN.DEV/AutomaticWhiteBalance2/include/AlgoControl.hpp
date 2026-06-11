#pragma once

#include <iostream>
#include <cstdint>
#include <algorithm>
#include "ColorTransformMatrix.hpp"
#include "AlgCommonEnums.hpp"

/**
 * @brief Configuration parameters for the PCA-Based Automatic White Balance algorithm.
 * @note This struct integrates with your existing eCOLOR_SPACE, eChromaticAdaptation,
 *       eILLUMINATE, and eCOLOR_OBSERVER type definitions.
 */
struct AlgoControls
{
    // ==========================================
    // Core Pipeline Configurations
    // ==========================================

    /**
     * @brief The color standard of the input/output image (determines linearization curve).
     * @default BT709
     */
    eCOLOR_SPACE colorSpace = BT709;

    /**
     * @brief Chromatic adaptation transform model used to perform the white balance shift.
     * @default CHROMATIC_BRADFORD
     */
    eChromaticAdaptation chromatic = CHROMATIC_BRADFORD;

    /**
     * @brief Target white point / illuminate to map the estimated light source to.
     * @default DAYLIGHT (D65)
     */
    eILLUMINATE illuminate = DAYLIGHT;

    /**
     * @brief CIE standard color observer definition (used for color matching function calculations).
     * @default CieLabDefaultObserver (observer_CIE_1931)
     */
    eCOLOR_OBSERVER observer = CieLabDefaultObserver;


    // ==========================================
    // PCA-Specific Extraction Controls (Float32 Pipeline)
    // ==========================================

    /**
     * @brief Percentage of darkest and brightest pixels to select for PCA.
     * @details Low values isolate pure highlight/shadow light sources; high values prevent noise.
     * @default 3.5f (3.5%)
     * @range [1.0f, 10.0f]
     */
    float percentExtremePixels = 3.5f;

    /**
     * @brief Upper threshold to ignore sensor clipping/saturation in highlights.
     * @details Pixels with any RGB channel exceeding this value are ignored.
     * @default 0.95f (relative to 1.0f maximum intensity in float32 linear space)
     * @range [0.80f, 1.00f]
     */
    float saturationThreshold = 0.95f;

    /**
     * @brief Lower threshold to ignore camera sensor noise in deep shadows.
     * @details Pixels with luminance lower than this value are ignored.
     * @default 0.02f (2% intensity in float32 linear space)
     * @range [0.00f, 0.10f]
     */
    float blackLevelThreshold = 0.02f;
};


AlgoControls getAlgoControlsDefault (void);
