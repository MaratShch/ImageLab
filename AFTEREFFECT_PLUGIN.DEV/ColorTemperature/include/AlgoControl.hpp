#pragma once

#include <iostream>
#include <cstdint>
#include <algorithm>


/**
 * @brief Configuration parameters for the PCA-Based Automatic White Balance algorithm.
 * @note This struct integrates with your existing eCOLOR_SPACE, eChromaticAdaptation,
 *       eILLUMINATE, and eCOLOR_OBSERVER type definitions.
 */
struct AlgoControls
{
    int dummy;
};


AlgoControls getAlgoControlsDefault (void);
