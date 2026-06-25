#ifndef __IMAGE_LAB_ADAPTIVE_FREQUENCY_MEDIAN_FILTER_ALGORITHM__
#define __IMAGE_LAB_ADAPTIVE_FREQUENCY_MEDIAN_FILTER_ALGORITHM__

#include <type_traits>
#include "AlgoControls.hpp"
#include "AlgoMemHandler.hpp"


void Algorithm_Main
(
    const MemHandler& memHandler,
    const int32_t sizeX,
    const int32_t sizeY,
    const AlgoControls& algoCtrl
);


#endif // __IMAGE_LAB_ADAPTIVE_FREQUENCY_MEDIAN_FILTER_ALGORITHM__
