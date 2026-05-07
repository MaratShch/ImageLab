#ifndef __IMAGE_LAB2_NOSIE_REDUCTION_ALGO_COLOR_CONVERT__
#define __IMAGE_LAB2_NOSIE_REDUCTION_ALGO_COLOR_CONVERT__

#include <immintrin.h>
#include <cstdint>
#include "Common.hpp"
#include "CommonPixFormat.hpp"
#include "AlgoMemHandler.hpp"

void dispatch_convert_to_planar
(
    const PF_Pixel_BGRA_8u* imgInBuffer, 
    const MemHandler& memHndl, 
    const int32_t sizeX, 
    const int32_t sizeY, 
    const int32_t linePitch
);

void dispatch_convert_to_interleaved
(
    const MemHandler& memHndl,
    const PF_Pixel_BGRA_8u* originalInBuffer, // Needed to copy the Alpha channel
    PF_Pixel_BGRA_8u* outBuffer,              // The final Adobe render destination
    const int32_t sizeX, 
    const int32_t sizeY, 
    const int32_t inLinePitchPixels,          // MUST be signed int32_t!
    const int32_t outLinePitchPixels          // MUST be signed int32_t!
);

#endif // __IMAGE_LAB2_NOSIE_REDUCTION_ALGO_COLOR_CONVERT__
