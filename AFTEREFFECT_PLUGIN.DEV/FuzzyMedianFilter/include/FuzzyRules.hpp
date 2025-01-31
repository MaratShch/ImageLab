#ifndef __FUZZY_MEDIAN_FILTER_RULES___
#define __FUZZY_MEDIAN_FILTER_RULES___

#ifdef __NVCC__
// CUDA GPU compiler
#include <cuda_runtime.h>
#include <math.h>

 inline __device__ float gaussian_sim
 (
    const float& d,
    const float& m,
    const float& sqStd
 ) noexcept
 {
    return exp(-((d - m) * (d - m)) / (2.f * sqStd));
 }


#else // #ifdef __NVCC__

 #define FAST_COMPUTE_EXTRA_PRECISION
 #include "FastAriphmetics.hpp"
// VS CPU compiler

 inline float gaussian_sim 
 (
    const float& d,     // Correlation with Pixels
    const float& m,     // Mean
    const float& sqStd  // Square of Standard Deviation
 ) noexcept
 {
    const float diff = d - m;
    return FastCompute::Exp(-(diff * diff) / (2.f * sqStd));
 }

#endif // #ifdef __NVCC__



#endif // __FUZZY_MEDIAN_FILTER_RULES___