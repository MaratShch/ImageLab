#ifndef __FUZZY_MEDIAN_FILTER_RULES___
#define __FUZZY_MEDIAN_FILTER_RULES___

// CUDA GPU compiler
#include <cuda_runtime.h>
#include <math.h>

#ifdef __NVCC__

 #ifndef INLINE_ALGO_CALL 
  #define INLINE_ALGO_CALL  inline __device__
 #endif

 INLINE_ALGO_CALL float gaussian_sim
 (
    const float& d,
    const float& m,
    const float& sqSigma
 ) noexcept
 {
     const float diff = d - m;
     return exp(-(diff * diff) / (2.f * sqSigma));
 }


#else // #ifdef __NVCC__

 #ifndef INLINE_ALGO_CALL 
  #define INLINE_ALGO_CALL  inline
 #endif

 #define FAST_COMPUTE_EXTRA_PRECISION
 #include "FastAriphmetics.hpp"
// VS CPU compiler

 inline float gaussian_sim 
 (
    const float& d,         // Correlation with Pixels
    const float& m,         // Mean
    const float& sqSigma    // Square of Sigma
 ) noexcept
 {
    const float diff = d - m;
    return FastCompute::Exp(-(diff * diff) / (2.f * sqSigma));
 }

#endif // #ifdef __NVCC__



#endif // __FUZZY_MEDIAN_FILTER_RULES___