#ifndef __IMAGE_LAB_FAST_PLATFORM_COMPUTATION_UTILS__
#define __IMAGE_LAB_FAST_PLATFORM_COMPUTATION_UTILS__

#include "hw_platform.hpp"

#ifndef FLOATING_POINT_FAST_COMPUTE
 #if !defined __INTEL_COMPILER 
  #define FLOATING_POINT_FAST_COMPUTE() \
   _MM_SET_DENORMALS_ZERO_MODE(_MM_DENORMALS_ZERO_ON); _MM_SET_FLUSH_ZERO_MODE(_MM_FLUSH_ZERO_ON);      
 #else
  #define FLOATING_POINT_FAST_COMPUTE() 
 #endif /* !__INTEL_COMPILER */
#endif /* FLOATING_POINT_FAST_COMPUTE */

#endif /* __IMAGE_LAB_FAST_PLATFORM_COMPUTATION_UTILS__ */ 