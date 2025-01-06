#ifndef __IMAGE_LAB_BILATERAL_FILTER_STANDALONE_STRUCTS__
#define __IMAGE_LAB_BILATERAL_FILTER_STANDALONE_STRUCTS__

#include "AE_Effect.h"

#pragma pack(push)
#pragma pack(1)

typedef struct 
{
    A_long fRadius;
    A_FpShort fSigma;
} BFilterParamsStr;

#pragma pack(pop)

constexpr size_t BFilterParamStrSize = sizeof(BFilterParamsStr);

#endif // __IMAGE_LAB_BILATERAL_FILTER_STANDALONE_STRUCTS__