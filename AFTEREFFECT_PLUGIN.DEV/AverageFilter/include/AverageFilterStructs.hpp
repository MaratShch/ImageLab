#ifndef __IMAGE_LAB_AVERAGE_FILTER_STANDALONE_STRUCTS__
#define __IMAGE_LAB_AVERAGE_FILTER_STANDALONE_STRUCTS__

#include "AE_Effect.h"
#include "AverageFilterEnum.hpp"


#pragma pack(push)
#pragma pack(1)

typedef struct 
{
    eAVERAGE_FILTER_WINDOW_SIZE eSize;
    A_long isGeometric;
} AFilterParamsStr;

#pragma pack(pop)

constexpr A_HandleSize AFilterParamStrSize = sizeof(AFilterParamsStr);

#endif // __IMAGE_LAB_AVERAGE_FILTER_STANDALONE_STRUCTS__