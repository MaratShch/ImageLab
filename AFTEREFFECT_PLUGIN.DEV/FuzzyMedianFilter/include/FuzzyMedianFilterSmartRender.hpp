#ifndef __IMAGE_LAB_FUZZY_MEDIAN_FILTER_SMART_RENDER__
#define __IMAGE_LAB_FUZZY_MEDIAN_FILTER_SMART_RENDER__

#include "AE_Effect.h"
#include "FuzzyMedianFilterEnum.hpp"

#pragma pack(push)
#pragma pack(1)

typedef struct
{
    eFUZZY_FILTER_WINDOW_SIZE fWindowSize;
    A_FpShort fSigma;
}FuzzyFilterParamsStr, *PFuzzyFilterParamsStr;

#pragma pack(pop)

constexpr size_t FuzzyFilterParamsStrSize = sizeof(FuzzyFilterParamsStr);


PF_Err
FuzzyMedian_PreRender
(
    PF_InData			*in_data,
    PF_OutData			*out_data,
    PF_PreRenderExtra	*extra
) noexcept;

PF_Err
FuzzyMedian_SmartRender
(
    PF_InData				*in_data,
    PF_OutData				*out_data,
    PF_SmartRenderExtra		*extraP
) noexcept;


#endif // __IMAGE_LAB_FUZZY_MEDIAN_FILTER_SMART_RENDER__