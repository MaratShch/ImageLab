#ifndef __IMAGE_LAB_FUZZYL_MEDIAN_FILTER_ENUMERATORS__
#define __IMAGE_LAB_FUZZYL_MEDIAN_FILTER_ENUMERATORS__

#ifndef __NVCC__
#include "AE_Effect.h"
#endif

typedef enum {
    eFUZZY_MEDIAN_FILTER_INPUT,
    eFUZZY_MEDIAN_FILTER_KERNEL_SIZE,
    eFUZZY_MEDIAN_FILTER_SIGMA_VALUE,
    eFUZZY_MEDIAN_TOTAL_CONTROLS
}eFUZZY_FILTER_ITEMS;


constexpr char strPopupName [] = "Filter Window";
constexpr char strSliderName[] = "Similarity Threshold";

typedef enum {
    eFUZZY_FILTER_BYPASSED,
    eFUZZY_FILTER_WINDOW_3x3,
    eFUZZY_FILTER_WINDOW_5x5,
    eFUZZY_FILTER_WINDOW_7x7,
    eFUZZY_FILTER_TOTAL_VARIANTS
}eFUZZY_FILTER_WINDOW_SIZE;

constexpr char strWindowSizes[] =
{
    "None |"
    "3 x 3|"
    "5 x 5|"
    "7 x 7"
};


constexpr float fSliderValMin = 0.5f;
constexpr float fSliderValMax = 7.5f;
constexpr float fSliderValDefault = (fSliderValMax + fSliderValMin) / 2.f;

#endif // __IMAGE_LAB_FUZZYL_MEDIAN_FILTER_ENUMERATORS__