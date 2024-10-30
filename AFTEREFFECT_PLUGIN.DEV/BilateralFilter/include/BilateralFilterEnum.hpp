#ifndef __IMAGE_LAB_BILATERAL_FILTER_STANDALONE_ENUMERATORS__
#define __IMAGE_LAB_BILATERAL_FILTER_STANDALONE_ENUMERATORS__

#include "AE_Effect.h"

typedef enum {
	eBILATERAL_FILTER_INPUT,
	eBILATERAL_FILTER_RADIUS,
    eBILATERAL_TOTAL_CONTROLS
}eAVERAGE_FILTER_ITEMS;


constexpr char FilterWindowSizeStr[]   = "Filter Radius";

constexpr A_long bilateralMinRadius = 0;
constexpr A_long bilateralMaxRadius = 10;
constexpr A_long bilateralDefRadius = bilateralMinRadius;

inline constexpr A_long bilateralWindowSize(const A_long& fRadius) noexcept
{
    return ((fRadius << 1) | 1);
}

constexpr A_long maxWindowSize = bilateralWindowSize(bilateralMaxRadius);
constexpr A_long maxMeshSize = maxWindowSize * maxWindowSize;


#endif // __IMAGE_LAB_BILATERAL_FILTER_STANDALONE_ENUMERATORS__
