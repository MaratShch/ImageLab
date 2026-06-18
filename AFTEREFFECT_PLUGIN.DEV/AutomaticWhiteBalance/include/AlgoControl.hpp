#pragma once

#include <cstdint>
#include <algorithm>
#include "ColorTransformMatrix.hpp"
#include "AlgCommonEnums.hpp"


struct AlgoControls
{
	eCOLOR_SPACE colorSpace;
	eChromaticAdaptation chromatic;
	eILLUMINATE  illuminate;
	int32_t sliderIterCnt;
	int32_t sliderThreshold;
};

AlgoControls getAlgoControlsDefault(void);
