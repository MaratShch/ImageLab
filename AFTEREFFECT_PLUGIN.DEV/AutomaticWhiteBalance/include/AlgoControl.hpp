#pragma once

#include <cstdint>
#include <algorithm>
#include "ColorTransformMatrix.hpp"
#include "AlgCommonEnums.hpp"
#include "AE_Effect.h"

struct AlgoControls
{
	eCOLOR_SPACE colorSpace;
	eChromaticAdaptation chromatic;
	eILLUMINATE  illuminate;
	int32_t sliderIterCnt;
	int32_t sliderThreshold;
};

AlgoControls getAlgoControlsDefault (void);
AlgoControls GetControlParametersStruct (PF_ParamDef* params[]);