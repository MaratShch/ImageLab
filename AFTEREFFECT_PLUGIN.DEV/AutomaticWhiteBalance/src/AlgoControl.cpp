#include "AE_Effect.h"
#include "AlgoControl.hpp"
#include "ColorTransformMatrix.hpp"

AlgoControls getAlgoControlsDefault (void)
{
    AlgoControls algoCtrl{};
    algoCtrl.colorSpace = BT601;
    algoCtrl.chromatic = CHROMATIC_CAT02;
    algoCtrl.illuminate = DAYLIGHT;
    algoCtrl.sliderIterCnt = 2;
    algoCtrl.sliderThreshold = 30;

    return algoCtrl;
}


AlgoControls GetControlParametersStruct (PF_ParamDef* params[])
{
    AlgoControls algoCtrl{};

    algoCtrl.illuminate = static_cast<eILLUMINATE>(params[AWB_ILLUMINATE_POPUP]->u.pd.value - 1);
    algoCtrl.chromatic  = static_cast<eChromaticAdaptation>(params[AWB_CHROMATIC_POPUP]->u.pd.value - 1);
    algoCtrl.colorSpace = static_cast<eCOLOR_SPACE>(params[AWB_COLOR_SPACE_POPUP]->u.pd.value - 1);
    algoCtrl.sliderThreshold = params[AWB_THRESHOLD_SLIDER]->u.sd.value;
    algoCtrl.sliderIterCnt = params[AWB_ITERATIONS_SLIDER]->u.sd.value;

    return algoCtrl;
}
