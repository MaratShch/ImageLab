#include "AlgoControl.hpp"
#include "AlgorithmEnums.hpp"
#include "AE_Effect.h"

AlgoControls getAlgoControlsDefault (void)
{
    AlgoControls algoCtrl{};
    
    algoCtrl.colorSpace = BT709;
    algoCtrl.chromatic  = eChromaticAdaptation::CHROMATIC_VON_KRIES;
    algoCtrl.illuminate = eILLUMINATE::DAYLIGHT;
    
    algoCtrl.percentExtremePixels = 3.5f;   // darkest% + brightest% selected   [1.0 .. 10.0]
    algoCtrl.saturationThreshold  = 0.95f;  // ignore any channel >= this       [0.80 .. 1.00]
    algoCtrl.blackLevelThreshold  = 0.02f;  // ignore luminance <  this         [0.00 .. 0.10]
   
    return algoCtrl;
}

AlgoControls getAlgoControlsParameters (PF_ParamDef* params[])
{
    AlgoControls algoCtrl{};

    algoCtrl.colorSpace = static_cast<eCOLOR_SPACE>(params[UnderlyingType(eImageLab2AWB_Controls::AWB2_COLOR_SPACE_POPUP)]->u.pd.value - 1);
    algoCtrl.illuminate = static_cast<eILLUMINATE> (params[UnderlyingType(eImageLab2AWB_Controls::AWB2_ILLUMINATE_POPUP)]->u.pd.value - 1);
    algoCtrl.chromatic  = static_cast<eChromaticAdaptation>(params[UnderlyingType(eImageLab2AWB_Controls::AWB2_CHROMATIC_POPUP)]->u.pd.value - 1);
    algoCtrl.percentExtremePixels = static_cast<float>(params[UnderlyingType(eImageLab2AWB_Controls::AWB2_EXTERME_PIXELS)]->u.fs_d.value);
    algoCtrl.saturationThreshold  = static_cast<float>(params[UnderlyingType(eImageLab2AWB_Controls::AWB2_SATRURATION_THRESHOLD)]->u.fs_d.value);
    algoCtrl.blackLevelThreshold  = static_cast<float>(params[UnderlyingType(eImageLab2AWB_Controls::AWB2_BLACK_LEVEL_THRESHOLD)]->u.fs_d.value);

    return algoCtrl;
}