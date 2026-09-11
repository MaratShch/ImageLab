#include "Common.hpp"
#include "CompileTimeUtils.hpp"
#include "AlgoControl.hpp"
#include "AlgoControlEnums.hpp"
#include "AE_Effect.h"



AlgoControls getAlgoControls (PF_ParamDef* params[], const double fps)
{
    CACHE_ALIGN AlgoControls algoParams = getAlgoControlsDefault();

    algoParams.filmProfile = static_cast<film::eFILM_PROFILE>(params[UnderlyingType(FilmSimulationCtrl::FILM_STOCK)]->u.pd.value - 1);
    algoParams.frameRate   = fps;
//    algoParams.filmFormat  = static_cast<FilmFormatCtrl>(params[UnderlyingType(FilmSimulationCtrl::FILM_FORMAT)]->u.pd.value - 1);

    return algoParams;
}