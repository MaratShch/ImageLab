#include "AFMedianControls.hpp"



const AfmfControls getAlgoControls (PF_ParamDef* params[])
{
    AfmfControls controls;

    controls.afmfRadius     = static_cast<int32_t>(params[UnderlyingType(AFMF::eIMAGE_AFMEDIAN_PARAM_RADIUS)]->u.pd.value - 1);
    controls.afmfTolerance  = static_cast<float>  (params[UnderlyingType(AFMF::eIMAGE_AFMEDIAN_PARAM_TOLERANCE)]->u.fs_d.value);
    controls.afmfIterations = static_cast<int32_t>(params[UnderlyingType(AFMF::eIMAGE_AFMEDIAN_PARAM_ITERATIONS)]->u.pd.value - 1);

    return controls;
}