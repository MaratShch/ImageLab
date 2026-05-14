#include "AlgoControls.hpp"
#include "AE_Effect.h"

const AlgoControls getAlgoControls (PF_ParamDef* params[])
{
    AlgoControls controls;

    controls.outputType = static_cast<AFMF_Output>(params[UnderlyingType(AFMF::eIMAGE_AFMEDIAN_OUTPUT_TYPE)]->u.pd.value - 1);
    controls.radius     = popup2value(static_cast<int32_t>    (params[UnderlyingType(AFMF::eIMAGE_AFMEDIAN_PARAM_RADIUS)]->u.pd.value - 1));
    controls.tolerance  = static_cast<float>      (params[UnderlyingType(AFMF::eIMAGE_AFMEDIAN_PARAM_TOLERANCE)]->u.fs_d.value / 10.0);
    controls.iterations = popup2value(static_cast<int32_t>    (params[UnderlyingType(AFMF::eIMAGE_AFMEDIAN_PARAM_ITERATIONS)]->u.pd.value - 1));

    controls.Sanitize();

    return controls;
}