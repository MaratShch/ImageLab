#include "AlgoRules.hpp"
#include "ColorTemperature.hpp"

PF_Err
ColorTemperarture_PreRender
(
    PF_InData			*in_data,
    PF_OutData			*out_data,
    PF_PreRenderExtra	*extra
) noexcept
{
    return PF_Err_NONE;
}


PF_Err
ColorTemperature_SmartRender
(
    PF_InData				*in_data,
    PF_OutData				*out_data,
    PF_SmartRenderExtra		*extraP
) noexcept
{
    return PF_Err_NONE;
}