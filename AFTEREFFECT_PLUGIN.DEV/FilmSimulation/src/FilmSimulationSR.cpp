#include "FilmSimulation.hpp"
#include "CommonSmartRender.hpp"

PF_Err
FilmSimulation_PreRender
(
    PF_InData			*in_data,
    PF_OutData			*out_data,
    PF_PreRenderExtra	*extra
)
{
    return PF_Err_NONE;
}


PF_Err
FilmSimulation_SmartRender
(
    PF_InData				*in_data,
    PF_OutData				*out_data,
    PF_SmartRenderExtra		*extraP
)
{
    const double fpsRate = image_lab_get_fps(in_data);
    return PF_Err_NONE;
}