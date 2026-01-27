#include "AFMedianFilter.hpp"
#include "AFMedianFilterEnum.hpp"
#include "CommonSmartRender.hpp"
#include "ImageLabMemInterface.hpp"



PF_Err
AFMedianFilter_PreRender
(
    PF_InData			*in_data,
    PF_OutData			*out_data,
    PF_PreRenderExtra	*extra
) 
{
    PF_Err err = PF_Err_NONE;
    PF_Err errParam = PF_Err_NONE;

    return (PF_Err_NONE == errParam ? err : PF_Err_INTERNAL_STRUCT_DAMAGED);
}


PF_Err
AFMedianFilter_SmartRender
(
    PF_InData				*in_data,
    PF_OutData				*out_data,
    PF_SmartRenderExtra		*extraP
)
{
    return PF_Err_NONE;
}