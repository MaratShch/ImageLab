#include "ImageLabDenoise.hpp"
#include "ImageLabDenoiseEnum.hpp"
#include "CommonSmartRender.hpp"
#include "ImageLabMemInterface.hpp"


PF_Err
ImageLabDenoise_PreRender
(
    PF_InData			*in_data,
    PF_OutData			*out_data,
    PF_PreRenderExtra	*extra
)
{
    PF_Err err = PF_Err_NONE;
    return err;
}


PF_Err
ImageLabDenoise_SmartRender
(
    PF_InData				*in_data,
    PF_OutData				*out_data,
    PF_SmartRenderExtra		*extraP
)
{
    PF_EffectWorld* input_worldP = nullptr;
    PF_EffectWorld* output_worldP = nullptr;
    PF_Err err = PF_Err_NONE;
    return err;
}
