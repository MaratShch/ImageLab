#include <cstring>
#include "ImageLabDenoise.hpp"
#include "ImageLabDenoiseEnum.hpp"
#include "PrSDKAESupport.h"


PF_Err
ImageLabDenoise_SequenceSetup
(
    PF_InData		*in_data,
    PF_OutData		*out_data,
    PF_ParamDef		*params[],
    PF_LayerDef		*output
)
{
    PF_Err err = PF_Err_NONE;
    return err;
}


PF_Err
ImageLabDenoise_SequenceReSetup
(
    PF_InData		*in_data,
    PF_OutData		*out_data,
    PF_ParamDef		*params[],
    PF_LayerDef		*output
)
{
    PF_Err err = PF_Err_NONE;
    return err;
}


PF_Err
ImageLabDenoise_SequenceFlatten
(
    PF_InData		*in_data,
    PF_OutData		*out_data,
    PF_ParamDef		*params[],
    PF_LayerDef		*output
)
{
    PF_Err err = PF_Err_NONE;
    return err;
}


PF_Err
ImageLabDenoise_SequenceSetdown
(
    PF_InData		*in_data,
    PF_OutData		*out_data,
    PF_ParamDef		*params[],
    PF_LayerDef		*output
)
{
    if (nullptr != in_data->sequence_data)
    {
        auto const handleSuite = AEFX_SuiteScoper<PF_HandleSuite1>(in_data, kPFHandleSuite, kPFHandleSuiteVersion1, out_data);
        handleSuite->host_dispose_handle (in_data->sequence_data);
        in_data->sequence_data = nullptr;
    }

    // Invalidate the sequence_data pointers in both AE's input and output data fields (to signal that we have properly disposed of the data)
    out_data->sequence_data = nullptr;

    return PF_Err_NONE;
}
