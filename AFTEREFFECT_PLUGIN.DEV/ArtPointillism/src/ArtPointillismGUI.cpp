#include "CompileTimeUtils.hpp"
#include "ArtPointillism.hpp"
#include "ArtPointillismEnums.hpp"
#include "DrawbotSuite.h"
#include "AEFX_SuiteHelper.h"
#include <cmath>
#include <atomic>


PF_Err DrawEvent
(
    PF_InData		*in_data,
    PF_OutData		*out_data,
    PF_ParamDef		*params[],
    PF_LayerDef		*output,
    PF_EventExtra	*event_extra
)
{
    DRAWBOT_DrawRef drawing_ref = nullptr;
    DRAWBOT_SupplierRef	supplier_ref = nullptr;
    DRAWBOT_SurfaceRef	surface_ref = nullptr;

    PF_Err err = PF_Err_INTERNAL_STRUCT_DAMAGED;

    return PF_Err_NONE;
}