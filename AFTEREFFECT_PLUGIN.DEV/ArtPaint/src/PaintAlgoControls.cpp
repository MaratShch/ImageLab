#include "ArtPaint.hpp"
#include "ArtPaintEnums.hpp"

const AlgoControls getControlsValues
(
    PF_InData*   RESTRICT in_data,
    PF_OutData*  RESTRICT out_data,
    PF_ParamDef* RESTRICT params[]
)
{
    CACHE_ALIGN AlgoControls algoControls{};

    algoControls.bias    = static_cast<StrokeBias>(params[UnderlyingType(ArtPaintControls::ART_PAINT_STYLE)]->u.pd.value - 1);
    algoControls.sigma   = static_cast<float>(params[UnderlyingType(ArtPaintControls::ART_PAINT_BRUSH_WIDTH)]->u.fs_d.value);
    algoControls.angular = static_cast<float>(params[UnderlyingType(ArtPaintControls::ART_PAINT_BRUSH_LENGTH)]->u.fs_d.value);
    algoControls.angle   = static_cast<float>(params[UnderlyingType(ArtPaintControls::ART_PAINT_STROKE_CURVATIVE)]->u.fs_d.value);
    algoControls.iter    = static_cast<int32_t>(params[UnderlyingType(ArtPaintControls::ART_PAINT_STROKE_SPREADING)]->u.sd.value);

    return algoControls;
}