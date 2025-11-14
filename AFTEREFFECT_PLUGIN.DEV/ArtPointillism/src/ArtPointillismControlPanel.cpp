#include "Common.hpp"
#include "ArtPointillism.hpp"
#include "ArtPointillismControl.hpp"
#include "Param_Utils.h"


PF_Err
SetupControlElements
(
    const PF_InData*  in_data,
    PF_OutData* out_data
)
{
    CACHE_ALIGN PF_ParamDef	def{};
    PF_Err		err = PF_Err_NONE;

    constexpr PF_ParamFlags   flags = PF_ParamFlag_SUPERVISE;
    constexpr PF_ParamUIFlags ui_flags = PF_PUI_NONE;

    AEFX_INIT_PARAM_STRUCTURE(def, flags, ui_flags);
    PF_ADD_POPUP(
        controlItemName[0],                                                         // pop-up name
        UnderlyingType(ArtPointilismPainter::ART_POINTILISM_PAINTER_TOTAL_NUMBER),  // number of variants
        UnderlyingType(ArtPointilismPainter::ART_POINTILISM_PAINTER_SEURAT),        // default variant
        PainterNameStr,                                                             // string for pop-up
        UnderlyingType(ArtPointilismControls::ART_POINTILISM_LIST_PAINTERS));       // control ID
                                                                                    // Setup 'Retro Monitor' popup - default value "CGA"
    AEFX_INIT_PARAM_STRUCTURE(def, flags, ui_flags);
    PF_ADD_SLIDER(
        controlItemName[1],
        dotSizeMin,
        dotSizeMax,
        dotSizeMin,
        dotSizeMax,
        dotSizeDef,
        UnderlyingType(ArtPointilismControls::ART_POINTILISM_SLIDER_DOT_SIZE));

    AEFX_INIT_PARAM_STRUCTURE(def, flags, ui_flags);
    PF_ADD_FLOAT_SLIDERX(
        controlItemName[2],
        fDotDencityMin,
        fDotDencityMax,
        fDotDencityMin,
        fDotDencityMax,
        fDotDencityDef,
        PF_Precision_HUNDREDTHS,
        0,
        0,
        UnderlyingType(ArtPointilismControls::ART_POINTILISM_FSLIDER_DOT_DENCITY));

    AEFX_INIT_PARAM_STRUCTURE(def, flags, ui_flags);
    PF_ADD_FLOAT_SLIDERX(
        controlItemName[3],
        fColorFidelityMin,
        fColorFidelityMax,
        fColorFidelityMin,
        fColorFidelityMax,
        fColorFidelityDef,
        PF_Precision_HUNDREDTHS,
        0,
        0,
        UnderlyingType(ArtPointilismControls::ART_POINTILISM_FSLIDER_COLOR_FIDELITY));

    AEFX_INIT_PARAM_STRUCTURE(def, flags, ui_flags);
    PF_ADD_POPUP(
        controlItemName[4],                                                         // pop-up name
        UnderlyingType(ArtPointilismBlending::ART_POINTILISM_BLEND_TOTAL_NUMBER),   // number of variants
        UnderlyingType(ArtPointilismBlending::ART_POINTILISM_BLEND_NONE),           // default variant
        PointillismBlendModeStr,                                                    // string for pop-up
        UnderlyingType(ArtPointilismControls::ART_POINTILISM_BLENDING_MODE_LIST));  // control ID

    AEFX_INIT_PARAM_STRUCTURE(def, flags, ui_flags);
    PF_ADD_POPUP(
        controlItemName[5],                                                         // pop-up name
        UnderlyingType(ArtPointilismStroke::ART_POINTILISM_STROKE_TOTAL_NUMBER),    // number of variants
        UnderlyingType(ArtPointilismStroke::ART_POINTILISM_STROKE_CIRCLE),          // default variant
        PointillismStrokeStr,                                                       // string for pop-up
        UnderlyingType(ArtPointilismControls::ART_POINTILISM_STROKE_SHAPE_LIST));   // control ID

    AEFX_INIT_PARAM_STRUCTURE(def, flags, ui_flags);
    PF_ADD_FLOAT_SLIDERX(
        controlItemName[6],
        fEdgeSensitiveMin,
        fEdgeSensitiveMax,
        fEdgeSensitiveMin,
        fEdgeSensitiveMax,
        fEdgeSensitiveDef,
        PF_Precision_HUNDREDTHS,
        0,
        0,
        UnderlyingType(ArtPointilismControls::ART_POINTILISM_FSLIDER_EDGE_SENSITIVITY));

    AEFX_INIT_PARAM_STRUCTURE(def, flags, ui_flags);
    PF_ADD_FLOAT_SLIDERX(
        controlItemName[7],
        fBackgroundColorMin,
        fBackgroundColorMax,
        fBackgroundColorMin,
        fBackgroundColorMax,
        fBackgroundColorDef,
        PF_Precision_HUNDREDTHS,
        0,
        0,
        UnderlyingType(ArtPointilismControls::ART_POINTILISM_SLIDER_BACKGROUND_COLOR));

    out_data->num_params = UnderlyingType(ArtPointilismControls::ART_POINTILISM_TOTAL_PARAMS);

    return err;
}


PontillismControls GetControlParametersStruct
(
    PF_ParamDef* params[]
) noexcept
{
    PontillismControls algoParams{};

    algoParams.ctrlPainter          = static_cast<ArtPointilismPainter>(params[UnderlyingType(ArtPointilismControls::ART_POINTILISM_LIST_PAINTERS)]->u.pd.value - 1);
    algoParams.ctrlDotSize          = static_cast<int32_t>(params[UnderlyingType(ArtPointilismControls::ART_POINTILISM_SLIDER_DOT_SIZE)]->u.sd.value);
    algoParams.ctrlDotDensity       = static_cast<float>(params[UnderlyingType(ArtPointilismControls::ART_POINTILISM_FSLIDER_DOT_DENCITY)]->u.fs_d.value);
    algoParams.ctrlColorFidelity    = static_cast<float>(params[UnderlyingType(ArtPointilismControls::ART_POINTILISM_FSLIDER_COLOR_FIDELITY)]->u.fs_d.value);
    algoParams.ctrlBlending         = static_cast<ArtPointilismBlending>(params[UnderlyingType(ArtPointilismControls::ART_POINTILISM_BLENDING_MODE_LIST)]->u.pd.value - 1);;
    algoParams.ctrlStroke           = static_cast<ArtPointilismStroke>(params[UnderlyingType(ArtPointilismControls::ART_POINTILISM_STROKE_SHAPE_LIST)]->u.pd.value - 1);
    algoParams.ctrlEdgeSensitivity  = static_cast<float>(params[UnderlyingType(ArtPointilismControls::ART_POINTILISM_FSLIDER_EDGE_SENSITIVITY)]->u.fs_d.value);
    algoParams.ctrlBackgroundColor  = static_cast<float>(params[UnderlyingType(ArtPointilismControls::ART_POINTILISM_SLIDER_BACKGROUND_COLOR)]->u.fs_d.value);;

    return algoParams;
}