#include "Common.hpp"
#include "ArtPointillism.hpp"
#include "ArtPointillismControl.hpp"
#include "Param_Utils.h"


PF_Err
SetupControlElements
(
    const PF_InData* RESTRICT in_data,
    PF_OutData* RESTRICT out_data
)
{
    CACHE_ALIGN PF_ParamDef	def{};
    PF_Err		err = PF_Err_NONE;

    constexpr PF_ParamFlags     flags = PF_ParamFlag_SUPERVISE | PF_ParamFlag_CANNOT_TIME_VARY | PF_ParamFlag_CANNOT_INTERP;
    constexpr PF_ParamUIFlags   ui_flags = PF_PUI_NONE;
    constexpr PF_ParamUIFlags   ui_disabled = ui_flags | PF_PUI_DISABLED;

    AEFX_INIT_PARAM_STRUCTURE(def, flags, ui_flags);
    PF_ADD_POPUP(
        controlItemName[0],                                                         // pop-up name
        UnderlyingType(ArtPointillismPainter::ART_POINTILLISM_PAINTER_TOTAL_NUMBER),// number of variants
        UnderlyingType(ArtPointillismPainter::ART_POINTILLISM_PAINTER_SEURAT),      // default variant
        PainterNameStr,                                                             // string for pop-up
        UnderlyingType(ArtPointillismControls::ART_POINTILLISM_PAINTER_STYLE));     // control ID

    AEFX_INIT_PARAM_STRUCTURE(def, flags, ui_flags);
    PF_ADD_SLIDER(
        controlItemName[1], 
        DotDencityMin,
        DotDencityMax,
        DotDencityMin,
        DotDencityMax,
        DotDencityDef,
        UnderlyingType(ArtPointillismControls::ART_POINTILLISM_SLIDER_DOT_DENCITY));

    AEFX_INIT_PARAM_STRUCTURE(def, flags, ui_flags);
    PF_ADD_SLIDER(
        controlItemName[2],
        DotSizeMin,
        DotSizeMax,
        DotSizeMin,
        DotSizeMax,
        DotSizeDef,
        UnderlyingType(ArtPointillismControls::ART_POINTILLISM_SLIDER_DOT_SIZE));

    AEFX_INIT_PARAM_STRUCTURE(def, flags, ui_flags);
    PF_ADD_SLIDER(
        controlItemName[3],
        EdgeSensitivityMin,
        EdgeSensitivityMax,
        EdgeSensitivityMin,
        EdgeSensitivityMax,
        EdgeSensitivityDef,
        UnderlyingType(ArtPointillismControls::ART_POINTILLISM_SLIDER_EDGE_SENSITIVITY));

    AEFX_INIT_PARAM_STRUCTURE(def, flags, ui_flags);
    PF_ADD_SLIDER(
        controlItemName[4],
        ColorVibrancyMin,
        ColorVibrancyMax,
        ColorVibrancyMin,
        ColorVibrancyMax,
        ColorVibrancyDef,
        UnderlyingType(ArtPointillismControls::ART_POINTILLISM_SLIDER_COLOR_VIBRANCE));

    AEFX_INIT_PARAM_STRUCTURE(def, flags, ui_flags);
    PF_ADD_POPUP(
        controlItemName[5],
        UnderlyingType(StrokeShape::ART_POINTILLISM_SHAPE_TOTALS),
        UnderlyingType(StrokeShape::ART_POINTILLISM_SHAPE_CIRCLE),
        StrokeShapeStr,
        UnderlyingType(ArtPointillismControls::ART_POINTILLISM_STROKE_STROKE_SHAPE));

    AEFX_INIT_PARAM_STRUCTURE(def, flags, ui_flags);
    PF_ADD_POPUP(
        controlItemName[6],
        UnderlyingType(BackgroundArt::ART_POINTILLISM_BACKGROUND_TOTALS),
        UnderlyingType(BackgroundArt::ART_POINTILLISM_BACKGROUND_CANVAS),
        BackgroundStr,
        UnderlyingType(ArtPointillismControls::ART_POINTILLISM_BACKGROUND_ART));

    AEFX_INIT_PARAM_STRUCTURE(def, flags, ui_disabled);
    PF_ADD_SLIDER(
        controlItemName[7],
        OpacityMin,
        OpacityMax,
        OpacityMin,
        OpacityMax,
        OpacityDef,
        UnderlyingType(ArtPointillismControls::ART_POINTILLISM_OPACITY));

    AEFX_INIT_PARAM_STRUCTURE(def, flags, ui_flags);
    PF_ADD_SLIDER(
        controlItemName[8],
        RandomSeedMin,
        RandomSeedMax,
        RandomSeedMin,
        RandomSeedMax,
        RandomSeedDef,
        UnderlyingType(ArtPointillismControls::ART_POINTILLISM_RANDOM_SEED));

    out_data->num_params = UnderlyingType(ArtPointillismControls::ART_POINTILLISM_TOTAL_PARAMS);

    return err;
}


PontillismControls GetControlParametersStruct
(
    PF_ParamDef* RESTRICT params[]
) noexcept
{
    CACHE_ALIGN PontillismControls algoParams{};

    algoParams.PainterStyle     = static_cast<ArtPointillismPainter>(params[UnderlyingType(ArtPointillismControls::ART_POINTILLISM_PAINTER_STYLE)]->u.pd.value - 1);
    algoParams.DotDencity       = params[UnderlyingType(ArtPointillismControls::ART_POINTILLISM_SLIDER_DOT_DENCITY)]->u.sd.value;
    algoParams.DotSize          = params[UnderlyingType(ArtPointillismControls::ART_POINTILLISM_SLIDER_DOT_SIZE)]->u.sd.value;
    algoParams.EdgeSensitivity  = params[UnderlyingType(ArtPointillismControls::ART_POINTILLISM_SLIDER_EDGE_SENSITIVITY)]->u.sd.value;
    algoParams.Vibrancy         = params[UnderlyingType(ArtPointillismControls::ART_POINTILLISM_SLIDER_COLOR_VIBRANCE)]->u.sd.value;
    algoParams.Shape            = static_cast<StrokeShape>(params[UnderlyingType(ArtPointillismControls::ART_POINTILLISM_STROKE_STROKE_SHAPE)]->u.pd.value - 1);
    algoParams.Background       = static_cast<BackgroundArt>(params[UnderlyingType(BackgroundArt::ART_POINTILLISM_BACKGROUND_CANVAS)]->u.pd.value - 1);
    algoParams.Opacity          = params[UnderlyingType(ArtPointillismControls::ART_POINTILLISM_OPACITY)]->u.sd.value;
    algoParams.RandomSeed       = params[UnderlyingType(ArtPointillismControls::ART_POINTILLISM_RANDOM_SEED)]->u.sd.value;

    return algoParams;
}