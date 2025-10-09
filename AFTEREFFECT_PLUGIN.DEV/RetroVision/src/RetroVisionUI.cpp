#include <atomic>
#include "RetroVision.hpp"
#include "RetroVisionEnum.hpp"
#include "RetroVisionGui.hpp"


void
CgaPalette_SetBitmap
(
    const PF_ParamDef* cgaPalette,
    const PF_ParamDef* intencity
)
{
    const PaletteCGA palette = static_cast<PaletteCGA>(cgaPalette->u.pd.value - 1);
    switch (palette)
    {
        case PaletteCGA::eRETRO_PALETTE_CGA1:
            SetBitmapIdx(0 == intencity->u.bd.value ? 1 : 2);
        break;

        case PaletteCGA::eRETRO_PALETTE_CGA2:
            SetBitmapIdx(0 == intencity->u.bd.value ? 3 : 4);
        break;

        default:
        /* nothing */
        break;
    }
    return;
}


void
EgaPalette_SetBitmap
(
    const PF_ParamDef* egaPalette
)
{
    const PaletteEGA palette = static_cast<PaletteEGA>(egaPalette->u.pd.value - 1);
    switch (palette)
    {
        case PaletteEGA::eRETRO_PALETTE_EGA_STANDARD:
            SetBitmapIdx(5);
        break;

        case PaletteEGA::eRETRO_PALETTE_EGA_KING_QUESTS:
            SetBitmapIdx(6);
        break;

        case PaletteEGA::eRETRO_PALETTE_EGA_KYRANDIA:
            SetBitmapIdx(7);
        break;

        case PaletteEGA::eRETRO_PALETTE_EGA_THEXDER:
            SetBitmapIdx(8);
        break;

        case PaletteEGA::eRETRO_PALETTE_EGA_DUNE:
            SetBitmapIdx(9);
        break;

        case PaletteEGA::eRETRO_PALETTE_EGA_DOOM:
            SetBitmapIdx(10);
        break;

        case PaletteEGA::eRETRO_PALETTE_EGA_METAL_MUTANT:
            SetBitmapIdx(11);
        break;

        case PaletteEGA::eRETRO_PALETTE_EGA_WOLFENSTEIN:
            SetBitmapIdx(12);
        break;

        default:
        /* nothing */
        break;
    }
    return;
}


void
VgaPalette_SetBitmap
(
    const PF_ParamDef* vgaPalette
)
{
    const PaletteVGA palette = static_cast<PaletteVGA>(vgaPalette->u.pd.value - 1);
    switch (palette)
    {
        case PaletteVGA::eRETRO_PALETTE_VGA_16_BITS:
            SetBitmapIdx(13);
        break;

        case PaletteVGA::eRETRO_PALETTE_VGA_256_BITS:
            SetBitmapIdx(14);
        break;

        default:
        /* nothing */
        break;
    }
    return;
}


static PF_Err
RetroVision_UpdateControls_UI
(
    PF_InData	*in_data,
    PF_OutData	*out_data,
    PF_ParamDef	*params[],
    bool bAlgoActive,
    const AEFX_SuiteScoper<PF_ParamUtilsSuite3>& paramUtilsSuite
)
{
    if (true == bAlgoActive)
    {
        params[UnderlyingType(RetroVision::eRETRO_VISION_MONITOR_TYPE_START)]->ui_flags &= ~PF_PUI_DISABLED;
        params[UnderlyingType(RetroVision::eRETRO_VISION_DISPLAY)]->ui_flags            &= ~PF_PUI_DISABLED;

        const RetroMonitor monitor = static_cast<RetroMonitor>(params[UnderlyingType(RetroVision::eRETRO_VISION_DISPLAY)]->u.pd.value - 1);
        switch (monitor)
        {
            case RetroMonitor::eRETRO_BITMAP_CGA:
                CgaPalette_SetBitmap (params[UnderlyingType(RetroVision::eRETRO_VISION_CGA_PALETTE)], params[UnderlyingType(RetroVision::eRETRO_VISION_CGA_INTTENCITY_BIT)]);

                params[UnderlyingType(RetroVision::eRETRO_VISION_CGA_PALETTE)]->ui_flags        &= ~PF_PUI_DISABLED;
                params[UnderlyingType(RetroVision::eRETRO_VISION_CGA_INTTENCITY_BIT)]->ui_flags &= ~PF_PUI_DISABLED;

                params[UnderlyingType(RetroVision::eRETRO_VISION_EGA_PALETTE)]->ui_flags                        |= PF_PUI_DISABLED;
                params[UnderlyingType(RetroVision::eRETRO_VISION_VGA_PALETTE)]->ui_flags                        |= PF_PUI_DISABLED;
                params[UnderlyingType(RetroVision::eRETRO_VISION_CRT_ARTIFACTS_HERCULES_WHITE_COLOR)]->ui_flags |= PF_PUI_DISABLED;
                break;

            case RetroMonitor::eRETRO_BITMAP_EGA:
                EgaPalette_SetBitmap (params[UnderlyingType(RetroVision::eRETRO_VISION_EGA_PALETTE)]);

                params[UnderlyingType(RetroVision::eRETRO_VISION_EGA_PALETTE)]->ui_flags &= ~PF_PUI_DISABLED;

                params[UnderlyingType(RetroVision::eRETRO_VISION_CGA_PALETTE)]->ui_flags                        |= PF_PUI_DISABLED;
                params[UnderlyingType(RetroVision::eRETRO_VISION_CGA_INTTENCITY_BIT)]->ui_flags                 |= PF_PUI_DISABLED;
                params[UnderlyingType(RetroVision::eRETRO_VISION_VGA_PALETTE)]->ui_flags                        |= PF_PUI_DISABLED;
                params[UnderlyingType(RetroVision::eRETRO_VISION_CRT_ARTIFACTS_HERCULES_WHITE_COLOR)]->ui_flags |= PF_PUI_DISABLED;
            break;

            case RetroMonitor::eRETRO_BITMAP_VGA:
                VgaPalette_SetBitmap (params[UnderlyingType(RetroVision::eRETRO_VISION_VGA_PALETTE)]);

                params[UnderlyingType(RetroVision::eRETRO_VISION_VGA_PALETTE)]->ui_flags &= ~PF_PUI_DISABLED;

                params[UnderlyingType(RetroVision::eRETRO_VISION_CGA_PALETTE)]->ui_flags                        |= PF_PUI_DISABLED;
                params[UnderlyingType(RetroVision::eRETRO_VISION_CGA_INTTENCITY_BIT)]->ui_flags                 |= PF_PUI_DISABLED;
                params[UnderlyingType(RetroVision::eRETRO_VISION_EGA_PALETTE)]->ui_flags                        |= PF_PUI_DISABLED;
                params[UnderlyingType(RetroVision::eRETRO_VISION_CRT_ARTIFACTS_HERCULES_WHITE_COLOR)]->ui_flags |= PF_PUI_DISABLED;
            break;

            case RetroMonitor::eRETRO_BITMAP_HERCULES:
                SetBitmapIdx(15);

                params[UnderlyingType(RetroVision::eRETRO_VISION_CGA_PALETTE)]->ui_flags        |= PF_PUI_DISABLED;
                params[UnderlyingType(RetroVision::eRETRO_VISION_CGA_INTTENCITY_BIT)]->ui_flags |= PF_PUI_DISABLED;
                params[UnderlyingType(RetroVision::eRETRO_VISION_EGA_PALETTE)]->ui_flags        |= PF_PUI_DISABLED;
                params[UnderlyingType(RetroVision::eRETRO_VISION_VGA_PALETTE)]->ui_flags        |= PF_PUI_DISABLED;

                params[UnderlyingType(RetroVision::eRETRO_VISION_CRT_ARTIFACTS_HERCULES_WHITE_COLOR)]->ui_flags &= ~PF_PUI_DISABLED;
            break;
        }

        params[UnderlyingType(RetroVision::eRETRO_VISION_CRT_ARTIFACTS_START)                   ]->ui_flags &= ~PF_PUI_DISABLED;
        params[UnderlyingType(RetroVision::eRETRO_VISION_CRT_ARTIFACTS_SCANLINES)               ]->ui_flags &= ~PF_PUI_DISABLED;
        params[UnderlyingType(RetroVision::eRETRO_VISION_CRT_ARTIFACTS_SMOOTH_SCANLINES)        ]->ui_flags &= ~PF_PUI_DISABLED;
        params[UnderlyingType(RetroVision::eRETRO_VISION_CRT_ARTIFACTS_SCANLINES_INTERVAL)      ]->ui_flags &= ~PF_PUI_DISABLED;
        params[UnderlyingType(RetroVision::eRETRO_VISION_CRT_ARTIFACTS_SCANLINES_DARKNESS)      ]->ui_flags &= ~PF_PUI_DISABLED;
        params[UnderlyingType(RetroVision::eRETRO_VISION_CRT_ARTIFACTS_PHOSPHOR_GLOW)           ]->ui_flags &= ~PF_PUI_DISABLED;
        params[UnderlyingType(RetroVision::eRETRO_VISION_CRT_ARTIFACTS_PHOSPHOR_GLOW_STRENGHT)  ]->ui_flags &= ~PF_PUI_DISABLED;
        params[UnderlyingType(RetroVision::eRETRO_VISION_CRT_ARTIFACTS_PHOSPHOR_GLOW_OPACITY)   ]->ui_flags &= ~PF_PUI_DISABLED;
        params[UnderlyingType(RetroVision::eRETRO_VISION_CRT_ARTIFACTS_APPERTURE_GRILL)         ]->ui_flags &= ~PF_PUI_DISABLED;
        params[UnderlyingType(RetroVision::eRETRO_VISION_CRT_ARTIFACTS_APPERTURE_GRILL_POPUP)   ]->ui_flags &= ~PF_PUI_DISABLED;
        params[UnderlyingType(RetroVision::eRETRO_VISION_CRT_ARTIFACTS_APPERTURE_GRILL_INTERVAL)]->ui_flags &= ~PF_PUI_DISABLED;
        params[UnderlyingType(RetroVision::eRETRO_VISION_CRT_ARTIFACTS_APPERTURE_GRILL_DARKNESS)]->ui_flags &= ~PF_PUI_DISABLED;

        paramUtilsSuite->PF_UpdateParamUI(in_data->effect_ref, UnderlyingType(RetroVision::eRETRO_VISION_MONITOR_TYPE_START), params[UnderlyingType(RetroVision::eRETRO_VISION_MONITOR_TYPE_START)]);
        paramUtilsSuite->PF_UpdateParamUI(in_data->effect_ref, UnderlyingType(RetroVision::eRETRO_VISION_DISPLAY), params[UnderlyingType(RetroVision::eRETRO_VISION_DISPLAY)]);
        paramUtilsSuite->PF_UpdateParamUI(in_data->effect_ref, UnderlyingType(RetroVision::eRETRO_VISION_CGA_PALETTE), params[UnderlyingType(RetroVision::eRETRO_VISION_CGA_PALETTE)]);
        paramUtilsSuite->PF_UpdateParamUI(in_data->effect_ref, UnderlyingType(RetroVision::eRETRO_VISION_CGA_INTTENCITY_BIT), params[UnderlyingType(RetroVision::eRETRO_VISION_CGA_INTTENCITY_BIT)]);
        paramUtilsSuite->PF_UpdateParamUI(in_data->effect_ref, UnderlyingType(RetroVision::eRETRO_VISION_EGA_PALETTE), params[UnderlyingType(RetroVision::eRETRO_VISION_EGA_PALETTE)]);
        paramUtilsSuite->PF_UpdateParamUI(in_data->effect_ref, UnderlyingType(RetroVision::eRETRO_VISION_VGA_PALETTE), params[UnderlyingType(RetroVision::eRETRO_VISION_VGA_PALETTE)]);
        paramUtilsSuite->PF_UpdateParamUI(in_data->effect_ref, UnderlyingType(RetroVision::eRETRO_VISION_CRT_ARTIFACTS_START), params[UnderlyingType(RetroVision::eRETRO_VISION_CRT_ARTIFACTS_START)]);
        paramUtilsSuite->PF_UpdateParamUI(in_data->effect_ref, UnderlyingType(RetroVision::eRETRO_VISION_CRT_ARTIFACTS_SCANLINES), params[UnderlyingType(RetroVision::eRETRO_VISION_CRT_ARTIFACTS_SCANLINES)]);
        paramUtilsSuite->PF_UpdateParamUI(in_data->effect_ref, UnderlyingType(RetroVision::eRETRO_VISION_CRT_ARTIFACTS_SMOOTH_SCANLINES), params[UnderlyingType(RetroVision::eRETRO_VISION_CRT_ARTIFACTS_SMOOTH_SCANLINES)]);
        paramUtilsSuite->PF_UpdateParamUI(in_data->effect_ref, UnderlyingType(RetroVision::eRETRO_VISION_CRT_ARTIFACTS_SCANLINES_INTERVAL), params[UnderlyingType(RetroVision::eRETRO_VISION_CRT_ARTIFACTS_SCANLINES_INTERVAL)]);
        paramUtilsSuite->PF_UpdateParamUI(in_data->effect_ref, UnderlyingType(RetroVision::eRETRO_VISION_CRT_ARTIFACTS_SCANLINES_DARKNESS), params[UnderlyingType(RetroVision::eRETRO_VISION_CRT_ARTIFACTS_SCANLINES_DARKNESS)]);
        paramUtilsSuite->PF_UpdateParamUI(in_data->effect_ref, UnderlyingType(RetroVision::eRETRO_VISION_CRT_ARTIFACTS_PHOSPHOR_GLOW), params[UnderlyingType(RetroVision::eRETRO_VISION_CRT_ARTIFACTS_PHOSPHOR_GLOW)]);
        paramUtilsSuite->PF_UpdateParamUI(in_data->effect_ref, UnderlyingType(RetroVision::eRETRO_VISION_CRT_ARTIFACTS_PHOSPHOR_GLOW_STRENGHT), params[UnderlyingType(RetroVision::eRETRO_VISION_CRT_ARTIFACTS_PHOSPHOR_GLOW_STRENGHT)]);
        paramUtilsSuite->PF_UpdateParamUI(in_data->effect_ref, UnderlyingType(RetroVision::eRETRO_VISION_CRT_ARTIFACTS_PHOSPHOR_GLOW_OPACITY), params[UnderlyingType(RetroVision::eRETRO_VISION_CRT_ARTIFACTS_PHOSPHOR_GLOW_OPACITY)]);
        paramUtilsSuite->PF_UpdateParamUI(in_data->effect_ref, UnderlyingType(RetroVision::eRETRO_VISION_CRT_ARTIFACTS_APPERTURE_GRILL), params[UnderlyingType(RetroVision::eRETRO_VISION_CRT_ARTIFACTS_APPERTURE_GRILL)]);
        paramUtilsSuite->PF_UpdateParamUI(in_data->effect_ref, UnderlyingType(RetroVision::eRETRO_VISION_CRT_ARTIFACTS_APPERTURE_GRILL_POPUP), params[UnderlyingType(RetroVision::eRETRO_VISION_CRT_ARTIFACTS_APPERTURE_GRILL_POPUP)]);
        paramUtilsSuite->PF_UpdateParamUI(in_data->effect_ref, UnderlyingType(RetroVision::eRETRO_VISION_CRT_ARTIFACTS_APPERTURE_GRILL_INTERVAL), params[UnderlyingType(RetroVision::eRETRO_VISION_CRT_ARTIFACTS_APPERTURE_GRILL_INTERVAL)]);
        paramUtilsSuite->PF_UpdateParamUI(in_data->effect_ref, UnderlyingType(RetroVision::eRETRO_VISION_CRT_ARTIFACTS_APPERTURE_GRILL_DARKNESS), params[UnderlyingType(RetroVision::eRETRO_VISION_CRT_ARTIFACTS_APPERTURE_GRILL_DARKNESS)]);
        paramUtilsSuite->PF_UpdateParamUI(in_data->effect_ref, UnderlyingType(RetroVision::eRETRO_VISION_CRT_ARTIFACTS_HERCULES_WHITE_COLOR), params[UnderlyingType(RetroVision::eRETRO_VISION_CRT_ARTIFACTS_HERCULES_WHITE_COLOR)]);
    }
    else
    {
        // Algorithm is deactivated
        SetBitmapIdx(0);

        params[UnderlyingType(RetroVision::eRETRO_VISION_MONITOR_TYPE_START)                    ]->ui_flags |= PF_PUI_DISABLED;
        params[UnderlyingType(RetroVision::eRETRO_VISION_DISPLAY)                               ]->ui_flags |= PF_PUI_DISABLED;
        params[UnderlyingType(RetroVision::eRETRO_VISION_CGA_PALETTE)                           ]->ui_flags |= PF_PUI_DISABLED;
        params[UnderlyingType(RetroVision::eRETRO_VISION_CGA_INTTENCITY_BIT)                    ]->ui_flags |= PF_PUI_DISABLED;
        params[UnderlyingType(RetroVision::eRETRO_VISION_EGA_PALETTE)                           ]->ui_flags |= PF_PUI_DISABLED;
        params[UnderlyingType(RetroVision::eRETRO_VISION_VGA_PALETTE)                           ]->ui_flags |= PF_PUI_DISABLED;
        params[UnderlyingType(RetroVision::eRETRO_VISION_CGA_INTTENCITY_BIT)                    ]->ui_flags |= PF_PUI_DISABLED;
        params[UnderlyingType(RetroVision::eRETRO_VISION_HERCULES_THRESHOLD)                    ]->ui_flags |= PF_PUI_DISABLED;
        params[UnderlyingType(RetroVision::eRETRO_VISION_CRT_ARTIFACTS_START)                   ]->ui_flags |= PF_PUI_DISABLED;
        params[UnderlyingType(RetroVision::eRETRO_VISION_CRT_ARTIFACTS_SCANLINES)               ]->ui_flags |= PF_PUI_DISABLED;
        params[UnderlyingType(RetroVision::eRETRO_VISION_CRT_ARTIFACTS_SMOOTH_SCANLINES)        ]->ui_flags |= PF_PUI_DISABLED;
        params[UnderlyingType(RetroVision::eRETRO_VISION_CRT_ARTIFACTS_SCANLINES_INTERVAL)      ]->ui_flags |= PF_PUI_DISABLED;
        params[UnderlyingType(RetroVision::eRETRO_VISION_CRT_ARTIFACTS_SCANLINES_DARKNESS)      ]->ui_flags |= PF_PUI_DISABLED;
        params[UnderlyingType(RetroVision::eRETRO_VISION_CRT_ARTIFACTS_PHOSPHOR_GLOW)           ]->ui_flags |= PF_PUI_DISABLED;
        params[UnderlyingType(RetroVision::eRETRO_VISION_CRT_ARTIFACTS_PHOSPHOR_GLOW_STRENGHT)  ]->ui_flags |= PF_PUI_DISABLED;
        params[UnderlyingType(RetroVision::eRETRO_VISION_CRT_ARTIFACTS_PHOSPHOR_GLOW_OPACITY)   ]->ui_flags |= PF_PUI_DISABLED;
        params[UnderlyingType(RetroVision::eRETRO_VISION_CRT_ARTIFACTS_APPERTURE_GRILL)         ]->ui_flags |= PF_PUI_DISABLED;
        params[UnderlyingType(RetroVision::eRETRO_VISION_CRT_ARTIFACTS_APPERTURE_GRILL_POPUP)   ]->ui_flags |= PF_PUI_DISABLED;
        params[UnderlyingType(RetroVision::eRETRO_VISION_CRT_ARTIFACTS_APPERTURE_GRILL_INTERVAL)]->ui_flags |= PF_PUI_DISABLED;
        params[UnderlyingType(RetroVision::eRETRO_VISION_CRT_ARTIFACTS_APPERTURE_GRILL_DARKNESS)]->ui_flags |= PF_PUI_DISABLED;
        params[UnderlyingType(RetroVision::eRETRO_VISION_CRT_ARTIFACTS_HERCULES_WHITE_COLOR)    ]->ui_flags |= PF_PUI_DISABLED;

        paramUtilsSuite->PF_UpdateParamUI(in_data->effect_ref, UnderlyingType(RetroVision::eRETRO_VISION_MONITOR_TYPE_START), params[UnderlyingType(RetroVision::eRETRO_VISION_MONITOR_TYPE_START)]);
        paramUtilsSuite->PF_UpdateParamUI(in_data->effect_ref, UnderlyingType(RetroVision::eRETRO_VISION_DISPLAY), params[UnderlyingType(RetroVision::eRETRO_VISION_DISPLAY)]);
        paramUtilsSuite->PF_UpdateParamUI(in_data->effect_ref, UnderlyingType(RetroVision::eRETRO_VISION_CGA_PALETTE), params[UnderlyingType(RetroVision::eRETRO_VISION_CGA_PALETTE)]);
        paramUtilsSuite->PF_UpdateParamUI(in_data->effect_ref, UnderlyingType(RetroVision::eRETRO_VISION_CGA_INTTENCITY_BIT), params[UnderlyingType(RetroVision::eRETRO_VISION_CGA_INTTENCITY_BIT)]);
        paramUtilsSuite->PF_UpdateParamUI(in_data->effect_ref, UnderlyingType(RetroVision::eRETRO_VISION_EGA_PALETTE), params[UnderlyingType(RetroVision::eRETRO_VISION_EGA_PALETTE)]);
        paramUtilsSuite->PF_UpdateParamUI(in_data->effect_ref, UnderlyingType(RetroVision::eRETRO_VISION_VGA_PALETTE), params[UnderlyingType(RetroVision::eRETRO_VISION_VGA_PALETTE)]);
        paramUtilsSuite->PF_UpdateParamUI(in_data->effect_ref, UnderlyingType(RetroVision::eRETRO_VISION_HERCULES_THRESHOLD), params[UnderlyingType(RetroVision::eRETRO_VISION_HERCULES_THRESHOLD)]);
        paramUtilsSuite->PF_UpdateParamUI(in_data->effect_ref, UnderlyingType(RetroVision::eRETRO_VISION_CRT_ARTIFACTS_START), params[UnderlyingType(RetroVision::eRETRO_VISION_CRT_ARTIFACTS_START)]);
        paramUtilsSuite->PF_UpdateParamUI(in_data->effect_ref, UnderlyingType(RetroVision::eRETRO_VISION_CRT_ARTIFACTS_SCANLINES), params[UnderlyingType(RetroVision::eRETRO_VISION_CRT_ARTIFACTS_SCANLINES)]);
        paramUtilsSuite->PF_UpdateParamUI(in_data->effect_ref, UnderlyingType(RetroVision::eRETRO_VISION_CRT_ARTIFACTS_SMOOTH_SCANLINES), params[UnderlyingType(RetroVision::eRETRO_VISION_CRT_ARTIFACTS_SMOOTH_SCANLINES)]);
        paramUtilsSuite->PF_UpdateParamUI(in_data->effect_ref, UnderlyingType(RetroVision::eRETRO_VISION_CRT_ARTIFACTS_SCANLINES_INTERVAL), params[UnderlyingType(RetroVision::eRETRO_VISION_CRT_ARTIFACTS_SCANLINES_INTERVAL)]);
        paramUtilsSuite->PF_UpdateParamUI(in_data->effect_ref, UnderlyingType(RetroVision::eRETRO_VISION_CRT_ARTIFACTS_SCANLINES_DARKNESS), params[UnderlyingType(RetroVision::eRETRO_VISION_CRT_ARTIFACTS_SCANLINES_DARKNESS)]);
        paramUtilsSuite->PF_UpdateParamUI(in_data->effect_ref, UnderlyingType(RetroVision::eRETRO_VISION_CRT_ARTIFACTS_PHOSPHOR_GLOW), params[UnderlyingType(RetroVision::eRETRO_VISION_CRT_ARTIFACTS_PHOSPHOR_GLOW)]);
        paramUtilsSuite->PF_UpdateParamUI(in_data->effect_ref, UnderlyingType(RetroVision::eRETRO_VISION_CRT_ARTIFACTS_PHOSPHOR_GLOW_STRENGHT), params[UnderlyingType(RetroVision::eRETRO_VISION_CRT_ARTIFACTS_PHOSPHOR_GLOW_STRENGHT)]);
        paramUtilsSuite->PF_UpdateParamUI(in_data->effect_ref, UnderlyingType(RetroVision::eRETRO_VISION_CRT_ARTIFACTS_PHOSPHOR_GLOW_OPACITY), params[UnderlyingType(RetroVision::eRETRO_VISION_CRT_ARTIFACTS_PHOSPHOR_GLOW_OPACITY)]);
        paramUtilsSuite->PF_UpdateParamUI(in_data->effect_ref, UnderlyingType(RetroVision::eRETRO_VISION_CRT_ARTIFACTS_APPERTURE_GRILL), params[UnderlyingType(RetroVision::eRETRO_VISION_CRT_ARTIFACTS_APPERTURE_GRILL)]);
        paramUtilsSuite->PF_UpdateParamUI(in_data->effect_ref, UnderlyingType(RetroVision::eRETRO_VISION_CRT_ARTIFACTS_APPERTURE_GRILL_POPUP), params[UnderlyingType(RetroVision::eRETRO_VISION_CRT_ARTIFACTS_APPERTURE_GRILL_POPUP)]);
        paramUtilsSuite->PF_UpdateParamUI(in_data->effect_ref, UnderlyingType(RetroVision::eRETRO_VISION_CRT_ARTIFACTS_APPERTURE_GRILL_INTERVAL), params[UnderlyingType(RetroVision::eRETRO_VISION_CRT_ARTIFACTS_APPERTURE_GRILL_INTERVAL)]);
        paramUtilsSuite->PF_UpdateParamUI(in_data->effect_ref, UnderlyingType(RetroVision::eRETRO_VISION_CRT_ARTIFACTS_APPERTURE_GRILL_DARKNESS), params[UnderlyingType(RetroVision::eRETRO_VISION_CRT_ARTIFACTS_APPERTURE_GRILL_DARKNESS)]);
        paramUtilsSuite->PF_UpdateParamUI(in_data->effect_ref, UnderlyingType(RetroVision::eRETRO_VISION_CRT_ARTIFACTS_HERCULES_WHITE_COLOR), params[UnderlyingType(RetroVision::eRETRO_VISION_CRT_ARTIFACTS_HERCULES_WHITE_COLOR)]);
    }

    return PF_Err_NONE;
}



static PF_Err
RetroVision_UpdateMonitor_UI
(
    PF_InData	*in_data,
    PF_OutData	*out_data,
    PF_ParamDef	*params[],
    const RetroMonitor monitor,
    const AEFX_SuiteScoper<PF_ParamUtilsSuite3>& paramUtilsSuite
)
{
    switch (monitor)
    {
        case RetroMonitor::eRETRO_BITMAP_CGA:
        {
            CgaPalette_SetBitmap (params[UnderlyingType(RetroVision::eRETRO_VISION_CGA_PALETTE)], params[UnderlyingType(RetroVision::eRETRO_VISION_CGA_INTTENCITY_BIT)]);
            params[UnderlyingType(RetroVision::eRETRO_VISION_CGA_PALETTE)                       ]->ui_flags &= ~PF_PUI_DISABLED;
            params[UnderlyingType(RetroVision::eRETRO_VISION_CGA_INTTENCITY_BIT)                ]->ui_flags &= ~PF_PUI_DISABLED;
            params[UnderlyingType(RetroVision::eRETRO_VISION_EGA_PALETTE)                       ]->ui_flags |= PF_PUI_DISABLED;
            params[UnderlyingType(RetroVision::eRETRO_VISION_VGA_PALETTE)                       ]->ui_flags |= PF_PUI_DISABLED;
            params[UnderlyingType(RetroVision::eRETRO_VISION_HERCULES_THRESHOLD)                ]->ui_flags |= PF_PUI_DISABLED;
            params[UnderlyingType(RetroVision::eRETRO_VISION_CRT_ARTIFACTS_HERCULES_WHITE_COLOR)]->ui_flags |= PF_PUI_DISABLED;
        }
        break;

        case RetroMonitor::eRETRO_BITMAP_EGA:
        {
            EgaPalette_SetBitmap (params[UnderlyingType(RetroVision::eRETRO_VISION_EGA_PALETTE)]);
            params[UnderlyingType(RetroVision::eRETRO_VISION_CGA_PALETTE)                       ]->ui_flags |= PF_PUI_DISABLED;
            params[UnderlyingType(RetroVision::eRETRO_VISION_CGA_INTTENCITY_BIT)                ]->ui_flags |= PF_PUI_DISABLED;
            params[UnderlyingType(RetroVision::eRETRO_VISION_EGA_PALETTE)                       ]->ui_flags &= ~PF_PUI_DISABLED;
            params[UnderlyingType(RetroVision::eRETRO_VISION_VGA_PALETTE)                       ]->ui_flags |= PF_PUI_DISABLED;
            params[UnderlyingType(RetroVision::eRETRO_VISION_HERCULES_THRESHOLD)                ]->ui_flags |= PF_PUI_DISABLED;
            params[UnderlyingType(RetroVision::eRETRO_VISION_CRT_ARTIFACTS_HERCULES_WHITE_COLOR)]->ui_flags |= PF_PUI_DISABLED;
        }
        break;

        case RetroMonitor::eRETRO_BITMAP_VGA:
        {
            VgaPalette_SetBitmap (params[UnderlyingType(RetroVision::eRETRO_VISION_VGA_PALETTE)]);
            params[UnderlyingType(RetroVision::eRETRO_VISION_CGA_PALETTE)                       ]->ui_flags |= PF_PUI_DISABLED;
            params[UnderlyingType(RetroVision::eRETRO_VISION_CGA_INTTENCITY_BIT)                ]->ui_flags |= PF_PUI_DISABLED;
            params[UnderlyingType(RetroVision::eRETRO_VISION_EGA_PALETTE)                       ]->ui_flags |= PF_PUI_DISABLED;
            params[UnderlyingType(RetroVision::eRETRO_VISION_VGA_PALETTE)                       ]->ui_flags &= ~PF_PUI_DISABLED;
            params[UnderlyingType(RetroVision::eRETRO_VISION_HERCULES_THRESHOLD)                ]->ui_flags |= PF_PUI_DISABLED;
            params[UnderlyingType(RetroVision::eRETRO_VISION_CRT_ARTIFACTS_HERCULES_WHITE_COLOR)]->ui_flags |= PF_PUI_DISABLED;
        }
        break;

        case RetroMonitor::eRETRO_BITMAP_HERCULES:
        {
            SetBitmapIdx(15);
            params[UnderlyingType(RetroVision::eRETRO_VISION_CGA_PALETTE)                       ]->ui_flags |= PF_PUI_DISABLED;
            params[UnderlyingType(RetroVision::eRETRO_VISION_CGA_INTTENCITY_BIT)                ]->ui_flags |= PF_PUI_DISABLED;
            params[UnderlyingType(RetroVision::eRETRO_VISION_EGA_PALETTE)                       ]->ui_flags |= PF_PUI_DISABLED;
            params[UnderlyingType(RetroVision::eRETRO_VISION_VGA_PALETTE)                       ]->ui_flags |= PF_PUI_DISABLED;
            params[UnderlyingType(RetroVision::eRETRO_VISION_HERCULES_THRESHOLD)                ]->ui_flags &= ~PF_PUI_DISABLED;
            params[UnderlyingType(RetroVision::eRETRO_VISION_CRT_ARTIFACTS_HERCULES_WHITE_COLOR)]->ui_flags &= ~PF_PUI_DISABLED;
        }
        break;

        default:
        break;
    }

    paramUtilsSuite->PF_UpdateParamUI(in_data->effect_ref, UnderlyingType(RetroVision::eRETRO_VISION_CGA_PALETTE), params[UnderlyingType(RetroVision::eRETRO_VISION_CGA_PALETTE)]);
    paramUtilsSuite->PF_UpdateParamUI(in_data->effect_ref, UnderlyingType(RetroVision::eRETRO_VISION_CGA_INTTENCITY_BIT), params[UnderlyingType(RetroVision::eRETRO_VISION_CGA_INTTENCITY_BIT)]);
    paramUtilsSuite->PF_UpdateParamUI(in_data->effect_ref, UnderlyingType(RetroVision::eRETRO_VISION_EGA_PALETTE), params[UnderlyingType(RetroVision::eRETRO_VISION_EGA_PALETTE)]);
    paramUtilsSuite->PF_UpdateParamUI(in_data->effect_ref, UnderlyingType(RetroVision::eRETRO_VISION_VGA_PALETTE), params[UnderlyingType(RetroVision::eRETRO_VISION_VGA_PALETTE)]);
    paramUtilsSuite->PF_UpdateParamUI(in_data->effect_ref, UnderlyingType(RetroVision::eRETRO_VISION_HERCULES_THRESHOLD), params[UnderlyingType(RetroVision::eRETRO_VISION_HERCULES_THRESHOLD)]);
    paramUtilsSuite->PF_UpdateParamUI(in_data->effect_ref, UnderlyingType(RetroVision::eRETRO_VISION_CRT_ARTIFACTS_HERCULES_WHITE_COLOR), params[UnderlyingType(RetroVision::eRETRO_VISION_CRT_ARTIFACTS_HERCULES_WHITE_COLOR)]);

    return PF_Err_NONE;
}



PF_Err
RetroVision_UserChangedParam
(
    PF_InData						*in_data,
    PF_OutData						*out_data,
    PF_ParamDef						*params[],
    PF_LayerDef						*outputP,
    const PF_UserChangedParamExtra	*which_hitP
)
{
    PF_Err err = PF_Err_NONE;
    AEFX_SuiteScoper<PF_ParamUtilsSuite3> paramUtilsSuite = AEFX_SuiteScoper<PF_ParamUtilsSuite3>(in_data, kPFParamUtilsSuite, kPFParamUtilsSuiteVersion3, out_data);

    switch (which_hitP->param_index)
    {
        case UnderlyingType(RetroVision::eRETRO_VISION_ENABLE):
        {
            const bool selected = (0 != params[UnderlyingType(RetroVision::eRETRO_VISION_ENABLE)]->u.bd.value);
            err = RetroVision_UpdateControls_UI (in_data, out_data, params, selected, paramUtilsSuite);
        }
        break;

        case UnderlyingType(RetroVision::eRETRO_VISION_DISPLAY):
        {
            const RetroMonitor monitor = static_cast<RetroMonitor>(params[UnderlyingType(RetroVision::eRETRO_VISION_DISPLAY)]->u.pd.value - 1);
            err = RetroVision_UpdateMonitor_UI (in_data, out_data, params, monitor, paramUtilsSuite);
        }
        break;

        case UnderlyingType(RetroVision::eRETRO_VISION_CGA_PALETTE):
            CgaPalette_SetBitmap (params[UnderlyingType(RetroVision::eRETRO_VISION_CGA_PALETTE)], params[UnderlyingType(RetroVision::eRETRO_VISION_CGA_INTTENCITY_BIT)]);
        break;

        case UnderlyingType(RetroVision::eRETRO_VISION_CGA_INTTENCITY_BIT):
            CgaPalette_SetBitmap (params[UnderlyingType(RetroVision::eRETRO_VISION_CGA_PALETTE)], params[UnderlyingType(RetroVision::eRETRO_VISION_CGA_INTTENCITY_BIT)]);
        break;

        case UnderlyingType(RetroVision::eRETRO_VISION_EGA_PALETTE):
            EgaPalette_SetBitmap (params[UnderlyingType(RetroVision::eRETRO_VISION_EGA_PALETTE)]);
        break;

        case UnderlyingType(RetroVision::eRETRO_VISION_VGA_PALETTE):
            VgaPalette_SetBitmap (params[UnderlyingType(RetroVision::eRETRO_VISION_VGA_PALETTE)]);
        break;

        default:
        // nothing TODO
        break;
    }

    return err;
}


PF_Err
RetroVision_UpdateParameterUI
(
    PF_InData			*in_data,
    PF_OutData			*out_data,
    PF_ParamDef			*params[],
    PF_LayerDef			*outputP
)
{
    return PF_Err_NONE;
}