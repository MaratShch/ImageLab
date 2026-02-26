#include "AlgoControls.hpp"
#include "ImageLabDenoise.hpp"
#include "ImageLabDenoiseEnum.hpp"

PF_Err
SetupControlElements
(
    const PF_InData*  RESTRICT in_data,
          PF_OutData* RESTRICT out_data
)
{
    CACHE_ALIGN PF_ParamDef	def{};
    PF_Err		err = PF_Err_NONE;

    constexpr PF_ParamFlags     flags = PF_ParamFlag_SUPERVISE | PF_ParamFlag_CANNOT_TIME_VARY | PF_ParamFlag_CANNOT_INTERP;
    constexpr PF_ParamUIFlags   ui_flags = PF_PUI_NONE;

    AEFX_INIT_PARAM_STRUCTURE(def, flags, ui_flags);
    PF_ADD_POPUP
    (
        controlItemName[0],
        UnderlyingType(eDenoiseMethod::eIMAGE_LAB_DENOISE_TOTAL),
        UnderlyingType(eDenoiseMethod::eIMAGE_LAB_DENOISE_DRAFT),
        eDenoiseMethodStr,
        UnderlyingType(eDenoiseControl::eIMAGE_LAB_DENOISE_ACC_SANDARD)
    );

    AEFX_INIT_PARAM_STRUCTURE(def, flags, ui_flags);
    PF_ADD_FLOAT_SLIDERX
    (
        controlItemName[1],
        MasterDenoiseAmountMin,
        MasterDenoiseAmountMax,
        MasterDenoiseAmountMin,
        MasterDenoiseAmountMax,
        MasterDenoiseAmountDef,
        PF_Precision_HUNDREDTHS,
        0,
        0,
        UnderlyingType(eDenoiseControl::eIMAGE_LAB_DENOISE_AMOUNT)
    );

    AEFX_INIT_PARAM_STRUCTURE(def, flags, ui_flags);
    PF_ADD_FLOAT_SLIDERX
    (
        controlItemName[2],
        LumaStrengthMin,
        LumaStrengthMax,
        LumaStrengthMin,
        LumaStrengthMax,
        LumaStrengthDef,
        PF_Precision_HUNDREDTHS,
        0,
        0,
        UnderlyingType(eDenoiseControl::eIMAGE_LAB_DENOISE_LUMA_STRENGTH)
    );

    AEFX_INIT_PARAM_STRUCTURE(def, flags, ui_flags);
    PF_ADD_FLOAT_SLIDERX
    (
        controlItemName[3],
        ChromaStrengthMin,
        ChromaStrengthMax,
        ChromaStrengthMin,
        ChromaStrengthMax,
        ChromaStrengthDef,
        PF_Precision_HUNDREDTHS,
        0,
        0,
        UnderlyingType(eDenoiseControl::eIMAGE_LAB_DENOISE_CHROMA_STRENGTH)
    );

    AEFX_INIT_PARAM_STRUCTURE(def, flags, ui_flags);
    PF_ADD_FLOAT_SLIDERX
    (
        controlItemName[4],
        DetailsPreservationMin,
        DetailsPreservationMax,
        DetailsPreservationMin,
        DetailsPreservationMax,
        DetailsPreservationDef,
        PF_Precision_HUNDREDTHS,
        0,
        0,
        UnderlyingType(eDenoiseControl::eIMAGE_LAB_DENOISE_DETAILS_PRESERVATION)
    );

    AEFX_INIT_PARAM_STRUCTURE(def, flags, ui_flags);
    PF_ADD_FLOAT_SLIDERX
    (
        controlItemName[5],
        CoarseNoiseMin,
        CoarseNoiseMax,
        CoarseNoiseMin,
        CoarseNoiseMax,
        CoarseNoiseDef,
        PF_Precision_HUNDREDTHS,
        0,
        0,
        UnderlyingType(eDenoiseControl::eIMAGE_LAB_DENOISE_COARSE_NOISE)
    );

    out_data->num_params = UnderlyingType(eDenoiseControl::eIMAGE_LAB_DENOISE_CONTROLS);

    return err;
}


AlgoControls GetControlParametersStruct
(
    PF_ParamDef* RESTRICT params[]
)
{
    CACHE_ALIGN AlgoControls algoCtrl{};

    algoCtrl.accuracy = static_cast<ProcAccuracy>(params[UnderlyingType(eDenoiseControl::eIMAGE_LAB_DENOISE_ACC_SANDARD)]->u.pd.value - 1);
    algoCtrl.master_denoise_amount = static_cast<float>(params[UnderlyingType(eDenoiseControl::eIMAGE_LAB_DENOISE_AMOUNT)]->u.fs_d.value);
    algoCtrl.luma_strength         = static_cast<float>(params[UnderlyingType(eDenoiseControl::eIMAGE_LAB_DENOISE_LUMA_STRENGTH)]->u.fs_d.value);
    algoCtrl.chroma_strength       = static_cast<float>(params[UnderlyingType(eDenoiseControl::eIMAGE_LAB_DENOISE_CHROMA_STRENGTH)]->u.fs_d.value);
    algoCtrl.fine_detail_preservation = static_cast<float>(params[UnderlyingType(eDenoiseControl::eIMAGE_LAB_DENOISE_DETAILS_PRESERVATION)]->u.fs_d.value);
    algoCtrl.coarse_noise_reduction = static_cast<float>(params[UnderlyingType(eDenoiseControl::eIMAGE_LAB_DENOISE_COARSE_NOISE)]->u.fs_d.value);

    return algoCtrl;
}



AlgoControls getAlgoControlsDefault(void)
{
    CACHE_ALIGN AlgoControls algoCtrl{};
    
    // =========================================================
    // GLOBAL STRENGTH
    // =========================================================
    
    // Usage: Scales the baseline noise variance estimated by the Oracle across all channels and scales.
    // Impact: Values > 1.0 force stronger blurring/denoising. Values < 1.0 recover original film grain.
    // Range: [0.0f to 3.0f]
    // Default: 1.0f (Perfect mathematical trust in the Blind Noise Oracle)
    algoCtrl.master_denoise_amount = 1.0f;
    
    // =========================================================
    // CHANNEL SEPARATION
    // =========================================================

    // Usage: Applied directly to the Y-channel (Luma) covariance matrix during Step 1 and Step 2.
    // Impact: Controls how much structural/brightness noise is removed. Often kept lower than chroma to preserve edges.
    // Range: [0.0f to 3.0f]
    // Default: 1.0f
    algoCtrl.luma_strength = 1.0f;

    // Usage: Applied directly to the U and V channel (Chroma) covariance matrices during Step 1 and Step 2.
    // Impact: Controls how aggressively color blotches and chromatic noise are eliminated. 
    // Range: [0.0f to 3.0f]
    // Default: 1.0f
    algoCtrl.chroma_strength = 1.0f;

    // =========================================================
    // FREQUENCY / SCALE TUNING
    // =========================================================

    // Usage: Modifies the noise variance multiplier specifically at Level 0 (Full Resolution) of the Laplacian pyramid.
    // Impact: Lower values (e.g., 0.5) protect fine, high-frequency details (pores, fabric textures, fine grain). 
    // Range: [0.0f to 2.0f]
    // Default: 1.0f
    algoCtrl.fine_detail_preservation = 1.0f;

    // Usage: Modifies the noise variance multiplier specifically at Level 2 (Quarter Resolution) of the Laplacian pyramid.
    // Impact: Higher values (e.g., 1.5) aggressively target massive, low-frequency color blotches and chunky noise.
    // Range: [0.0f to 2.0f]
    // Default: 1.0f
    algoCtrl.coarse_noise_reduction = 1.0f;

    // =========================================================
    // PERFORMANCE / ACCURACY
    // =========================================================

    // Usage: Determines the loop stride (`y += step`, `x += step`) inside the main Bayes processing loops.
    // Impact: Controls how many overlapping 3D patches are aggregated. 
    //         AccDraft (Stride 4) is fastest, skipping 75% of patches. AccHigh (Stride 1) is maximum quality.
    // Options: ProcAccuracy::AccDraft, ProcAccuracy::AccStandard, ProcAccuracy::AccHigh
    // Default: ProcAccuracy::AccStandard (Stride 2 - balances speed and artifact suppression)
    algoCtrl.accuracy = ProcAccuracy::AccStandard;
    
    return algoCtrl;
}