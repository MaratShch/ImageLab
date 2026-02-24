#include "AlgoControls.hpp"

AlgoControls getAlgoControlsDefault(void)
{
    AlgoControls algoCtrl{};
    
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