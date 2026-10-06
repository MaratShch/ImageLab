#ifndef __IMAGE_LAB_GUIDED_FILTER_CONTROLS__
#define __IMAGE_LAB_GUIDED_FILTER_CONTROLS__

#include <cstdint>
#include <algorithm>

// ============================================================================
// Processing mode (Render control: ListBox / popup).
// The numeric values are part of the contract between the Render controls and
// the algorithm: ListBox item index == static_cast<uint32_t>(FilterMode).
//   Item 0: "Entire Image"   Item 1: "Human Skin"   Item 2: "Background"
// ============================================================================
enum class FilterMode : uint32_t
{
    Entire_Image = 0,
    Skin,
    Background
};

// ============================================================================
// Control ranges and defaults (for the Render-control definitions and for the
// algorithm-side validation in Algorithm_Main). Values outside a range are
// clamped by the algorithm; non-finite floats are replaced by the default.
// ============================================================================
constexpr int32_t AlgoRadiusMin         = 0;        // 0 = identity
constexpr int32_t AlgoRadiusMax         = 255;
constexpr int32_t AlgoRadiusDef         = 8;

constexpr float   AlgoEpsilonMin        = 0.00001f;
constexpr float   AlgoEpsilonMax        = 1.0f;
constexpr float   AlgoEpsilonDef        = 0.01f;

constexpr float   AlgoSkinToleranceMin  = 0.25f;
constexpr float   AlgoSkinToleranceMax  = 2.0f;
constexpr float   AlgoSkinToleranceDef  = 1.0f;
constexpr float   AlgoSkinToleranceStep = 0.05f;

constexpr float   AlgoSkinSoftnessMin   = 0.0f;
constexpr float   AlgoSkinSoftnessMax   = 1.0f;
constexpr float   AlgoSkinSoftnessDef   = 0.25f;
constexpr float   AlgoSkinSoftnessStep  = 0.05f;

struct AlgoControls
{
    // ------------------------------------------------------------------------
    // Parameter: Radius
    // Description: The spatial size of the sliding window for the Guided Filter.
    //              Controls how "wide" the smoothing effect spreads.
    // Range:       0 to 255 (0 = no filtering)
    // Default:     8
    // Adobe UI:    Integer Slider (0 to 255)
    // ------------------------------------------------------------------------
    int32_t radius;

    // ------------------------------------------------------------------------
    // Parameter: Epsilon
    // Description: The regularization parameter. Controls edge preservation.
    //              Lower values preserve finer details (less smoothing).
    //              Higher values blur over stronger edges (more aggressive).
    // Range:       0.00001f to 1.0f
    // Default:     0.01f
    // Adobe UI:    Floating Point Slider (Logarithmic scaling recommended in UI)
    // ------------------------------------------------------------------------
    float epsilon;

    // ------------------------------------------------------------------------
    // Parameter: Filter Mode  (Render control: ListBox)
    // Description: Selects WHAT is processed.
    //              Entire_Image: the filtered image everywhere.
    //              Skin:         filtered image blended in by the skin confidence c.
    //              Background:   filtered image blended in by (1 - c).
    // Range:       0 (Entire_Image), 1 (Skin), 2 (Background)
    // Default:     0 (Entire_Image)
    // ------------------------------------------------------------------------
    FilterMode filterMode;

    // ------------------------------------------------------------------------
    // Parameter: Show Confidence Map  (Render control: Checkbox)
    // Description: Selects WHAT is shown. false: filtered image.
    //              true: confidence map of the selected region (Skin: c,
    //              Background: 1 - c) written as grey into the output image.
    //              Ignored for Entire_Image (filtered image is always output).
    // Default:     false
    // ------------------------------------------------------------------------
    bool showConfidenceMap;

    // ------------------------------------------------------------------------
    // Parameter: Skin Tolerance  (dimensionless)
    // Description: Scale of the skin ellipse in the BT.601 Cb/Cr plane. The
    //              ellipse is centred at (Cb, Cr) = (102, 153) with semi-axes
    //              25 * T (Cb) and 20 * T (Cr); T = 1 is the ellipse inscribed in
    //              the Chai & Ngan (1999) skin box Cb 77..127, Cr 133..173.
    //              Larger T accepts more colours as skin.
    // Range:       0.25 to 2.0, step 0.05
    // Default:     1.0
    // ------------------------------------------------------------------------
    float skinTolerance;

    // ------------------------------------------------------------------------
    // Parameter: Skin Softness  (dimensionless, fraction of the ellipse radius)
    // Description: Width S of the confidence transition around the ellipse
    //              border. With normalised distance d (d = 1 on the border):
    //              c = 1 for d <= 1 - S/2, c = 0 for d >= 1 + S/2, smoothstep
    //              in between (c = 0.5 at d = 1). S = 0 gives a binary mask.
    // Range:       0.0 to 1.0, step 0.05
    // Default:     0.25
    // ------------------------------------------------------------------------
    float skinSoftness;
};

inline AlgoControls getAlgoControlsDefault(void)
{
    AlgoControls ctrl;
    ctrl.radius            = AlgoRadiusDef;
    ctrl.epsilon           = AlgoEpsilonDef;
    ctrl.filterMode        = FilterMode::Entire_Image;
    ctrl.showConfidenceMap = false;
    ctrl.skinTolerance     = AlgoSkinToleranceDef;
    ctrl.skinSoftness      = AlgoSkinSoftnessDef;
    return ctrl;
}

#endif // __IMAGE_LAB_GUIDED_FILTER_CONTROLS__