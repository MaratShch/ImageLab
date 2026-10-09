#include <cmath>
#include <cstring>
#include "AlgorithmMain.hpp"
#include "AlgorithmProces.hpp"


// Guided filter of one plane over the whole image: out_plane = q = mean_a * I + mean_b.
static void Process_Guided_Filter_Plane
(
    const float* in_plane,
    float* out_plane,
    const MemHandler& memHandler,
    const int32_t width,
    const int32_t height,
    const AlgoControls& ctrl
)
{
    const int32_t radius = ctrl.radius;

    // Epsilon is defined for RGB normalised to [0..1] (He et al. convention). The planes
    // hold 0..255-scale orthonormal YUV; the orthonormal transform preserves noise variance,
    // so one scale 255^2 = 65025 is correct for Y, U and V. This is the ONLY place epsilon
    // is scaled (defect D1: it was also multiplied by 195000 in math_Compute_Coefficients_AB).
    const float scaled_epsilon = ctrl.epsilon * 65025.0f;

    // 1. Calculate mean_I: boxBlur(I)
    math_BoxBlur_O1(in_plane, memHandler.mean_I, memHandler.temp_blur, width, height, radius);

    // 2. Calculate I^2 temporarily into mean_II buffer
    math_Square_Elements(in_plane, memHandler.mean_II, width, height);

    // 3. Calculate mean_II: boxBlur(I^2). We blur mean_II in-place.
    math_BoxBlur_O1(memHandler.mean_II, memHandler.mean_II, memHandler.temp_blur, width, height, radius);

    // 4. Calculate coefficients 'a' and 'b' 
    // scaled_epsilon is passed as-is; the callee must not rescale it.
    math_Compute_Coefficients_AB(memHandler.mean_I, memHandler.mean_II, 
                                 memHandler.coef_a, memHandler.coef_b, 
                                 width, height, scaled_epsilon);

    // 5. Blur coefficient 'a' in-place: mean_a = boxBlur(a)
    math_BoxBlur_O1(memHandler.coef_a, memHandler.coef_a, memHandler.temp_blur, width, height, radius);

    // 6. Blur coefficient 'b' in-place: mean_b = boxBlur(b)
    math_BoxBlur_O1(memHandler.coef_b, memHandler.coef_b, memHandler.temp_blur, width, height, radius);

    // 7. Apply the final linear coefficients over the whole plane (region selection happens later)
    math_Apply_Filter_And_Mask(in_plane, memHandler.coef_a, memHandler.coef_b, 
                               nullptr, out_plane, 
                               width, height, FilterMode::Entire_Image);
}


void Algorithm_Main
(
    const MemHandler& memHandler,
    const int32_t sizeX, 
    const int32_t sizeY, 
    const AlgoControls& algoCtrl
)
{
    // 1. Guided filter of the Y, U and V orthonormal planes over the whole image (q -> out planes).
    Process_Guided_Filter_Plane(memHandler.in_Y, memHandler.out_Y, memHandler, sizeX, sizeY, algoCtrl);
    Process_Guided_Filter_Plane(memHandler.in_U, memHandler.out_U, memHandler, sizeX, sizeY, algoCtrl);
    Process_Guided_Filter_Plane(memHandler.in_V, memHandler.out_V, memHandler, sizeX, sizeY, algoCtrl);

    // Entire_Image: the filtered image is the output; no classification is computed
    // and the confidence-map checkbox has no effect.
    if (FilterMode::Entire_Image == algoCtrl.filterMode)
        return;

    // 2. Skin classification on the DENOISED colour (q planes): continuous confidence c in [0..1].
    //    It is stored in the arena's mask plane (owned by the algorithm arena, never host memory;
    //    the pointer is const in MemHandler only because the plane is an input of the blend stage).
    float* conf = const_cast<float*>(memHandler.in_Mask);
    math_Skin_Confidence(memHandler.out_Y, memHandler.out_U, memHandler.out_V, conf,
                         sizeX, sizeY, algoCtrl.skinTolerance, algoCtrl.skinSoftness);

    if (algoCtrl.showConfidenceMap)
    {
        // 3a. Output = selection view: original input pixels where the filter acts (blend weight > 0),
        //     black where the pixel is excluded from filtering (the region blend is not needed).
        math_Write_Confidence_Map(conf, memHandler.in_Y, memHandler.in_U, memHandler.in_V,
                                  memHandler.out_Y, memHandler.out_U, memHandler.out_V,
                                  sizeX, sizeY, algoCtrl.filterMode);
    }
    else
    {
        // 3b. Output = filtered image restricted to the selected region by the confidence.
        math_Blend_Region(memHandler.in_Y, conf, memHandler.out_Y, sizeX, sizeY, algoCtrl.filterMode);
        math_Blend_Region(memHandler.in_U, conf, memHandler.out_U, sizeX, sizeY, algoCtrl.filterMode);
        math_Blend_Region(memHandler.in_V, conf, memHandler.out_V, sizeX, sizeY, algoCtrl.filterMode);
    }

    return;
}