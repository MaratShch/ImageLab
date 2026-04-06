#include <cmath>
#include "PaintAlgoMain.hpp"
#include "Common.hpp"
#include "FastAriphmetics.hpp"
#include "PaintCocircularity_graphs.hpp"

void PaintAlgorithmMain (const MemHandler& memHndl, const AlgoControls& algoCtrl)
{
    // ==================================================================================
    // 1. SETUP & CONSTANTS
    // ==================================================================================
    const A_long width  = memHndl.proc_width;
    const A_long height = memHndl.proc_height;
    
    const A_long scaleX = memHndl.origin_width  / memHndl.proc_width;
    const A_long scaleY = memHndl.origin_height / memHndl.proc_height;
    const A_long scale  = std::max(scaleX, scaleY);
    
    const float* pLuminanceIn = memHndl.Y_planar;
    float* pLuminanceOut = memHndl.Y_planar; 

    const A_long frameSize = width * height;
    
    const float coCircParam = FastCompute::PIdiv180 * algoCtrl.angular;
    const float coConeParam = FastCompute::PIdiv180 * algoCtrl.angle;

    const float coCirc = FastCompute::Cos(coCircParam);
    const float coCone = FastCompute::Cos(coConeParam);
    
    // ==================================================================================
    // 2. AVX2 CONTIGUOUS MATH PIPELINE (FUSED)
    // ==================================================================================
    // Fused Phase: Calculate Gradients AND Tensors simultaneously in cache
    compute_initial_tensors_fused_AVX2
    (
        memHndl.Y_planar, 
        memHndl.tensorA, 
        memHndl.tensorB, 
        memHndl.tensorC, 
        width, 
        height
    );   
    
    const float sigma = std::max(1.f, algoCtrl.sigma / static_cast<float>(scale));
    
    // Smooth the Tensors (Gaussian Blur)
    smooth_structure_tensors_AVX2
    (
        memHndl.tensorA, 
        memHndl.tensorB, 
        memHndl.tensorC, 
        sigma, 
        width, 
        height, 
        memHndl.tensorA_sm, 
        memHndl.tensorB_sm, 
        memHndl.tensorC_sm, 
        memHndl.tmpBlur
    );

    // 4. Diagonalize Tensors
    diagonalize_structure_tensors_AVX2
    (
        memHndl.tensorA_sm, 
        memHndl.tensorB_sm, 
        memHndl.tensorC_sm, 
        width, 
        height, 
        memHndl.Lambda1, 
        memHndl.Lambda2, 
        memHndl.EigVectX, 
        memHndl.EigVectY
    );

    // ==================================================================================
    // 3. GRAPH CONSTRUCTION (FLAT EDGE-LIST AVX2)
    // ==================================================================================
    const A_long p_radius_tmp = static_cast<A_long>(std::ceil(algoCtrl.sigma * 1.5f));
    const A_long p_radius = std::max(1, p_radius_tmp >> 1);
    
    // We stream the surviving connections directly into our pre-allocated Arena arrays.
    // Zero heap allocations. Zero SparseMatrix overhead. 
    const A_long nonZeros = bw_image2cocircularity_graph_AVX2_flat
    (
        memHndl.EigVectX, 
        memHndl.EigVectY, 
        memHndl.pI_arena, 
        memHndl.pJ_arena, 
        memHndl.max_edges,
        width, 
        height, 
        coCirc, 
        coCone, 
        p_radius
    );

    // ==================================================================================
    // 4. PHASE 4: EXECUTE MORPHOLOGY (ZERO-COPY IN-PLACE)
    // ==================================================================================
    const A_long iter = std::max(1, algoCtrl.iter >> 1);
    
    switch (algoCtrl.bias)
    {
        case StrokeBias::DarkBias_Open:
            morpho_open
            (
                memHndl.Y_planar, memHndl.pI_arena, memHndl.pJ_arena, 
                algoCtrl.iter, nonZeros, width, height, memHndl
            );
        break;

        case StrokeBias::LightBias_Close:
            morpho_close
            (
                memHndl.Y_planar, memHndl.pI_arena, memHndl.pJ_arena, 
                algoCtrl.iter, nonZeros, width, height, memHndl
            );
        break;

        case StrokeBias::Balanced_ASF:
            morpho_asf
            (
                memHndl.Y_planar, memHndl.pI_arena, memHndl.pJ_arena, 
                algoCtrl.iter, nonZeros, width, height, memHndl
            );
        break;
    }
    
    return;    
}