#include <cmath>
#include "PaintAlgoMain.hpp"
#include "Common.hpp"
#include "FastAriphmetics.hpp"
#include "PaintCocircularity_graphs.hpp"

void PaintAlgorithmMain (const MemHandler& memHndl, const AlgoControls& algoCtrl, A_long width, A_long height)
{
    // ==================================================================================
    // 1. SETUP & CONSTANTS
    // ==================================================================================
    const float* pLuminanceIn = memHndl.Y_planar;
    float* pLuminanceOut = memHndl.Y_planar; // We process in-place to save memory

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
    
    // Smooth the Tensors (Gaussian Blur)
    smooth_structure_tensors_AVX2
    (
        memHndl.tensorA, 
        memHndl.tensorB, 
        memHndl.tensorC, 
        algoCtrl.sigma, 
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
    const A_long p_radius = 7; 

    // We stream the surviving connections directly into our pre-allocated Arena arrays.
    // Zero heap allocations. Zero SparseMatrix overhead. 
    const A_long nonZeros = bw_image2cocircularity_graph_AVX2_flat
    (
        memHndl.EigVectX, 
        memHndl.EigVectY, 
        memHndl.pI_arena, 
        memHndl.pJ_arena, 
        memHndl.pLogW_arena, 
        memHndl.max_edges,
        width, 
        height, 
        coCirc, 
        coCone, 
        p_radius
    );

    // ==================================================================================
    // 4. PHASE 4: EXECUTE MORPHOLOGY (FUNCTION-LEVEL BRANCHING)
    // ==================================================================================
    switch (algoCtrl.bias)
    {
        case StrokeBias::DarkBias_Open:
            morpho_open
            (
                memHndl.Y_planar, pLuminanceOut, memHndl.pI_arena, memHndl.pJ_arena, 
                algoCtrl.iter, nonZeros, width, height, memHndl
            );
        break;

        case StrokeBias::LightBias_Close:
            morpho_close
            (
                memHndl.Y_planar, pLuminanceOut, memHndl.pI_arena, memHndl.pJ_arena, 
                algoCtrl.iter, nonZeros, width, height, memHndl
            );
        break;

        case StrokeBias::Balanced_ASF:
            morpho_asf
            (
                memHndl.Y_planar, pLuminanceOut, memHndl.pI_arena, memHndl.pJ_arena, 
                algoCtrl.iter, nonZeros, width, height, memHndl
            );
        break;
    }
    
    return;    
}