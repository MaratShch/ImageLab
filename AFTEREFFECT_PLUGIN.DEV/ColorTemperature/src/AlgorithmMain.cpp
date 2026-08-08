#include <cstdint>
#include <cstring>
#include <cmath>

#include "Common.hpp"   
#include "AlgoControl.hpp"
#include "AlgorithmMain.hpp"

#include "CCTLut/CCTLut.hpp"
#include "LinearLut/LinearLut.hpp"


// ============================================================================
// Algorithm_Main - PHASE 2 (RENDER), Steps A..D of the pseudo-code, plus the
// tail of PHASE 1 (turn the already-computed SuperPixel into CCT/Duv).
//
// PRECONDITIONS (done by the caller in main / the plugin render entry):
//   - memHandler already holds the LINEARIZED, interleaved input RGB buffer
//     (filled by ingest) AND a same-size output RGB buffer; NO allocation
//     happens here.
//   - superPixel is the weighted neutral estimate from ingest_and_superpixel
//     (locus-gated), computed in double.
//   - params carries every user control (wbMode, temperatureK/FineK, tint,
//     confidenceMap, workingSpace, observer, catModel, adaptationDegree).
//
// INPUTS  : superPixel (measured neutral, RGB double)
//           memHandler (in: linear input RGB; out: linear output RGB; scratch)
//           sizeX, sizeY (pixels)
//           params (AlgoControls)
// OUTPUTS : the linear OUTPUT RGB buffer in memHandler, filled with the
//           corrected image (or the confidence-map passthrough); and the
//           reported CCT/Duv for the UI readout.
//
// STAGES this function is responsible for (see per-line notes below):
//   STEP A  establish the SOURCE white:
//             wbMode == Measure -> superpixel_to_cct(superPixel) -> (CCT,Duv)
//             wbMode == Manual  -> source from params: CCT = temperatureK +
//                                  temperatureFineK ; Duv = -tint / k
//   STEP B  establish the TARGET white (v1: fixed D65 reference).
//   STEP C  build ONE correction matrix M_wb once for the frame:
//             source/target (CCT,Duv) --getPlanckianUV--> uv --> XYZ (Y=1);
//             CAT (params.catModel, degree params.adaptationDegree) gives
//             M_adapt = M^-1 * diag(gains) * M ; then
//             M_wb = XYZtosRGB * M_adapt * sRGBtoXYZ   (RGB->corrected RGB).
//   STEP D  produce the OUTPUT linear RGB buffer from the INPUT linear RGB:
//             params.confidenceMap == 0 -> out[p] = M_wb * in[p]  (per pixel,
//                                          linear, unclamped; parallelizable)
//             params.confidenceMap != 0 -> the confidence map already lives in
//                                          the input linear buffer (ingest was
//                                          run in map mode) -> COPY in->out
//                                          unchanged (no correction applied).
//   (The observer used in Steps A/C MUST match the gate/observer used at
//    ingest and the one passed to ComputeCct - params.observer throughout.)
// ============================================================================
void Algorithm_Main
(
    AlgoCCT::CctHandle<double>& cctHdnl,    
    const SuperPixel<double>& superPixel, // Previously computed SuperPixel	
    const MemHandler&     memHandler, 	// contains linearized input and output RGB buffers, and buffers for intermediate processing/compute
    const int32_t         sizeX,	// horizontal linearized image size in pixels	
    const int32_t         sizeY,	// vertical linearized image size in pixels
    const AlgoControls&   params,	// Algorithm Control parameters
    CctDuv<double>&       cct_duv   // Computed CCT and Duv/Tint values    
) 
{
     	double cct_computed = 0.0;      // STEP A output: measured/target CCT [K]
       	double duv_computed = 0.0;      // STEP A output: measured/target Duv

        // ---- STEP A' : MEASURE the incoming CCT/Duv - ALWAYS (unconditional).
        // The scene's measured white point is a first-class OUTPUT of every
        // call (needed for the UI readout and the confidence workflow),
        // independent of wbMode and independent of whether a correction is
        // applied. So this runs before, and regardless of, Steps B..D.
        // IN : superPixel (from ingest), RGB->XYZ matrix, params.observer.
        // OUT: cct_duv  (the measured source white).
       	superpixel_to_cct (superPixel, cctHdnl, sRGBtoXYZ_f64, observer_CIE_1931,
                      		cct_computed, duv_computed);

        cct_duv.cct = cct_computed;    // measured CCT reported unconditionally
        cct_duv.duv = duv_computed;    // measured Duv reported unconditionally

        // ---- STEP A : establish the SOURCE white for the CORRECTION ---------
        // (distinct from the measurement above: in Manual mode the correction
        //  source comes from the sliders, not from the measured value.)
        //   wbMode == Measure -> source = (cct_computed, duv_computed) above
        //   wbMode == Manual  -> source CCT = params.temperatureK +
        //                        params.temperatureFineK ;
        //                        source Duv = -(params.tint)/k  (UI->CIE)
        //   NOTE: the Manual branch is NOT YET PRESENT.

        // ---- STEP B : target white (v1 fixed D65) : NOT YET PRESENT ---------

        // ---- STEP C : build M_wb from source/target via CAT : NOT YET PRESENT
        //     IN : (cct_computed,duv_computed) source, D65 target,
        //          params.catModel, params.adaptationDegree, params.observer,
        //          sRGBtoXYZ / XYZtosRGB
        //     OUT: M_wb[9]  (built ONCE here, not per pixel)

        // ---- STEP D : fill output linear RGB from input linear RGB ----------
        //     IN : memHandler input linear RGB, M_wb, params.confidenceMap
        //     OUT: memHandler output linear RGB
        //     confidenceMap==0 -> out = M_wb * in  (per pixel, parallel)
        //     confidenceMap!=0 -> copy in -> out   (map passthrough)

    return;
}

