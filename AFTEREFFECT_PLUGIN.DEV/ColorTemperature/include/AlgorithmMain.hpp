#ifndef __IMAGELAB2_ALGORITHM_MAIN_HPP__
#define __IMAGELAB2_ALGORITHM_MAIN_HPP__

// Declarations ONLY. AlgoControls and MemHandler are YOUR types and are
// defined in your own headers - this file must never redefine them.

#include <cstdint>
#include "AlgoControl.hpp"          // YOUR AlgoControls
#include "AlgoMemHandler.hpp"       // YOUR MemHandler
#include "cct_interface.hpp"        // AlgoCCT::CctHandle
#include "super_pixel.hpp"          // SuperPixel<>, CctDuv<>
#include "AlgoPrFormatIngest.hpp"   // AlgoPrIngest::LocusGate (scalar path)
#include "AlgoReference.hpp"        // AlgoWB::WbReference
#ifdef __AVX2__
  #include "AlgoMeasureAVX2.hpp"    // AlgoPrIngest::avx2::MeasureCtxAVX2
#endif

// -----------------------------------------------------------------------------
// AlgoMeasureSetup - everything the MEASUREMENT depends on, built from the
// controls observer / workingSpace / gateDuvFull / gateDuvZero. Immutable
// while rendering; lives in the host's sequence data.
//
// ALIGNMENT (C++14): MeasureCtxAVX2 holds __m256 members and 64-byte aligned
// tables. Stack and static instances are aligned correctly. If the host
// allocates this on the HEAP (e.g. inside sequence data), plain operator new
// in C++14 does NOT honour over-alignment: use an aligned allocator
// (_mm_malloc / _aligned_malloc / posix_memalign) or the AVX2 loads may fault.
// -----------------------------------------------------------------------------
struct AlgoMeasureSetup
{
    AlgoPrIngest::LocusGate gate;          // scalar ingest_and_superpixel()
#ifdef __AVX2__
    AlgoPrIngest::avx2::MeasureCtxAVX2 ctx;  // AVX2 measure_avx2() + refinement
#endif
    int32_t observer     = -1;             // what it was built from
    int32_t workingSpace = -1;
    float   gateFull     = -1.0f;
    float   gateZero     = -1.0f;
    bool    valid        = false;
};

// Build (or rebuild) the measurement setup from the controls.
bool Algorithm_BuildMeasureSetup (const AlgoCCT::CctHandle<double>& cctHdnl,
                                  const AlgoControls& params,
                                  AlgoMeasureSetup& setup) noexcept;

// True when observer / working space / gate band changed since the setup was
// built -> rebuild it, and re-Compute the reference.
bool Algorithm_SetupIsStale (const AlgoMeasureSetup& setup,
                             const AlgoControls& params) noexcept;

// True when the stored reference was measured under a different observer,
// working space or gate band. Algorithm_Main ignores a stale reference and
// falls back to the frame's own measurement.
bool Algorithm_ReferenceIsStale (const AlgoWB::WbReference& ref,
                                 const AlgoControls& params) noexcept;

#ifdef __AVX2__
// What the Compute button (measureNow) runs. linearImage must be the NORMAL
// linear image (ingest with confidenceMap OFF). Returns false -> keep the
// previous reference (confidence too low and not overridden, or no white).
bool Algorithm_ComputeReference (AlgoCCT::CctHandle<double>& cctHdnl,
                                 const AlgoMeasureSetup&     setup,
                                 const SuperPixel<double>&   superPixel,
                                 double                      keptFraction,
                                 const float*                linearImage,
                                 int32_t                     sizeX,
                                 int32_t                     sizeY,
                                 const AlgoControls&         params,
                                 AlgoWB::WbReference&        ref) noexcept;
#endif

// Every render call. 'reference' is the stored reference from Compute; pass
// nullptr to use the frame's own measurement.
void Algorithm_Main (AlgoCCT::CctHandle<double>& cctHdnl,
                     const SuperPixel<double>&   superPixel,
                     const MemHandler&           memHandler,
                     const int32_t               sizeX,
                     const int32_t               sizeY,
                     const AlgoControls&         params,
                     CctDuv<double>&             cct_duv,
                     const AlgoWB::WbReference*  reference = nullptr) noexcept;

#endif // __IMAGELAB2_ALGORITHM_MAIN_HPP__
