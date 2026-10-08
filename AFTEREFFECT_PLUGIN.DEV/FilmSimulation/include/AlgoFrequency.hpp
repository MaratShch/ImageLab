#pragma once

// ---------------------------------------------------------------------------
//  AlgoFrequency.hpp
//
//  Frequency-domain filtering for the film engine, on the owner's FFT library
//  (namespace FourierTransform, optimised copy delivered 2026-10-06). Not a
//  pipeline stage: infrastructure used by every stage whose reference
//  (film_sim.py) multiplies a half-spectrum transfer function:
//
//      3b  veiling flare        multi_gaussian((1500, 6000, 20000) um)
//      5   halation             multi_gaussian(hal.lobes(c)), ring included
//      6   emulsion MTF         mtf(f50_c, adjacency, adjacency_um, spec, c)
//      9   DIR coupler lateral  gaussian(radius_um), gaussian(edge_um)
//      10  scan MTF + misreg    mtf(scan_f50) * shift(dy, dx)
//      13  duplication          mtf(dupe.mtf_f50)
//      14b reseau reconstruct   gaussian(reconstruction_pitches * pitch_um)
//
//  WHY THIS REPLACES THE SEPARABLE GAUSSIANS (owner decision 2026-10-06)
//
//  The C++ engines had no FFT, so each of those transfers was approximated by
//  sampled, truncated, separable Gaussian kernels: the measured MTF law
//  1/(1+(f/f50)^q) by a two- or three-lobe fit (worst error 0.043, then 0.016
//  with the G5 tables), every lobe below 0.25 px dropped, the flare and wide
//  halation lobes capped at 64 taps. With the owner's FFT the engines apply
//  EXACTLY the transfer film_sim applies -- same formula, same frequency grid,
//  same circular boundary -- so the three implementations run one flow on one
//  set of database metrics instead of a reference and two approximations.
//
//  THE GRID (film_sim.FreqGrid, isotropic -- anisotropy is grain only)
//
//      fy[r] = numpy.fft.fftfreq(H)[r]   cycles/pixel,  r = 0 .. H-1
//      fx[c] = numpy.fft.rfftfreq(W)[c]  cycles/pixel,  c = 0 .. W/2
//      f_mm^2 = (fy * pxPerMm)^2 + (fx * pxPerMm)^2
//
//  Every transfer below is a function of f_mm^2 (and, for the shift, of fy, fx).
//
//  THE TRANSFER DESCRIPTOR
//
//      T(f) = SumA(f) * SumB(f) * Law(f) * Shift(fy, fx)
//
//      SumA, SumB   sum_i w_i exp(-a_i f_mm^2)       (empty sum == 1)
//                   a Gaussian of sigma s um has a = 2 pi^2 (s/1000)^2;
//                   the legacy Gaussian MTF law exp(-ln2 (f/f50)^2) has
//                   a = ln2 / f50^2; a = 0 is a constant term.
//      Law          1 / (1 + (f/f50)^q)   (the measured emulsion rolloff)
//      Shift        exp(-2 pi j (fy dy + fx dx))     (sub-pixel translation)
//
//  Every Gaussian term is SEPARABLE, exp(-a fy^2) exp(-a fx^2), so a transfer
//  costs H + W/2 exponentials to set up and a few multiplies per bin to apply;
//  it is evaluated on the fly inside the inverse transform's column gather and
//  never stored as a plane.
//
//  MEMORY: ALGO_FREQ_SPECTRA half spectra (H x specPitch complex AlgoType each),
//  the FFT work area, the plan tables and the per-transfer 1D tables, all carved
//  from the arena by alloc_memory_buffers. Executing allocates nothing.
//
//  Raw pointers, explicit geometry, no mutable state outside the arena.
// ---------------------------------------------------------------------------

#include "Common.hpp"
#include "CompileTimeUtils.hpp"
#include "AlgoTypes.hpp"

#include "fft_real2d.hpp"     // owner's FFT library: PlanReal2D, Real2DForward/Inverse

#include <cstdint>
#include <cstddef>


//: Most Gaussian terms in one sum. Halation: three lobes plus the ring = 4;
//: flare: 3; the adjacency lift: constant + 2.
constexpr int32_t ALGO_FREQ_MAX_TERMS = 6;

//: Adjacency band-pass lobe scales, as multiples of MTFSpec.adjacency_um
//: (film_sim FreqGrid.mtf: gaussian(adjacency_um * 0.4) - gaussian(adjacency_um * 2.0)).
//: The shape of the effect rather than free parameters: they place the overshoot
//: peak at the diffusion length itself.
constexpr double ALGO_FREQ_ADJACENCY_INNER = 0.4;
constexpr double ALGO_FREQ_ADJACENCY_OUTER = 2.0;

//: Half spectra held at once. One serves every single-plane filter; three let
//: stage 9 combine the records' spectra (the DIR coupler's long-range term
//: reads the mean of all three) without a transform per intermediate.
constexpr int32_t ALGO_FREQ_SPECTRA = 3;

//: Bytes reserved per FFT lane vector when sizing the work area. 64 covers the
//: widest lane (LaneAvx2x2, eight complex floats); the scalar lane uses 16.
constexpr std::size_t ALGO_FREQ_LANE_BYTES = 64;


// ---------------------------------------------------------------------------
//  Arena-resident state. A plain aggregate held inside MemHandler.
// ---------------------------------------------------------------------------
struct AlgoFreqState
{
    FourierTransform::PlanReal2D<AlgoType> plan;   // tables live in the arena
    int32_t   sizeX;
    int32_t   sizeY;
    int32_t   specPitch;        // complex values per spectrum row
    AlgoType* spec[ALGO_FREQ_SPECTRA];  // sizeY x specPitch complex each, interleaved
    void*     work;             // FFT work area
    // per-frame 1D tables (written by AlgoFreqBeginFrame and per transfer)
    AlgoType* fy;               // sizeY        cycles/pixel
    AlgoType* fx;               // specPitch    cycles/pixel (0 beyond W/2)
    AlgoType* fy2mm;            // sizeY        (fy * pxPerMm)^2
    AlgoType* fx2mmD;           // 2*specPitch  (fx * pxPerMm)^2, duplicated pairs
    AlgoType* gyA;              // ALGO_FREQ_MAX_TERMS x sizeY
    AlgoType* gxA;              // ALGO_FREQ_MAX_TERMS x 2*specPitch (duplicated pairs)
    AlgoType* gyB;
    AlgoType* gxB;
    AlgoType* shY;              // 2*sizeY      complex exp(-2 pi j fy dy)
    AlgoType* shX;              // 2*specPitch  complex exp(-2 pi j fx dx)
};


// ---------------------------------------------------------------------------
//  Sizing and construction (allocator only).
// ---------------------------------------------------------------------------
std::size_t AlgoFreqPlanBytes  (int32_t sizeX, int32_t sizeY) noexcept;
std::size_t AlgoFreqSpecBytes  (int32_t sizeX, int32_t sizeY) noexcept;
std::size_t AlgoFreqWorkBytes  (int32_t sizeX, int32_t sizeY) noexcept;
std::size_t AlgoFreqTableBytes (int32_t sizeX, int32_t sizeY) noexcept;

//: Build the plan into planMem and attach the other three areas. Returns false
//: on a size the plan cannot carry.
//: specMem holds ALGO_FREQ_SPECTRA spectra (AlgoFreqSpecBytes covers all of them).
bool AlgoFreqInit (AlgoFreqState& st, int32_t sizeX, int32_t sizeY,
                   void* planMem, void* specMem, void* workMem, void* tableMem) noexcept;


// ---------------------------------------------------------------------------
//  Transfer descriptor.
// ---------------------------------------------------------------------------
struct AlgoFreqGaussSum
{
    int32_t      n;                          // 0 == the factor is 1
    HighPrecType a[ALGO_FREQ_MAX_TERMS];     // mm^2
    HighPrecType w[ALGO_FREQ_MAX_TERMS];
};

struct AlgoFreqTransfer
{
    AlgoFreqGaussSum sumA;
    AlgoFreqGaussSum sumB;
    int32_t          hasLaw;                 // 1 / (1 + (f/f50)^q)
    HighPrecType     lawQ;
    HighPrecType     lawF50;                 // cycles/mm
    int32_t          hasShift;               // exp(-2 pi j (fy dy + fx dx))
    HighPrecType     shiftDy;                // pixels
    HighPrecType     shiftDx;
};

inline void AlgoFreqTransferClear (AlgoFreqTransfer& t) noexcept
{
    t.sumA.n = 0;
    t.sumB.n = 0;
    t.hasLaw = 0;
    t.lawQ = 0.0;
    t.lawF50 = 0.0;
    t.hasShift = 0;
    t.shiftDy = 0.0;
    t.shiftDx = 0.0;
    for (int32_t i = 0; i < ALGO_FREQ_MAX_TERMS; i++)
    {
        t.sumA.a[i] = 0.0; t.sumA.w[i] = 0.0;
        t.sumB.a[i] = 0.0; t.sumB.w[i] = 0.0;
    }
}

//: film_sim FreqGrid.gaussian: exp(-2 pi^2 (sigma_um/1000)^2 f_mm^2) -> a.
inline HighPrecType AlgoFreqGaussianA (const HighPrecType sigmaUm) noexcept
{
    const HighPrecType sMm = sigmaUm / 1000.0;
    return 2.0 * 9.869604401089358 * sMm * sMm;          // 2 pi^2 s^2
}

//: Append one term to a sum (ignored when the sum is full).
inline void AlgoFreqSumAdd (AlgoFreqGaussSum& s, const HighPrecType a, const HighPrecType w) noexcept
{
    if (s.n < ALGO_FREQ_MAX_TERMS)
    {
        s.a[s.n] = a;
        s.w[s.n] = w;
        s.n++;
    }
}

//: film_sim FreqGrid.gaussian(sigma_um), as sum A.
inline void AlgoFreqSetGaussian (AlgoFreqTransfer& t, const HighPrecType sigmaUm) noexcept
{
    AlgoFreqTransferClear(t);
    AlgoFreqSumAdd(t.sumA, AlgoFreqGaussianA(sigmaUm), 1.0);
}

//: film_sim FreqGrid.multi_gaussian(radii_um, weights): sum_i (w_i / sum w) G(r_i).
//: Signed weights (the halation ring) normalise by the signed sum, as film_sim does.
inline void AlgoFreqSetMultiGaussian (AlgoFreqTransfer& t, const HighPrecType* radiiUm,
                                      const HighPrecType* weights, const int32_t n) noexcept
{
    AlgoFreqTransferClear(t);
    HighPrecType wsum = 0.0;
    for (int32_t i = 0; i < n; i++) wsum += weights[i];
    for (int32_t i = 0; i < n; i++)
        AlgoFreqSumAdd(t.sumA, AlgoFreqGaussianA(radiiUm[i]), weights[i] / wsum);
}

//: film_sim FreqGrid.mtf(f50, adjacency, adjacency_um, spec, channel).
//:   measured (q > 0): law 1/(1+(f/f50)^q); otherwise exp(-ln2 (f/f50)^2);
//:   f50 <= 0: unity. adjacency > 0 multiplies by
//:   1 + adjacency (G(0.4 adjacency_um) - G(2.0 adjacency_um)).
//: The scanner and duplication stages pass measured = false (no MTFSpec).
inline void AlgoFreqSetMtf (AlgoFreqTransfer& t, const HighPrecType f50, const bool measured,
                            const HighPrecType q, const HighPrecType adjacency,
                            const HighPrecType adjacencyUm) noexcept
{
    AlgoFreqTransferClear(t);
    if (f50 > 0.0)
    {
        if (measured && (q > 0.0))
        {
            t.hasLaw = 1;
            t.lawQ = q;
            t.lawF50 = f50;
        }
        else
        {
            AlgoFreqSumAdd(t.sumA, 0.69314718055994530942 / (f50 * f50), 1.0);
        }
    }
    if (adjacency > 0.0)
    {
        AlgoFreqSumAdd(t.sumB, 0.0, 1.0);
        AlgoFreqSumAdd(t.sumB, AlgoFreqGaussianA(adjacencyUm * ALGO_FREQ_ADJACENCY_INNER),  adjacency);
        AlgoFreqSumAdd(t.sumB, AlgoFreqGaussianA(adjacencyUm * ALGO_FREQ_ADJACENCY_OUTER), -adjacency);
    }
}

//: Multiply in film_sim FreqGrid.shift(dy, dx).
inline void AlgoFreqAddShift (AlgoFreqTransfer& t, const HighPrecType dyPx, const HighPrecType dxPx) noexcept
{
    t.hasShift = 1;
    t.shiftDy = dyPx;
    t.shiftDx = dxPx;
}

//: True when the transfer is exactly 1 everywhere (nothing to filter).
inline bool AlgoFreqIsIdentity (const AlgoFreqTransfer& t) noexcept
{
    return (0 == t.sumA.n) && (0 == t.sumB.n) && (0 == t.hasLaw) && (0 == t.hasShift);
}


// ---------------------------------------------------------------------------
//  Per frame: the f_mm^2 grid for this frame's pxPerMm.
// ---------------------------------------------------------------------------
void AlgoFreqBeginFrame (const AlgoFreqState& st, AlgoType pxPerMm) noexcept;


// ---------------------------------------------------------------------------
//  dst = irfft2(rfft2(src) * T)   -- film_sim.apply_transfer.
//
//  src and dst may be the same plane. Rows are 'pitch' elements apart; only the
//  sizeX active elements of each row are read and written. Requires
//  AlgoFreqBeginFrame for the current pxPerMm.
// ---------------------------------------------------------------------------
void AlgoFreqFilterPlane (const AlgoFreqState& st, const AlgoType* src, AlgoType* dst,
                          int32_t pitch, const AlgoFreqTransfer& t) noexcept;


// ---------------------------------------------------------------------------
//  Spectrum slots, for stages that combine planes in the frequency domain.
//
//  AlgoFreqForward   spec[slot] = rfft2(src)
//  AlgoFreqInverse   dst = irfft2(spec[slot] * T); consumes spec[slot]
//  AlgoFreqMeanMix3  for c = 0, 1, 2:
//                        spec[c] = keep * spec[c] + mix * T * (spec[0] + spec[1] + spec[2]) / 3
//                    -- the spectrum of keep * x_c + mix * (T conv mean(x)), i.e.
//                    film_sim's x_c += s * (x_c - blur(mean x)) with keep = 1 + s,
//                    mix = -s, evaluated without transforming the mean.
//  slot is 0 .. ALGO_FREQ_SPECTRA-1. AlgoFreqFilterPlane uses slot 0.
// ---------------------------------------------------------------------------
void AlgoFreqForward  (const AlgoFreqState& st, const AlgoType* src, int32_t pitch, int32_t slot) noexcept;
void AlgoFreqInverse  (const AlgoFreqState& st, int32_t slot, const AlgoFreqTransfer& t,
                       AlgoType* dst, int32_t pitch) noexcept;
void AlgoFreqMeanMix3 (const AlgoFreqState& st, HighPrecType keep, HighPrecType mix,
                       const AlgoFreqTransfer& t) noexcept;
