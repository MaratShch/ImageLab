#ifndef __IMAGELAB2_ALGO_REFERENCE_HPP__
#define __IMAGELAB2_ALGO_REFERENCE_HPP__

// =============================================================================
// AlgoReference.hpp — the WHITE-BALANCE REFERENCE and how it is established.
//
// This header exists because of one measured fact: applying M_wb once does NOT
// land the frame exactly on the target. M_wb maps the measured SUPER-PIXEL onto
// the target exactly (verified to 1.4e-07), but re-measuring the corrected
// frame is a SECOND, INDEPENDENT estimate — every pixel moved, so the locus
// gate re-weights a different population. On Sunset one pass leaves
// 6037 K against a 6504 K target: a ~12 mired residual, which sits between an
// 81 and an 81A filter, i.e. visible to a colorist.
//
// The fix costs nothing per frame. Because the reference is established ONCE,
// by an explicit user action (Compute) on a parked frame, we can afford to
// iterate there and solve for the EFFECTIVE source white — the source value
// that makes a SINGLE application land on the target. Every render frame then
// does exactly one apply, as before.
//
//     compute_reference()   <- runs at Compute time, 2-4 measure/apply rounds
//     WbReference           <- what it stores; feeds every later frame
//
// Also here: the confidence classification that decides whether a measurement
// may be trusted at all, and the clipping predictor that warns when a
// correction is so extreme that highlights will blow.
//
// Single-threaded, portable C++14; the AVX2 fast paths are used when the
// translation unit is built with AVX2 enabled.
// =============================================================================

#include <cstdint>
#include <cmath>
#include <vector>
#include "Common.hpp"
#include "AlgoPrFormatIngest.hpp"     // LocusGate, SuperPixel
#include "AlgoWhiteBalance.hpp"       // build_wb_matrix
#include "AlgoApplyWB.hpp"            // apply_white_balance
#include "cct_interface.hpp"


#ifdef __AVX2__
  #include "AlgoMeasureAVX2.hpp"
#endif

namespace AlgoWB
{
    // ---------------------------------------------------------------- confidence
    enum eConfidence : int32_t
    {
        conf_None = 0,   // refuse: too little neutral evidence to trust
        conf_Low  = 1,   // usable but caution: sparse, or far off the locus
        conf_High = 2    // trustworthy
    };

    // PLACEHOLDER THRESHOLDS — calibrate against the real image batch before
    // shipping. The right procedure: record keptFraction for every test image,
    // find the lowest value that still produced a correct answer, and set
    // keptLow below it with margin. Reference points measured so far:
    // Sunset keptFraction = 0.776 (correct answer, 2514.38 K).
    //
    // NOTE ON WHAT THIS MEASURES: at 4 MP even 0.4 % kept is ~16000 pixels, so
    // this is NOT a sample-size floor — it is a REPRESENTATIVENESS signal. A
    // low fraction means the gate barely found anything neutral, so the
    // estimate may rest on one coloured object that happened to sit near the
    // locus. That is why the response is "refuse and explain", not "widen the
    // error bars".
    // USER-CONTROLLABLE. These are exposed in the Effect Control Panel: the
    // customer decides what counts as trustworthy, and may override a refusal
    // outright. Defaults below are STARTING POINTS to be calibrated against
    // the real image batch, not constants.
    //
    // WHAT THIS MEASURES: not sample size - at 4 MP even 0.4 % kept is ~16000
    // pixels. It is REPRESENTATIVENESS. A low fraction means the gate barely
    // found anything neutral, so the estimate may rest on one coloured object
    // that happened to sit near the locus.
    struct ConfidenceThresholds
    {
        double keptHigh;            // >= this (and on-locus) -> HIGH
        double keptLow;             // >= this -> LOW ; below -> NONE
        double duvCaution;          // |Duv| above this caps the rating at LOW
        bool   acceptBelowThreshold;// user override: accept a NONE measurement
                                    // anyway instead of refusing
        ConfidenceThresholds()
            : keptHigh(0.15), keptLow(0.02), duvCaution(0.010)
            , acceptBelowThreshold(false) {}
    };

    // Factory from UI values (percentages) - keeps the panel's units at the
    // boundary and guarantees a sane ordering whatever the user types.
    inline ConfidenceThresholds make_thresholds (double highPct, double lowPct,
                                                 double duvCaution,
                                                 bool acceptBelow) noexcept
    {
        ConfidenceThresholds t;
        if (!(highPct >= 0.0))   highPct = 0.0;
        if (highPct > 100.0)     highPct = 100.0;
        if (!(lowPct  >= 0.0))   lowPct  = 0.0;
        if (lowPct  > 100.0)     lowPct  = 100.0;
        if (lowPct > highPct)    lowPct  = highPct;      // LOW can never exceed HIGH
        if (!(duvCaution > 0.0)) duvCaution = 0.001;
        if (duvCaution > 0.08)   duvCaution = 0.08;
        t.keptHigh            = highPct * 0.01;
        t.keptLow             = lowPct  * 0.01;
        t.duvCaution          = duvCaution;
        t.acceptBelowThreshold= acceptBelow;
        return t;
    }

    // The panel's "Reset" for this group.
    inline ConfidenceThresholds default_thresholds (void) noexcept
    { return ConfidenceThresholds(); }

    inline eConfidence classify_confidence (const double keptFraction,
                                            const double duv,
                                            const ConfidenceThresholds& t
                                                = ConfidenceThresholds()) noexcept
    {
        if (!(keptFraction >= t.keptLow))            return conf_None;
        if (std::fabs(duv) > t.duvCaution)           return conf_Low;
        if (keptFraction >= t.keptHigh)              return conf_High;
        return conf_Low;
    }

    inline const char* confidence_text (const eConfidence c) noexcept
    {
        switch (c) {
            case conf_High: return "HIGH";
            case conf_Low:  return "LOW";
            case conf_None:
            default:        return "NO MEASUREMENT";
        }
    }

    // --------------------------------------------------- locus gate band
    // ALSO USER-CONTROLLABLE, and far more consequential than the thresholds
    // above: the thresholds only change how a measurement is LABELLED, the
    // gate band changes WHICH PIXELS ARE COUNTED - so it moves keptFraction
    // AND the measured CCT/Duv themselves. Widen it and Sunset stops reading
    // 2514.38 K.
    //
    // duvFull : |Duv| up to here counts at FULL weight
    // duvZero : weight tapers to zero here; beyond it the pixel is excluded
    // Changing either requires REBUILDING the LocusGate and the AVX2 measure
    // context, and invalidates any stored reference.
    inline void sanitize_gate_band (double& duvFull, double& duvZero) noexcept
    {
        if (!(duvFull > 0.0))  duvFull = 0.001;
        if (duvFull > 0.100)   duvFull = 0.100;
        if (!(duvZero > 0.0))  duvZero = duvFull * 2.0;
        if (duvZero > 0.200)   duvZero = 0.200;
        // the taper must have somewhere to happen
        if (duvZero <= duvFull) duvZero = duvFull * 1.5;
    }

    // ------------------------------------------------------- clipping predictor
    // Response of the matrix to neutral white: the per-channel gain a neutral
    // pixel receives. If the largest exceeds 1, neutrals above 1/max will clip.
    inline double max_channel_response (const double M[9]) noexcept
    {
        double mx = 0.0;
        for (int r = 0; r < 3; ++r) {
            const double s = M[r*3+0] + M[r*3+1] + M[r*3+2];
            if (s > mx) mx = s;
        }
        return mx;
    }

    // True when the correction is extreme enough that the UI should say so.
    // Sunset -> D65 at full strength gives max response 2.84, i.e. neutrals
    // above 0.35 clip. That is the honest answer to "neutralise a 2514 K
    // sunset" - warn, do not silently soften the default.
    inline bool clipping_warning (const double M[9],
                                  const double threshold = 1.5) noexcept
    {
        return max_channel_response(M) > threshold;
    }

    // Neutral level above which the most-amplified channel clips (1.0 = none).
    inline double clipping_onset (const double M[9]) noexcept
    {
        const double mx = max_channel_response(M);
        return (mx > 1.0) ? (1.0 / mx) : 1.0;
    }

    // ------------------------------------------------------------- the reference
    // Everything the correction needs, established once at Compute time.
    struct WbReference
    {
        bool        valid;          // false -> keep whatever was stored before
        eConfidence confidence;
        double      keptFraction;

        double      measuredCct;    // RAW measurement — this is the UI READOUT
        double      measuredDuv;

        double      sourceCct;      // EFFECTIVE source — this drives build_wb_matrix.
        double      sourceDuv;      // Equals the measurement unless refinement ran.

        int32_t     refineRounds;   // diagnostics
        double      residualMired;  // achieved |target - re-measured| in mired

        WbReference()
            : valid(false), confidence(conf_None), keptFraction(0.0)
            , measuredCct(0.0), measuredDuv(0.0)
            , sourceCct(0.0), sourceDuv(0.0)
            , refineRounds(0), residualMired(0.0) {}
    };

    // --------------------------------------------------------------- internals
    namespace detail
    {
        inline double kelvin_to_mired (const double k) noexcept
        { return (k > 0.0) ? (1000000.0 / k) : 0.0; }
        inline double mired_to_kelvin (double m) noexcept
        {
            if (m < 25.0)   m = 25.0;
            if (m > 1111.0) m = 1111.0;
            return 1000000.0 / m;
        }
    } // namespace detail

#ifdef __AVX2__
    // =========================================================================
    // compute_reference — run at COMPUTE time on a parked frame.
    //
    // Solves for the effective source white S such that ONE application of
    // M_wb(S -> target) makes the corrected frame RE-MEASURE at the target.
    //
    // Method: secant iteration in (mired, Duv). f(S) = measure(apply(M_wb(S->T))).
    // We want f(S) = T. Two evaluations give a secant step; a third confirms.
    // Each evaluation is one apply + one measure over the frame, so the whole
    // thing costs a few tens of ms ONCE, at an explicit user action.
    //
    // IN : linearRGB (the canonical buffer for the parked frame), size,
    //      measure context, target white, CAT settings, working-space pair.
    // OUT: ref (valid=false and unchanged storage if confidence is too low).
    //
    // The caller keeps the PREVIOUS reference when this returns false.
    // =========================================================================
    inline bool compute_reference (const float* linearRGB,
                                   const int32_t sizeX, const int32_t sizeY,
                                   const AlgoPrIngest::avx2::MeasureCtxAVX2& ctx,
                                   AlgoCCT::CctHandle<double>& cctHdnl,
                                   const AlgoCCT::LutRow* lut, const std::size_t lutN,
                                   const double lutMin, const double lutMax,
                                   const double measuredCct, const double measuredDuv,
                                   const double keptFraction,
                                   const double targetCct, const double targetDuv,
                                   const int32_t catModel, const double adaptDegree,
                                   const double rgb2xyz[9], const double xyz2rgb[9],
                                   const eCOLOR_OBSERVER observer,
                                   const ConfidenceThresholds& thresholds,
                                   const int32_t maxRounds,
                                   WbReference& ref)
    {
        // ---- 1. trust gate: refuse rather than substitute -------------------
        const eConfidence conf = classify_confidence(keptFraction, measuredDuv, thresholds);
        // A measurement with no white point at all cannot be used by anyone.
        if (!(measuredCct > 0.0))
            return false;
        // Below the floor: refuse UNLESS the user has explicitly chosen to
        // accept low-confidence measurements. The rating stays honest either
        // way - accepting does not promote conf_None to something better.
        if (conf_None == conf && !thresholds.acceptBelowThreshold)
            return false;                    // caller keeps the stored reference

        ref.valid        = true;
        ref.confidence   = conf;
        ref.keptFraction = keptFraction;
        ref.measuredCct  = measuredCct;      // readout always shows the RAW value
        ref.measuredDuv  = measuredDuv;
        ref.sourceCct    = measuredCct;      // fallback if refinement cannot run
        ref.sourceDuv    = measuredDuv;
        ref.refineRounds = 0;

        if (maxRounds <= 0)
            return true;                     // refinement disabled

        // ---- 2. scratch buffers (Compute-time only, not per frame) ----------
        const std::size_t n3 = (std::size_t)sizeX * (std::size_t)sizeY * 3u;
        std::vector<float> corrected(n3);
        std::vector<float> scratch  (n3);

        const double tM = detail::kelvin_to_mired(targetCct);

        // f(S) -> the re-measured white of the frame after ONE correction from S
        struct Eval { double m; double duv; bool ok; };
        auto evaluate = [&](const double sM, const double sDuv) -> Eval
        {
            Eval e; e.m = 0.0; e.duv = 0.0; e.ok = false;
            double M[9];
            if (!build_wb_matrix(lut, lutN, lutMin, lutMax,
                                 CctDuv<double>{ detail::mired_to_kelvin(sM), sDuv },
                                 CctDuv<double>{ targetCct, targetDuv },
                                 catModel, adaptDegree, rgb2xyz, xyz2rgb, M))
                return e;

            ApplyParams ap;
            ap.strength   = 1.0;             // solve at full strength; Strength
            ap.clipPolicy = clip_Never;      // scales the result afterwards
            apply_white_balance(linearRGB, corrected.data(), sizeX, sizeY, M, ap, true);

            SuperPixel<double> sp;
            sp.r = sp.g = sp.b = 0.0;
            double kept = 0.0;
            AlgoPrIngest::avx2::measure_linear_rgb3(corrected.data(), sizeX, sizeY,
                                                    ctx, scratch.data(), sp, &kept);
            const double X = rgb2xyz[0]*sp.r + rgb2xyz[1]*sp.g + rgb2xyz[2]*sp.b;
            const double Y = rgb2xyz[3]*sp.r + rgb2xyz[4]*sp.g + rgb2xyz[5]*sp.b;
            const double Z = rgb2xyz[6]*sp.r + rgb2xyz[7]*sp.g + rgb2xyz[8]*sp.b;
            const double den = X + 15.0*Y + 3.0*Z;
            if (!(den > 0.0) || !(kept > 0.0)) return e;
            const std::pair<double,double> r =
                cctHdnl.ComputeCct({ 4.0*X/den, 6.0*Y/den }, observer);
            if (!(r.first > 0.0)) return e;
            e.m   = detail::kelvin_to_mired(r.first);
            e.duv = r.second;
            e.ok  = true;
            return e;
        };

        // ---- 3. damped Newton on a finite-difference 2x2 Jacobian ---------
        // The two axes are NOT independent: a Duv step of 0.003 was measured
        // to move the result by ~26 mired. Treating them as separate secants
        // overshoots wildly, so solve them jointly.
        //   unknowns : source (mired, Duv)
        //   residual : r(S) = f(S) - target,  f = re-measured white
        double sM = detail::kelvin_to_mired(measuredCct), sD = measuredDuv;
        Eval f0 = evaluate(sM, sD);
        if (!f0.ok) return true;                 // keep the plain measurement

        // objective: mired error dominates; Duv is scaled into mired-equivalent
        // units so the two are comparable in one norm (1 mired ~ 3.4e-4 duv).
        const double kDuvToMired = 1.0 / 3.4e-4;
        auto cost = [&](const Eval& e) {
            const double a = e.m - tM;
            const double b = (e.duv - targetDuv) * kDuvToMired;
            return std::sqrt(a*a + b*b);
        };

        double best = cost(f0);
        ref.residualMired = std::fabs(f0.m - tM);

        const double hM = 8.0;        // finite-difference steps: large enough
        const double hD = 0.004;      // to clear measurement noise, still local

        for (int32_t it = 0; it < maxRounds; ++it)
        {
            const Eval fm = evaluate(sM + hM, sD);
            if (!fm.ok) break;
            const Eval fd = evaluate(sM, sD + hD);
            if (!fd.ok) break;
            ref.refineRounds = it + 1;

            // J = d(f) / d(S)
            const double j11 = (fm.m   - f0.m  ) / hM;   // d mired / d srcMired
            const double j21 = (fm.duv - f0.duv) / hM;   // d duv   / d srcMired
            const double j12 = (fd.m   - f0.m  ) / hD;   // d mired / d srcDuv
            const double j22 = (fd.duv - f0.duv) / hD;   // d duv   / d srcDuv
            const double det = j11*j22 - j12*j21;
            if (std::fabs(det) < 1.0e-12) break;         // singular: stop

            const double r1 = tM        - f0.m;
            const double r2 = targetDuv - f0.duv;
            double dM = ( j22*r1 - j12*r2) / det;
            double dD = (-j21*r1 + j11*r2) / det;

            // damped line search - a full Newton step can leave the region
            // where the linearisation holds (the gate population shifts).
            bool improved = false;
            for (double lambda : { 1.0, 0.5, 0.25 })
            {
                double tM2 = sM + lambda*dM, tD2 = sD + lambda*dD;
                if (tM2 < 25.0)   tM2 = 25.0;
                if (tM2 > 1111.0) tM2 = 1111.0;
                if (tD2 < -0.08)  tD2 = -0.08;
                if (tD2 >  0.08)  tD2 =  0.08;
                const Eval f2 = evaluate(tM2, tD2);
                if (!f2.ok) continue;
                const double c2 = cost(f2);
                if (c2 < best) {
                    best = c2; sM = tM2; sD = tD2; f0 = f2;
                    ref.sourceCct     = detail::mired_to_kelvin(sM);
                    ref.sourceDuv     = sD;
                    ref.residualMired = std::fabs(f2.m - tM);
                    improved = true;
                    break;
                }
            }
            if (!improved)            break;             // converged or stuck
            if (ref.residualMired < 0.5) break;          // far below JND
        }
        return true;
    }
#endif // __AVX2__

} // namespace AlgoWB

#endif // __IMAGELAB2_ALGO_REFERENCE_HPP__
