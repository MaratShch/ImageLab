// ===========================================================================
//  AlgorithmMain.cpp -- film simulation core, scalar reference implementation
//
//  CONTRACT
//    IN   : memHandler.Src_* -- scene-linear planar samples in STORAGE type,
//           already linearised by the caller.
//    OUT  : memHandler.Dst_* -- display-linear planar samples in STORAGE type.
//           The caller packs to the host pixel format.
//    NOT DONE HERE: sRGB or Rec709 decode, output packing, alpha handling.
//
//  FIXED FOUR-PARAMETER PROTOTYPE
//    Everything that affects the algorithm arrives in AlgoControls - including the
//    film stock, the gauge, the frame number, the film frame rate and the seed, not
//    only the obvious sliders. Everything that is memory arrives in MemHandler,
//    including the two caller-owned lookup tables the engine may not build for
//    itself. Only the image geometry is passed directly. The prototype must not
//    grow.
//
//  RAW POINTERS, EXPLICIT GEOMETRY
//    Every buffer is a raw RESTRICT pointer pulled out of the handler once, at the
//    top. Size and pitch are passed to every stage as plain int32_t parameters.
//    There are no view or wrapper objects anywhere in the engine: a stage signature
//    that takes a pointer, a width, a height and a pitch states exactly what it
//    touches, and it is the form the eventual AVX2 path wants.
//
//    sizeX and sizeY ARE the image geometry and are used as such. pitch comes from
//    the handler's padded width, and because that padding satisfies the larger of
//    the two element alignment quanta, one pitch value is correct for every plane
//    of either element type.
//
//  BUFFER POLICY -- ONE DEDICATED BUFFER PER STAGE, NO PING-PONG.
//    Every stage reads the previous stage's buffer and writes its own, which is
//    retained for the whole frame. No stage overwrites another's output. This is a
//    deliberate development decision: any stage's output can be dumped or
//    visualised without re-running the chain, which is the only practical way to
//    debug a twenty-stage physical model against a reference. Memory optimisation
//    is deferred until the scalar and vector paths are both verified.
//
//    A consequence worth understanding: a stage that does nothing for a given stock
//    still COPIES its input to its own buffer. Skipping the pass would leave a stale
//    or uninitialised buffer in the chain, and any inspection of it would show
//    garbage.
//
//  NO ALLOCATION, NO VALIDATION, NO STATE
//    Allocates nothing; every buffer comes from the arena. Validates nothing:
//    control fields and arena pointers were both checked by the caller, and
//    re-checking them here would be duplicated work in the hottest part of the
//    program. Holds no mutable global or static state, so an arbitrary number of
//    instances may run concurrently on different frames, in any order, each with its
//    own arena.
//
//  DETERMINISM
//    Every random quantity is a pure function of (seed, frameIndex, stageId,
//    ordinal). No state carries between calls. This is mandatory, not stylistic: the
//    host renders out of order, speculatively, and from multiple concurrent
//    instances. A history-dependent model breaks under scrubbing.
//
//  TYPE POLICY
//    AlgoType for computation, ImgType for boundary storage, both chosen in
//    AlgoTypes.hpp. AlgoType is double for the first pass: verify against the
//    reference, then switch to float and re-measure. ImgType widens to AlgoType
//    once, in stage 2; the reverse happens once, in the final stage.
//
//  TIME BASE
//    frameIdx  : signed, CLIP/LAYER relative, so damage stays glued to the film
//                rather than the timeline position. May be negative. May repeat if
//                the host renders between frame boundaries.
//    frameRate : frames per second OF FILM, following layer time stretch. Correct
//                for negative-side defects, which are baked into the film, and wrong
//                for gate-side ones, which happen at projection.
// ===========================================================================

#include "AlgorithmMain.hpp"


// ---------------------------------------------------------------------------
//  PER-STAGE PERFORMANCE METRICS -- COMPILE-TIME SWITCH
//
//  1 : every stage boundary is timestamped with RDTSC and a table is written to
//      stderr as the function returns.
//  0 : every macro below expands to nothing. Not one instruction, not one byte of
//      stack, no <cstdio> and no <chrono> in the translation unit.
//
//  Defaulted ON here rather than left to the build system, because there is no
//  project makefile to carry a -D and an instrumentation switch that nobody can
//  find is an instrumentation switch that never gets used. Set it to 0 for a
//  shipping build.
//
//  A -D on the command line wins: the guard only supplies a default.
// ---------------------------------------------------------------------------
#ifndef ALGO_PROFILE_STAGES
#define ALGO_PROFILE_STAGES 1
#endif

#if defined(ALGO_PROFILE_STAGES) && (ALGO_PROFILE_STAGES != 0)
    #include <cstdio>    // std::fprintf, the report
    #include <chrono>    // steady_clock, ONE wall-clock pair per frame, see below
#endif

// Stage interfaces, in pipeline order.
#include "AlgoRelativeExposure.hpp"      // stage 2
#include "AlgoTakingFilters.hpp"         // stage 2b
#include "AlgoStockColourBalance.hpp"    // stage 3
#include "AlgoVeilingFlare.hpp"          // stage 3b
#include "AlgoTemporalFlicker.hpp"       // stage 3c   STUB
#include "AlgoCoatingField.hpp"          // stage 4 + 4b
#include "AlgoHalation.hpp"              // stage 5
#include "AlgoEmulsionMtf.hpp"           // stage 6
#include "AlgoCornerDefocus.hpp"         // stage 6b
#include "AlgoEmulsionRecord.hpp"        // stage 7
#include "AlgoCharacteristicCurve.hpp"   // stage 8
#include "AlgoReciprocity.hpp"           // stage 8, frame constant
#include "AlgoProcessVariant.hpp"        // frame setup, curve selection
#include "AlgoInterimage.hpp"            // stage 8b
#include "AlgoDirCoupler.hpp"            // stage 9
#include "AlgoNegativeDefects.hpp"       // stage 9b   IMPLEMENTED
#include "AlgoBromideDrag.hpp"           // stage 9c   IMPLEMENTED
#include "AlgoScanMtf.hpp"               // stage 10
#include "AlgoEdgeFog.hpp"               // stage 10b
#include "AlgoGrain.hpp"                 // stage 11
#include "AlgoDyeImpurity.hpp"
#include "AlgoCallier.hpp"   // stage 12b (queue C41)           // stage 12
#include "AlgoDuplication.hpp"           // stage 13
#include "AlgoTransmittance.hpp"         // stage 14
#include "AlgoReseauReconstruct.hpp"     // stage 14b
#include "AlgoSilverTone.hpp"            // stage 14c
#include "AlgoGateWeave.hpp"             // stage 15   IMPLEMENTED
#include "AlgoGateDefects.hpp"           // stage 16   IMPLEMENTED
#include "AlgoFinalClamp.hpp"            // stage 17


namespace
{
    // ----------------------------------------------------------------------
    //  Per-stage seed salts.
    //
    //  Each stage combines the caller's global seed with its own salt, so two
    //  stages drawing the same ordinal from the same generator stream still get
    //  independent numbers. Distinct arbitrary constants; nothing depends on their
    //  values beyond their being different from each other.
    // ----------------------------------------------------------------------
#if defined(ALGO_PROFILE_STAGES) && (ALGO_PROFILE_STAGES != 0)

    // ----------------------------------------------------------------------
    //  Slot capacity.
    //
    //  27 are used: one setup segment, one anchor solve and twenty-five stage
    //  calls. 48 leaves room for the defect classes that are still stubs without
    //  anyone having to remember to raise it. Cost is one pointer plus one 64-bit
    //  stamp per slot ON THE STACK - 768 bytes, gone when the function returns.
    // ----------------------------------------------------------------------
    constexpr int32_t ALGO_PROF_SLOTS = 48;


    // ----------------------------------------------------------------------
    //  Stage timer.
    //
    //  STACK-LOCAL, ONE PER CALL. Not a static, not a global, not a singleton.
    //  The engine's reentrancy guarantee - an arbitrary number of concurrent
    //  instances on different frames, in any order - would be void the moment the
    //  instrumentation introduced shared mutable state, and a profiler that
    //  breaks the thing it measures is worse than no profiler.
    //
    //  HOW THE INTERVALS ARE FORMED
    //
    //  mark() is called BEFORE each stage and stores one timestamp. Stage i then
    //  costs stamp[i+1] - stamp[i], and end() supplies the closing stamp. So N
    //  stages need N+1 timestamps and exactly N+1 RDTSC instructions - there is no
    //  start/stop pair per stage and therefore no way for the two halves of a pair
    //  to disagree about which stage they belong to.
    //
    //  WHY RDTSC AND NOT std::chrono PER STAGE
    //
    //  RDTSC is a handful of cycles. steady_clock::now() is a function call that on
    //  Linux reaches into the vDSO and costs on the order of 20-25 ns; twenty-seven
    //  of those is about 0.6 ms, which is a measurable slice of the frame being
    //  measured and would sit disproportionately on the cheap stages - the reading
    //  would be worst exactly where precision matters most.
    //
    //  WHY ONE std::chrono PAIR IS STILL TAKEN
    //
    //  RDTSC counts a reference clock, and nothing in the instruction tells you what
    //  that clock's rate is. Assuming the nominal CPU frequency is wrong on any part
    //  whose base and TSC rates differ, and turbo makes a guess worse rather than
    //  better. So the frame is bracketed by ONE pair of steady_clock reads and the
    //  cycles-to-milliseconds factor is DERIVED from the frame itself. Two vDSO
    //  calls per frame, and the millisecond column is then correct on any host
    //  rather than correct on the one it was calibrated against.
    //
    //  This relies on the TSC being invariant - constant rate regardless of core
    //  frequency and power state - which is guaranteed on every x86 part this engine
    //  targets. On a machine without it the cycle counts stay exact and the derived
    //  milliseconds become an average rather than a truth.
    //
    //  WHAT THIS MEASUREMENT CANNOT TELL YOU
    //
    //  RDTSC is not a serialising instruction, so the processor may retire it a
    //  little before or after the surrounding work. The skew is tens of cycles,
    //  against stages that measure in the tens of millions, so it is below the
    //  resolution of anything worth acting on - but it is real, and it is why a
    //  stage reading 0.02 ms should be read as "too cheap to measure" rather than as
    //  a number. No LFENCE is issued: the fence would cost more than the error it
    //  removes, and would itself perturb the pipeline being measured.
    //
    //  Time attributes to the CALLER. Eight stages spend most of their time inside
    //  AlgoSeparableBlur.cpp; that time appears against the stage that asked for the
    //  blur, which is the useful attribution when deciding what to optimise, and a
    //  misleading one if read as "code in this file".
    // ----------------------------------------------------------------------
    struct AlgoStageProfile
    {
        const char* label[ALGO_PROF_SLOTS];   // borrowed string literals, never freed
        uint64_t    stamp[ALGO_PROF_SLOTS];   // one per boundary, N stages -> N+1 used
        int32_t     used;                     // boundaries recorded so far

        std::chrono::steady_clock::time_point wall0;   // calibration bracket, open
        std::chrono::steady_clock::time_point wall1;   // calibration bracket, close

        // Arm the timer. The wall clock is read FIRST and the counter LAST, so the
        // calibration window encloses the counter window rather than overlapping it;
        // the derived rate is then a slight underestimate of cycles per millisecond
        // instead of an unbounded either-way error.
        inline void begin () noexcept
        {
            used  = 0;
            wall0 = std::chrono::steady_clock::now();
        }

        // Open the segment named by 'label'. Called immediately before the stage.
        //
        // The bounds test is a compile-time-constant comparison against a
        // stack-array size and predicts perfectly; it is here so that adding a
        // twenty-eighth stage and forgetting to raise the capacity silently drops
        // the last row rather than corrupting the stack frame.
        inline void mark (const char* const name) noexcept
        {
            if (used < ALGO_PROF_SLOTS)
            {
                label[used] = name;
                stamp[used] = RDTSC();
                used++;
            }
        }

        // Close the last segment and write the table.
        void end (const int32_t sizeX, const int32_t sizeY) noexcept
        {
            // The closing timestamp needs a slot of its own, one past the last
            // segment. If every slot was consumed by marks there is nowhere to put it,
            // so the final segment is discarded rather than written out of bounds -
            // the table then comes up one row short, which is a visible symptom, and
            // the alternative is a stack smash, which is not.
            if (used >= ALGO_PROF_SLOTS)
                used = ALGO_PROF_SLOTS - 1;

            stamp[used] = RDTSC();

            wall1 = std::chrono::steady_clock::now();

            // Nothing to report if nobody marked anything.
            if (used < 1)
                return;

            // One row per mark: mark() OPENS a segment and end() closed the last one,
            // so N marks produced N segments and N+1 timestamps. Reading rows as
            // used-1 here is what dropped the final stage from the first version of
            // this table - the row count follows the marks, not the boundaries.
            const int32_t  rows  = used;
            const uint64_t total = stamp[used] > stamp[0]
                                 ? (stamp[used] - stamp[0])
                                 : 1;                            // 1, never 0: it divides

            // Wall time of the whole frame, milliseconds, from the one chrono pair.
            const double wallMs =
                std::chrono::duration<double, std::milli>(wall1 - wall0).count();

            // Cycles to milliseconds, derived rather than assumed. Guarded because a
            // sub-microsecond frame - a 1x1 render in a unit test - can round the
            // wall measurement to zero and would otherwise divide by it.
            const double msPerCycle = (wallMs > 0.0)
                                    ? (wallMs / static_cast<double>(total))
                                    : 0.0;

            // Everything goes to stderr, not stdout: a host that captures the
            // engine's stdout for image data must not receive a diagnostic table in
            // the middle of it.
            std::fprintf(stderr,
                "\n[Algorithm_Main] per-stage metrics   %d x %d   (%.3f Mpx)\n",
                sizeX, sizeY,
                (static_cast<double>(sizeX) * static_cast<double>(sizeY)) * 1.0e-6);

            std::fprintf(stderr,
                "  #   stage                            Mcycles       mSec       %%     ns/px\n");
            std::fprintf(stderr,
                "  --  ------------------------------  --------   --------   -----   -------\n");

            // Pixels, for the per-pixel column - the only figure in the table that
            // is comparable between two different frame sizes.
            const double px = (static_cast<double>(sizeX) * static_cast<double>(sizeY));

            for (int32_t i = 0; i < rows; i++)
            {
                const uint64_t dc = (stamp[i + 1] > stamp[i])
                                  ? (stamp[i + 1] - stamp[i])
                                  : 0;

                const double ms = static_cast<double>(dc) * msPerCycle;

                std::fprintf(stderr,
                    "  %2d  %-30s  %8.2f   %8.3f   %5.1f   %7.3f\n",
                    i,
                    label[i],
                    static_cast<double>(dc) * 1.0e-6,
                    ms,
                    (static_cast<double>(dc) * 100.0) / static_cast<double>(total),
                    (px > 0.0) ? (ms * 1.0e6 / px) : 0.0);
            }

            std::fprintf(stderr,
                "  --  ------------------------------  --------   --------   -----   -------\n");
            std::fprintf(stderr,
                "      %-30s  %8.2f   %8.3f   %5.1f   %7.3f\n",
                "TOTAL",
                static_cast<double>(total) * 1.0e-6,
                static_cast<double>(total) * msPerCycle,
                100.0,
                (px > 0.0) ? (static_cast<double>(total) * msPerCycle * 1.0e6 / px) : 0.0);

            // Wall time alongside the summed cycles. They should agree to a fraction
            // of a per cent; a visible gap means the thread was descheduled or
            // migrated mid-frame, and the whole run should be discarded rather than
            // interpreted.
            std::fprintf(stderr,
                "      wall clock %.3f mSec   (sum of stages %.3f mSec)\n\n",
                wallMs,
                static_cast<double>(total) * msPerCycle);

            return;
        }
    };

#endif  // ALGO_PROFILE_STAGES


    constexpr uint32_t ALGO_SALT_FLICKER      = 0x9E3779B1u;
    constexpr uint32_t ALGO_SALT_COATING      = 0x85EBCA77u;
    constexpr uint32_t ALGO_SALT_NEG_DEFECTS  = 0xC2B2AE3Du;
    constexpr uint32_t ALGO_SALT_MISREG       = 0x27D4EB2Fu;
    constexpr uint32_t ALGO_SALT_GRAIN        = 0x165667B1u;
    constexpr uint32_t ALGO_SALT_DUPE         = 0xD3A2646Cu;
    constexpr uint32_t ALGO_SALT_PRINT_GRAIN  = 0xFD7046C5u;
    constexpr uint32_t ALGO_SALT_WEAVE        = 0xB55A4F09u;
    constexpr uint32_t ALGO_SALT_GATE_DEFECTS = 0x2545F491u;


    // ----------------------------------------------------------------------
    //  Find a print stock by name, falling back to a second name.
    //
    //  The table is caller-owned memory reached through the handler, because
    //  film::GetPrintStocks() returns a std::vector by value and would allocate on
    //  every frame.
    //
    //  Returns null when neither name matches, which every consumer handles: the
    //  print stage passes the negative through and reports the negative's own
    //  curves, and the anchor solve returns the neutral negative densities.
    // ----------------------------------------------------------------------
    const film::PrintStock* findPrintStock
    (
        const film::PrintStock* RESTRICT pDb,
        const int32_t                    count,
        const char* const                wanted,
        const std::string&               fallback
    ) noexcept
    {
        // First choice: the requested name, when the caller supplied one.
        if ((nullptr != wanted) && ('\0' != wanted[0]))
        {
            for (int32_t i = 0; i < count; i++)
            {
                if (pDb[i].name == wanted)
                    return &pDb[i];
            }
        }

        // Second choice: the fallback, which is normally the stock's own default. A
        // stale preset therefore degrades to the stock's intended print rather than
        // to nothing.
        for (int32_t i = 0; i < count; i++)
        {
            if (pDb[i].name == fallback)
                return &pDb[i];
        }

        return nullptr;
    }
}


// ---------------------------------------------------------------------------
//  INSTRUMENTATION MACROS
//
//  Three lines in the pipeline body, and all three vanish when the switch is 0 -
//  which is why they are macros and not calls on an object that would still have to
//  exist. There is no "null profiler" to be optimised away and therefore nothing to
//  trust the optimiser about.
//
//  ALGO_PROF_BEGIN declares the timer, so it must appear at function scope and not
//  inside a conditional. It is used exactly once, at the top.
// ---------------------------------------------------------------------------
#if defined(ALGO_PROFILE_STAGES) && (ALGO_PROFILE_STAGES != 0)
    #define ALGO_PROF_BEGIN()       AlgoStageProfile algoProf; algoProf.begin()
    #define ALGO_PROF_MARK(name)    algoProf.mark(name)
    #define ALGO_PROF_END()         algoProf.end(sizeX, sizeY)
#else
    #define ALGO_PROF_BEGIN()       ((void)0)
    #define ALGO_PROF_MARK(name)    ((void)0)
    #define ALGO_PROF_END()         ((void)0)
#endif


void Algorithm_Main
(
    const MemHandler&    memHandler,
    const int32_t        sizeX,
    const int32_t        sizeY,
    const AlgoControls&  algoCtrl
) noexcept
{
    // -----------------------------------------------------------------------
    // 0. SETUP
    //
    // Row pitch, in ELEMENTS, from the handler's padded width. Every plane of
    // either element type shares it, because the padding satisfies the larger of the
    // two alignment quanta.
    //
    // Frame-invariant quantities are recomputed per frame on purpose. Measured on
    // the reference model at HD they amount to a fraction of one per cent of the
    // frame, so caching them would add an API surface and a lifetime question for no
    // measurable gain.
    // -----------------------------------------------------------------------
    // Arm the per-stage timer and open the setup segment. Everything between here
    // and the first stage - stock lookup, gauge lookup, px_per_mm, the print and
    // dupe stock searches - lands in one row called "setup", because it is
    // frame-invariant work that would be cached if it were ever large enough to
    // matter and the point of measuring it is to confirm that it is not.
    ALGO_PROF_BEGIN();
    ALGO_PROF_MARK("00   setup / stock resolve");

    const int32_t pitch = memHandler.padW;

    // -----------------------------------------------------------------------
    //  Resolve the film stock from the control structure.
    //
    //  The profile database is caller-owned memory reached through the handler. The
    //  engine cannot call film::GetFilmDatabase() itself: that returns a std::vector
    //  by value and would allocate on every frame.
    //
    //  The index is used directly, with no range check: the enumerator comes from
    //  the effect panel and is pre-validated, and re-testing it here would be
    //  duplicated work on the hot path.
    // -----------------------------------------------------------------------
    const film::FilmProfile& profileAsShipped =
        memHandler.pProfileDb[UnderlyingType(algoCtrl.filmProfile)];

    // -----------------------------------------------------------------------
    //  The chosen PROCESS, resolved before anything reads a curve.
    //
    //  A process variant is a different DEVELOPMENT of the same emulsion, and
    //  where the manufacturer plotted it separately the record carries its own
    //  traced curve set. Applying it by overriding the profile - rather than by
    //  handing a curve set to each consumer - is what guarantees the anchor
    //  solve, stage 8, the grain amplitude and the dupe chain all render the
    //  same film. See AlgoProcessVariant.hpp.
    //
    //  INERT AT THE DEFAULT: processVariant is -1 unless the caller selects
    //  one, `variantStore` is never written, and the reference below binds
    //  straight to the database entry.
    // -----------------------------------------------------------------------
    film::FilmProfile variantStore;

    const film::FilmProfile& profile =
        AlgoResolveProcessVariant(profileAsShipped,
                                  algoCtrl.processVariant,
                                  variantStore);

    // -----------------------------------------------------------------------
    //  Resolve the gauge, and from it every physical frame dimension.
    //
    //  An empty filmFormat string means "use the stock's own default_format", which
    //  is how the database supplies a sensible gauge without the caller having to
    //  know one. A name that matches nothing falls back to the same default, so a
    //  stale preset degrades to the stock's intended gauge rather than to zero.
    // -----------------------------------------------------------------------
    const char* const wantFormat =
        (algoCtrl.filmFormat != nullptr && algoCtrl.filmFormat[0] != '\0')
            ? algoCtrl.filmFormat
            : profile.default_format.c_str();

    const film::FilmFormat* pFmt = nullptr;

    for (int32_t i = 0; i < memHandler.formatCount; i++)
    {
        if (memHandler.pFormatDb[i].name == wantFormat)
        {
            pFmt = &memHandler.pFormatDb[i];
            break;
        }
    }

    if (nullptr == pFmt)
    {
        // Requested gauge unknown: fall back to the stock's own default.
        for (int32_t i = 0; i < memHandler.formatCount; i++)
        {
            if (memHandler.pFormatDb[i].name == profile.default_format)
            {
                pFmt = &memHandler.pFormatDb[i];
                break;
            }
        }
    }

    // Physical frame extents on the film, millimetres, and the web advance per
    // frame. A format that is still unresolved leaves these at zero, which disables
    // every physically scaled effect rather than producing a wrong scale - the
    // conservative failure, and the only one available without validating.
    const AlgoType negWidthMm   = (pFmt != nullptr)
                                ? static_cast<AlgoType>(pFmt->width_mm)  : ALGO_ZERO;
    const AlgoType negHeightMm  = (pFmt != nullptr)
                                ? static_cast<AlgoType>(pFmt->height_mm) : ALGO_ZERO;
    const AlgoType framePitchMm = (pFmt != nullptr)
                                ? static_cast<AlgoType>(pFmt->FramePitchMm()) : ALGO_ZERO;

    // px_per_mm. THE resolution-independence mechanism: it converts every physical
    // quantity in the profile - micrometres, cycles per millimetre, lines per
    // millimetre - into pixels, which is how one profile renders correctly at any
    // resolution. Derived from the requested width and the gauge, so an 8 mm stock
    // rendered at its 4.80 mm gauge scales correctly without the caller computing
    // anything.
    const AlgoType pxPerMm = (negWidthMm > ALGO_ZERO)
                           ? (static_cast<AlgoType>(sizeX) / negWidthMm)
                           : ALGO_ZERO;

    // -----------------------------------------------------------------------
    //  Resolve the print and duplicating stocks.
    //
    //  Both tables are the same caller-owned array. The print stock is needed by the
    //  anchor solve at stage 8, by the print at 13 and by print grain at 14; the
    //  duplicating stock only by the generation chain at 13.
    // -----------------------------------------------------------------------
    const film::PrintStock* const pPrint =
        findPrintStock(memHandler.pPrintDb, memHandler.printCount,
                       algoCtrl.printStock, profile.default_print);

    // The dupe stock has no per-stock default, so the fallback is the release print
    // itself: printing onto the release stock is wrong for an intermediate but is a
    // far smaller error than skipping the generation chain the user asked for.
    const film::PrintStock* const pDupe =
        findPrintStock(memHandler.pPrintDb, memHandler.printCount,
                       algoCtrl.dupeStock,
                       (nullptr != pPrint) ? pPrint->name : profile.default_print);

    // -----------------------------------------------------------------------
    //  Scan optics.
    //
    //  The 50 per cent modulation frequency of the scanner or telecine, which
    //  band-limits both the image at stage 10 AND every grain field, because the
    //  same lens sits between the film and the sensor in both cases.
    //
    //  KNOWN OMISSION: AlgoControls has no scannerF50 field, so the print stock's
    //  combined print-plus-scanner figure is used unconditionally. The reference
    //  model lets the user override it. Adding the control is a change to
    //  AlgoControl.hpp and is reported rather than made here.
    // -----------------------------------------------------------------------
    const AlgoType scanF50 = (nullptr != pPrint)
                           ? static_cast<AlgoType>(pPrint->mtf_f50)
                           : ALGO_ZERO;

    // As a Gaussian sigma in pixels, which is the form both consumers want.
    const AlgoType scanSigmaPx = AlgoScanSigmaMm(scanF50) * pxPerMm;

    // -----------------------------------------------------------------------
    //  Did stage 7 actually build a mosaic?
    //
    //  Not the same question as profile.has_reseau. The mosaic is skipped when the
    //  grid cannot be resolved at this render size, and the grain stage has to follow
    //  the same decision: a low-resolution Dufaycolor render carries three ordinary
    //  records and must get three independent grain fields, while a high-resolution
    //  one carries a single panchromatic record and must get one.
    //
    //  Computed once here from the same helper stage 7 uses, so the two cannot
    //  disagree.
    // -----------------------------------------------------------------------
    const bool hasMosaic =
        profile.has_reseau
        && algoCtrl.reseau
        && (AlgoReseauPitchPx(profile.reseau, pxPerMm) >= ALGO_RESEAU_MIN_PITCH_PX);

    // Time base. frameIdx is CLIP relative, so damage stays glued to the film rather
    // than to the timeline position; frameRate follows layer time stretch.
    const int32_t  frameIndex = algoCtrl.frameIndex;
    const AlgoType frameRate  = static_cast<AlgoType>(algoCtrl.frameRate);

    // -----------------------------------------------------------------------
    // Raw pointers, pulled once.
    //
    // RESTRICT on all of them: they address distinct arena planes that provably do
    // not alias, and without the qualifier every store forces a reload of the source
    // and no loop will vectorise.
    // -----------------------------------------------------------------------
    const ImgType* RESTRICT iR = memHandler.Src_R;
    const ImgType* RESTRICT iG = memHandler.Src_G;
    const ImgType* RESTRICT iB = memHandler.Src_B;

    ImgType* RESTRICT oR = memHandler.Dst_R;
    ImgType* RESTRICT oG = memHandler.Dst_G;
    ImgType* RESTRICT oB = memHandler.Dst_B;

    AlgoType* RESTRICT s02R  = memHandler.S02_R;
    AlgoType* RESTRICT s02G  = memHandler.S02_G;
    AlgoType* RESTRICT s02B  = memHandler.S02_B;

    AlgoType* RESTRICT s02bR = memHandler.S02b_R;
    AlgoType* RESTRICT s02bG = memHandler.S02b_G;
    AlgoType* RESTRICT s02bB = memHandler.S02b_B;

    AlgoType* RESTRICT s03R  = memHandler.S03_R;
    AlgoType* RESTRICT s03G  = memHandler.S03_G;
    AlgoType* RESTRICT s03B  = memHandler.S03_B;

    AlgoType* RESTRICT s03bR = memHandler.S03b_R;
    AlgoType* RESTRICT s03bG = memHandler.S03b_G;
    AlgoType* RESTRICT s03bB = memHandler.S03b_B;

    AlgoType* RESTRICT s03cR = memHandler.S03c_R;
    AlgoType* RESTRICT s03cG = memHandler.S03c_G;
    AlgoType* RESTRICT s03cB = memHandler.S03c_B;

    AlgoType* RESTRICT s04R  = memHandler.S04_R;
    AlgoType* RESTRICT s04G  = memHandler.S04_G;
    AlgoType* RESTRICT s04B  = memHandler.S04_B;

    AlgoType* RESTRICT s05R  = memHandler.S05_R;
    AlgoType* RESTRICT s05G  = memHandler.S05_G;
    AlgoType* RESTRICT s05B  = memHandler.S05_B;

    AlgoType* RESTRICT s06R  = memHandler.S06_R;
    AlgoType* RESTRICT s06G  = memHandler.S06_G;
    AlgoType* RESTRICT s06B  = memHandler.S06_B;

    AlgoType* RESTRICT s06bR = memHandler.S06b_R;
    AlgoType* RESTRICT s06bG = memHandler.S06b_G;
    AlgoType* RESTRICT s06bB = memHandler.S06b_B;

    AlgoType* RESTRICT s07R  = memHandler.S07_R;
    AlgoType* RESTRICT s07G  = memHandler.S07_G;
    AlgoType* RESTRICT s07B  = memHandler.S07_B;

    AlgoType* RESTRICT s08R  = memHandler.S08_R;
    AlgoType* RESTRICT s08G  = memHandler.S08_G;
    AlgoType* RESTRICT s08B  = memHandler.S08_B;

    AlgoType* RESTRICT s08bR = memHandler.S08b_R;
    AlgoType* RESTRICT s08bG = memHandler.S08b_G;
    AlgoType* RESTRICT s08bB = memHandler.S08b_B;

    AlgoType* RESTRICT s09R  = memHandler.S09_R;
    AlgoType* RESTRICT s09G  = memHandler.S09_G;
    AlgoType* RESTRICT s09B  = memHandler.S09_B;

    AlgoType* RESTRICT s09bR = memHandler.S09b_R;
    AlgoType* RESTRICT s09bG = memHandler.S09b_G;
    AlgoType* RESTRICT s09bB = memHandler.S09b_B;

    AlgoType* RESTRICT s10R  = memHandler.S10_R;
    AlgoType* RESTRICT s10G  = memHandler.S10_G;
    AlgoType* RESTRICT s10B  = memHandler.S10_B;

    AlgoType* RESTRICT s10bR = memHandler.S10b_R;
    AlgoType* RESTRICT s10bG = memHandler.S10b_G;
    AlgoType* RESTRICT s10bB = memHandler.S10b_B;

    AlgoType* RESTRICT s11R  = memHandler.S11_R;
    AlgoType* RESTRICT s11G  = memHandler.S11_G;
    AlgoType* RESTRICT s11B  = memHandler.S11_B;

    AlgoType* RESTRICT s12R  = memHandler.S12_R;
    AlgoType* RESTRICT s12G  = memHandler.S12_G;
    AlgoType* RESTRICT s12B  = memHandler.S12_B;

    AlgoType* RESTRICT s13R  = memHandler.S13_R;
    AlgoType* RESTRICT s13G  = memHandler.S13_G;
    AlgoType* RESTRICT s13B  = memHandler.S13_B;

    AlgoType* RESTRICT s14R  = memHandler.S14_R;
    AlgoType* RESTRICT s14G  = memHandler.S14_G;
    AlgoType* RESTRICT s14B  = memHandler.S14_B;

    AlgoType* RESTRICT s14bR = memHandler.S14b_R;
    AlgoType* RESTRICT s14bG = memHandler.S14b_G;
    AlgoType* RESTRICT s14bB = memHandler.S14b_B;

    AlgoType* RESTRICT s14cR = memHandler.S14c_R;
    AlgoType* RESTRICT s14cG = memHandler.S14c_G;
    AlgoType* RESTRICT s14cB = memHandler.S14c_B;

    AlgoType* RESTRICT s15R  = memHandler.S15_R;
    AlgoType* RESTRICT s15G  = memHandler.S15_G;
    AlgoType* RESTRICT s15B  = memHandler.S15_B;

    AlgoType* RESTRICT s16R  = memHandler.S16_R;
    AlgoType* RESTRICT s16G  = memHandler.S16_G;
    AlgoType* RESTRICT s16B  = memHandler.S16_B;

    AlgoType* RESTRICT s17R  = memHandler.S17_R;
    AlgoType* RESTRICT s17G  = memHandler.S17_G;
    AlgoType* RESTRICT s17B  = memHandler.S17_B;

    // Log exposure, filled by stage 8 and RETAINED for stage 8b. Not scratch while
    // those two are in flight: the interimage effect reads it after stage 8 has
    // finished, and density cannot be inverted back to log exposure through the
    // shoulder. It IS reused as a scratch triple from stage 13 onward, by which point
    // nothing needs it any more.
    AlgoType* RESTRICT logER = memHandler.Scr_LogE_R;
    AlgoType* RESTRICT logEG = memHandler.Scr_LogE_G;
    AlgoType* RESTRICT logEB = memHandler.Scr_LogE_B;

    // -----------------------------------------------------------------------
    //  Scratch planes.
    //
    //  SCRATCH REUSE ACROSS STAGES IS SAFE, AND WHY.
    //
    //  The stages run strictly in sequence and no scratch plane carries information
    //  from one stage to the next: each one writes every element it later reads,
    //  within a single stage.
    //
    //  What is NOT safe, and has bitten this code twice, is aliasing WITHIN one
    //  stage. The multi-lobe blur re-reads its source once per lobe, so a scratch
    //  plane that also serves as the source is destroyed by the first lobe and the
    //  remaining lobes integrate the wreckage. Stage 3b needs four distinct working
    //  planes, stage 5 needs five, stage 13 needs seven; every call below is given
    //  planes distinct from each other and from its own source and destination.
    // -----------------------------------------------------------------------
    AlgoType* RESTRICT scrLuma     = memHandler.Scr_Luma;
    AlgoType* RESTRICT scrBlurA    = memHandler.Scr_BlurA;
    AlgoType* RESTRICT scrBlurB    = memHandler.Scr_BlurB;
    AlgoType* RESTRICT scrDbar     = memHandler.Scr_Dbar;
    AlgoType* RESTRICT scrDbarBlur = memHandler.Scr_DbarBlur;
    AlgoType* RESTRICT scrField    = memHandler.Scr_Field;
    AlgoType* RESTRICT scrFieldLo  = memHandler.Scr_FieldLo;
    AlgoType* RESTRICT scrGrainR   = memHandler.Scr_Grain_R;
    AlgoType* RESTRICT scrGrainG   = memHandler.Scr_Grain_G;
    AlgoType* RESTRICT scrGrainB   = memHandler.Scr_Grain_B;

    // -----------------------------------------------------------------------
    // 1. LINEARISATION -- done by the caller before this function is entered.
    //    The source planes are already scene-linear.
    // -----------------------------------------------------------------------

    // -----------------------------------------------------------------------
    // 2. RELATIVE EXPOSURE                                 Src -> S02
    //
    //    Scene linear to exposure units: divide by the 18 per cent mid-grey
    //    reference, then apply the camera exposure offset in stops. Also the one
    //    place ImgType widens to AlgoType.
    //
    //    Establishes the unit system every later stage depends on. An 18 per cent
    //    grey card lands at exactly 1.0, so log10 of exposure is 0 there, which is
    //    the origin the curve parameters and the anchor solve are both expressed
    //    against.
    //
    //    Output is UNCLAMPED and stays so until the single final clamp at stage 17.
    //    The characteristic curve's shoulder needs real highlight information in
    //    order to roll it off; clamping early is what makes digital highlights look
    //    digital.
    // -----------------------------------------------------------------------
    ALGO_PROF_MARK("02   relative exposure");
    AlgoStage02_RelativeExposure(iR, iG, iB,
                                 s02R, s02G, s02B,
                                 sizeX, sizeY, pitch, algoCtrl);

    // -----------------------------------------------------------------------
    // 2b. CAMERA TAKING FILTERS                            S02 -> S02b
    //
    //     A 3x3 mix across the three colour records, with POSITIVE off-diagonals and
    //     row sums above one, because camera filters OVERLAP: a red filter passes
    //     some green light, so the red record receives more exposure than it was
    //     meant to. Additive.
    //
    //     Not to be confused with the dye matrix at stage 12, which is subtractive
    //     and has unit row sums. Same shape, opposite convention.
    // -----------------------------------------------------------------------
    ALGO_PROF_MARK("02b  taking filters");
    AlgoStage02b_TakingFilters(s02R, s02G, s02B,
                               s02bR, s02bG, s02bB,
                               sizeX, sizeY, pitch, profile);

    // -----------------------------------------------------------------------
    // 3. STOCK COLOUR BALANCE                              S02b -> S03
    //
    //    The mismatch between the scene illuminant and the colour temperature the
    //    stock was balanced for, from Planck's law. Tungsten stock in daylight goes
    //    blue and daylight stock under tungsten goes orange, and both are the correct
    //    answer rather than an error to be corrected.
    // -----------------------------------------------------------------------
    ALGO_PROF_MARK("03   stock colour balance");
    AlgoStage03_StockColourBalance(s02bR, s02bG, s02bB,
                                   s03R, s03G, s03B,
                                   sizeX, sizeY, pitch, profile, algoCtrl);

    // -----------------------------------------------------------------------
    // 3b. VEILING FLARE                                    S03 -> S03b
    //
    //     Broad scatter inside the taking lens. Lifts the black floor and compresses
    //     contrast across the whole frame - which nothing else in the pipeline does,
    //     and is the commonest reason a period emulsion still has modern blacks.
    //
    //     Energy conserving: the direct image is scaled DOWN by the flare fraction
    //     and the scattered component added in its place.
    // -----------------------------------------------------------------------
    ALGO_PROF_MARK("03b  veiling flare");
    AlgoStage03b_VeilingFlare(s03R, s03G, s03B,
                              s03bR, s03bG, s03bB,
                              scrLuma, scrBlurA, scrBlurB, scrDbar,
                              sizeX, sizeY, pitch,
                              profile, algoCtrl, pxPerMm);

    // -----------------------------------------------------------------------
    // 3c. TEMPORAL EXPOSURE FLICKER                        S03b -> S03c   STUB
    //
    //     Hand-cranked cameras and early intermittent mechanisms did not deliver
    //     equal exposure to successive frames.
    //
    //     NOT YET MODELLED - passes through. GENUINELY a stub: the stage voids
    //     all five of its arguments and copies input to output, and the control
    //     it would read (damage.flickerStops) has no consumer anywhere. It sits
    //     here rather than later because
    //     it must act on EXPOSURE, before the curve: a brightness change applied
    //     after development is a grade, and a grade does not move highlights through
    //     the shoulder the way a genuine exposure change does.
    // -----------------------------------------------------------------------
    ALGO_PROF_MARK("03c  temporal flicker");
    AlgoStage03c_TemporalFlicker(s03bR, s03bG, s03bB,
                                 s03cR, s03cG, s03cB,
                                 sizeX, sizeY, pitch,
                                 profile, algoCtrl,
                                 frameIndex, frameRate, ALGO_SALT_FLICKER);

    // -----------------------------------------------------------------------
    // 4 + 4b. COATING FIELD AND LENS VIGNETTE              S03c -> S04
    //
    //         Two mechanisms with different physics and different geometry, combined
    //         into one multiplicative field and applied in a single pass.
    //
    //         The vignette is locked to the FRAME - cos^4 geometry, corner pinned to
    //         the requested loss. The coating field is locked to the WEB and slides
    //         one frame pitch per frame along it, because a coating defect is fixed
    //         on the film and the film moves.
    //
    //         Last stage before halation.
    // -----------------------------------------------------------------------
    ALGO_PROF_MARK("04   coating + vignette");
    AlgoStage04_CoatingAndVignette(s03cR, s03cG, s03cB,
                                   s04R, s04G, s04B,
                                   scrField, scrFieldLo,
                                   sizeX, sizeY, pitch,
                                   profile, algoCtrl,
                                   negWidthMm, negHeightMm, framePitchMm,
                                   frameIndex, ALGO_SALT_COATING);

    // -----------------------------------------------------------------------
    // 5. HALATION                                          S04 -> S05
    //
    //    Light passes through the emulsion, reflects off the back of the base and
    //    re-enters displaced sideways, so a bright highlight grows a halo. Red
    //    dominates because its layer sits deepest and so nearest the base, which is
    //    why the glow round a tungsten highlight is orange rather than white.
    //
    //    ENERGY IS CONSERVED: the stage adds gain * (blur(above) - above), not
    //    gain * blur(above). Adding the blur alone would invent energy and lift the
    //    whole exposure scale in proportion to the gain.
    //
    //    Five distinct working planes, for the aliasing reason set out above.
    // -----------------------------------------------------------------------
    ALGO_PROF_MARK("05   halation");
    AlgoStage05_Halation(s04R, s04G, s04B,
                         s05R, s05G, s05B,
                         scrLuma, scrField, scrFieldLo, scrBlurA, scrBlurB,
                         sizeX, sizeY, pitch,
                         profile, algoCtrl, pxPerMm);

    // -----------------------------------------------------------------------
    // 6. EMULSION MTF                                      S05 -> S06
    //
    //    Scatter between the silver halide crystals spreads a point of light into a
    //    small patch before it is absorbed.
    //
    //    ORDER MATTERS ABSOLUTELY. This acts on EXPOSURE, before development, so it
    //    blurs the image but NOT the grain - grain is created later, at stage 11.
    //    Grain smoother than the image it sits on is the immediate visual signature
    //    of a simulation whose stage order is wrong.
    // -----------------------------------------------------------------------
    ALGO_PROF_MARK("06   emulsion MTF");
    AlgoStage06_EmulsionMtf(s05R, s05G, s05B,
                            s06R, s06G, s06B,
                            scrBlurA, scrBlurB,
                            sizeX, sizeY, pitch,
                            profile, pxPerMm);

    // -----------------------------------------------------------------------
    // 6b. CORNER DEFOCUS                                   S06 -> S06b
    //
    //     The pressure plate holds the middle of the frame against the aperture
    //     plate; the corners of a curling base lift out of the focal plane.
    //
    //     Corner SOFTNESS, never corner darkening - those two get conflated
    //     constantly and they live in different stages for exactly that reason.
    // -----------------------------------------------------------------------
    ALGO_PROF_MARK("06b  corner defocus");
    AlgoStage06b_CornerDefocus(s06R, s06G, s06B,
                               s06bR, s06bG, s06bB,
                               scrBlurA, scrBlurB,
                               sizeX, sizeY, pitch,
                               profile, algoCtrl);

    // -----------------------------------------------------------------------
    // 7. EMULSION RECORD                                   S06b -> S07
    //
    //    What the emulsion physically stores: three records for a tripack, one silver
    //    image for a monochrome stock, or one record behind a colour filter mosaic
    //    for an additive-colour stock.
    //
    //    Exposure domain, because light passes the filter grid BEFORE it reaches the
    //    emulsion. LAST STAGE IN EXPOSURE SPACE.
    // -----------------------------------------------------------------------
    ALGO_PROF_MARK("07   emulsion record");
    AlgoStage07_EmulsionRecord(s06bR, s06bG, s06bB,
                               s07R, s07G, s07B,
                               sizeX, sizeY, pitch,
                               profile, algoCtrl, pxPerMm);

    // -----------------------------------------------------------------------
    //  The anchor solve.
    //
    //  Where a neutral 18 per cent grey has to land. For a negative these come back
    //  as PRINT EXPOSURE OFFSETS; for a reversal stock they are LOG-EXPOSURE TRIMS
    //  consumed immediately below at stage 8. That asymmetry is inherent to the two
    //  processes: a negative gets graded when it is printed, a slide has only the
    //  exposure the photographer gave it.
    //
    //  Solved rather than guessed. The obvious choice of offset equal to the
    //  mid-scale density puts grey wherever the print curve happens to cross zero,
    //  which on a typical print stock is about two per cent display luminance -
    //  roughly three stops too dark.
    //
    //  Stage 13 re-solves the print offsets against the neutral density a dupe chain
    //  leaves behind, so what is computed here is used at stage 8 and at stage 8b and
    //  not carried further.
    // -----------------------------------------------------------------------
    HighPrecType anchor[3];

    ALGO_PROF_MARK("--   anchor solve");
    AlgoSolveAnchors(profile, pPrint,
                     static_cast<HighPrecType>(algoCtrl.greyTarget),
                     static_cast<HighPrecType>(algoCtrl.couplerScale),
                     static_cast<HighPrecType>(algoCtrl.scannerSpecular),
                     anchor);

    // -----------------------------------------------------------------------
    //  Reciprocity failure: three numbers for the whole frame.
    //
    //  \warning WIRED 2026-09-01, AND UNTIL THEN THIS ENGINE HAD NO RECIPROCITY
    //  MODEL AT ALL. AlgoReciprocity.hpp had been written, documented and left
    //  unincluded - no translation unit pulled it in - while film_sim applied
    //  the correction on every render, so the reference and the plugin
    //  disagreed on every long exposure of every stock that publishes a table.
    //  AlgoControl.hpp had already specified `exposureTimeS` for it and marked
    //  the stage PENDING.
    //
    //  There are no pixels to walk: every correction on file is a function of
    //  TIME alone, so this resolves to three constants that stage 8 adds to the
    //  logarithm it is computing anyway. Zero per-pixel cost.
    //
    //  INERT AT THE DEFAULT. exposureTimeS is 0 unless the caller states a
    //  shutter time, the shift is then exactly zero, and adding a floating zero
    //  is the identity - so every render made before this call existed is
    //  reproduced bit for bit.
    // -----------------------------------------------------------------------
    HighPrecType recipShift[3];

    ALGO_PROF_MARK("--   reciprocity");
    AlgoReciprocityLogShift(profile,
                            static_cast<HighPrecType>(algoCtrl.exposureTimeS),
                            recipShift);

    // -----------------------------------------------------------------------
    // 8. CHARACTERISTIC CURVE                              S07 -> S08
    //                                                      log E -> Scr_LogE_*
    //
    //    Exposure to optical density: the Hurter and Driffield curve, and the stage
    //    that makes film film. A difference of two softplus ramps, which produces the
    //    whole real topology from one expression and is MONOTONIC BY CONSTRUCTION - a
    //    non-monotonic patch in the shoulder solarises every highlight.
    //
    //    Everything from here on is in the DENSITY domain.
    // -----------------------------------------------------------------------
    ALGO_PROF_MARK("08   characteristic curve");
    AlgoStage08_CharacteristicCurve(s07R, s07G, s07B,
                                    s08R, s08G, s08B,
                                    logER, logEG, logEB,
                                    sizeX, sizeY, pitch,
                                    profile, anchor, recipShift);

    // -----------------------------------------------------------------------
    // 8b. INTERIMAGE EFFECTS                               S08 -> S08b
    //
    //     The VERTICAL half of the DIR coupler chemistry: inhibitor released while
    //     one layer develops diffuses into its neighbours and suppresses them, so
    //     each layer's effective exposure depends on what the other two are doing.
    //
    //     Referenced to the mid-grey density, which is what makes it a colour effect
    //     rather than a tone effect: on a neutral the correction vanishes and the
    //     grey scale is untouched, while a saturated colour separates further.
    //     Saturation rising WITHOUT gamma rising, which no per-channel curve can
    //     produce.
    //
    //     Reads Scr_LogE_*, because the correction re-enters in the exposure domain
    //     and density cannot be inverted back through the shoulder. An implicit
    //     equation, solved by fixed-point iteration.
    // -----------------------------------------------------------------------
    ALGO_PROF_MARK("08b  interimage");
    AlgoStage08b_Interimage(s08R, s08G, s08B,
                            s08bR, s08bG, s08bB,
                            logER, logEG, logEB,
                            scrLuma, scrField, scrFieldLo,
                            sizeX, sizeY, pitch,
                            profile, anchor);

    // -----------------------------------------------------------------------
    // 9. DIR COUPLER LATERAL EFFECTS                       S08b -> S09
    //
    //    The LATERAL half of the same chemistry: inhibitor spreading sideways within
    //    a layer. Two components - a long-range term pushing each layer away from the
    //    locally blurred mean of all three, and a short-range adjacency term
    //    sharpening edges.
    //
    //    After the curve rather than before, because the inhibitor is released BY
    //    development in proportion to the dye being formed, so its amount is a
    //    function of density and not of exposure.
    // -----------------------------------------------------------------------
    ALGO_PROF_MARK("09   DIR coupler lateral");
    AlgoStage09_DirCoupler(s08bR, s08bG, s08bB,
                           s09R, s09G, s09B,
                           scrDbar, scrDbarBlur, scrBlurA, scrBlurB,
                           sizeX, sizeY, pitch,
                           profile, algoCtrl, pxPerMm);

    // -----------------------------------------------------------------------
    // 9b. NEGATIVE-SIDE DEFECTS                            S09 -> S09b
    //
    //     Damage baked into the film emulsion. Three particulate classes are
    //     modelled: fine dust, coarse debris, hair and fibres - all of them the
    //     EMBEDDED population, the share that was pressed into swollen gelatin
    //     during drying and is therefore part of the negative for ever. Loose
    //     one-frame dirt and gate dirt are machine-side and belong to stage 16.
    //
    //     Scratches, processing mottle and drying marks are negative-side too and
    //     are not applied yet; they have their own controls.
    //
    //     It sits BEFORE the print at 13 on purpose, so the dupe chain and the print
    //     curve act on it exactly as they act on the picture. Camera-original damage
    //     applied after the print would be sharper and cleaner than the image around
    //     it, which reads as a digital overlay immediately.
    //
    //     Inert unless algoCtrl.filmDamageEnabled is true, and then still inert
    //     unless a class level is non-zero, so the clean pipeline pays one branch.
    // -----------------------------------------------------------------------
    ALGO_PROF_MARK("09b  negative defects");
    AlgoStage09b_NegativeDefects(s09R, s09G, s09B,
                                 s09bR, s09bG, s09bB,
                                 sizeX, sizeY, pitch,
                                 profile, algoCtrl,
                                 negWidthMm, negHeightMm, framePitchMm, pxPerMm,
                                 frameIndex, frameRate, ALGO_SALT_NEG_DEFECTS);

    // -----------------------------------------------------------------------
    // 9c. BROMIDE DRAG                                     S09b -> S09b, in place
    //
    //     The processing MACHINE's directional restraint, and the only stage in the
    //     chain that belongs to the lab rather than to the film. Bromide released by
    //     development restrains further development; the film moves through the
    //     bath, so the loaded solution is dragged along the transport axis and keeps
    //     restraining where it lands. The result is a one-sided streak trailing
    //     every dense region, aligned with the web - the archival lab-print
    //     signature.
    //
    //     ⚠ NOT A SECOND HELPING OF STAGE 9. That is inhibitor diffusing inside the
    //     gelatin: isotropic, tens of micrometres, a property of the coating. This
    //     is loaded developer sliding across the outside of it: one-sided,
    //     millimetres to centimetres, a property of the machine, and it therefore
    //     reads ProcessingSpec and not FilmProfile's own fields. The two are
    //     adjacent because the density stage 9 leaves is what releases the bromide.
    //
    //     ⚠ IN PLACE, like stage 12b. It is a multiply by a scalar field, so a
    //     destination plane set would buy a copy and nothing else. Scr_Dbar and
    //     Scr_DbarBlur are both dead after stage 9 and carry the source field and
    //     the accumulator.
    //
    //     Inert on every stock in the database: no document in the corpus
    //     quantifies a bromide gradient, so every BromideDragSpec ships at zero and
    //     the stage returns on its first branch. See queue row C23.
    // -----------------------------------------------------------------------
    ALGO_PROF_MARK("09c  bromide drag");
    (void)AlgoStage09c_BromideDrag(s09bR, s09bG, s09bB,
                                   memHandler.Scr_Dbar, memHandler.Scr_DbarBlur,
                                   sizeX, sizeY, pitch,
                                   profile, pxPerMm);

    // -----------------------------------------------------------------------
    // 10. SCAN MTF AND MISREGISTRATION                     S09b -> S10
    //
    //     The scanner's optical MTF is the PRE-SAMPLING filter: the lens sits between
    //     film and sensor, so it band-limits both image and grain before either is
    //     sampled. That is why this stage precedes grain, and why the same transfer
    //     is handed to the grain stage as its band limit - the only way to stop fine
    //     grain aliasing onto the pixel grid.
    //
    //     Registration error softens colour edges the way every real scan is
    //     softened. A few micrometres is invisible as a shift and very visible as an
    //     absence.
    // -----------------------------------------------------------------------
    ALGO_PROF_MARK("10   scan MTF + misreg");
    AlgoStage10_ScanMtf(s09bR, s09bG, s09bB,
                        s10R, s10G, s10B,
                        scrBlurA, scrBlurB,
                        sizeX, sizeY, pitch,
                        profile, algoCtrl, scanF50, pxPerMm,
                        frameIndex, ALGO_SALT_MISREG);

    // -----------------------------------------------------------------------
    // 10b. NARROW-GAUGE EDGE FOG                           S10 -> S10b
    //
    //      Additive density near the physical film edges. A GAUGE matter, not an era
    //      matter: Standard 8 is 16 mm slit down the middle after processing, so its
    //      picture sits at the film edge with no trimmed margin, while 35 mm margins
    //      carry the perforations and are cut away.
    // -----------------------------------------------------------------------
    ALGO_PROF_MARK("10b  edge fog");
    AlgoStage10b_EdgeFog(s10R, s10G, s10B,
                         s10bR, s10bG, s10bB,
                         sizeX, sizeY, pitch,
                         profile, algoCtrl, negWidthMm);

    // -----------------------------------------------------------------------
    // 11. GRAIN                                            S10b -> S11
    //
    //     A developed emulsion is a countable population of discrete crystals, so
    //     density is a random variable. Amplitude goes as the SQUARE ROOT of
    //     developed density, which puts grain strongest in the mid tones and weakest
    //     in the deep shadows - a constant-amplitude noise model gets that backwards.
    //
    //     One emulsion means one field: a monochrome stock and an additive-colour
    //     stock both get a single shared field, and three independent ones would
    //     produce coloured speckle on a black-and-white image.
    // -----------------------------------------------------------------------
    ALGO_PROF_MARK("11   grain");
    AlgoStage11_Grain(s10bR, s10bG, s10bB,
                      s11R, s11G, s11B,
                      scrDbar, scrDbarBlur, scrBlurA,
                      scrGrainR, scrGrainG, scrGrainB,
                      sizeX, sizeY, pitch,
                      profile, algoCtrl, scanSigmaPx, pxPerMm,
                      hasMosaic, frameIndex, ALGO_SALT_GRAIN);

    // -----------------------------------------------------------------------
    // 12. DYE IMPURITY AND SCANNER CROSSTALK               S11 -> S12
    //
    //     Real cyan dye absorbs some green and a little blue, and so on round the
    //     three. SUBTRACTIVE, with row sums near one - the opposite convention to the
    //     taking matrix at stage 2b, and the two must never be interchanged.
    //
    //     After grain, because a scanner reads dye through its own filters, so
    //     whatever is in the dye layers - image and grain alike - is mixed by the same
    //     matrix. Grain acquires a slight chromatic correlation, and it should.
    // -----------------------------------------------------------------------
    ALGO_PROF_MARK("12   dye impurity");
    AlgoStage12_DyeImpurity(s11R, s11G, s11B,
                            s12R, s12G, s12B,
                            sizeX, sizeY, pitch, profile);

    // -----------------------------------------------------------------------
    // 12b. CALLIER: THE DENSITY THE READER'S OPTICS SEE      S12 -> S12 (in place)
    //
    //     Developed silver scatters the measuring beam. An integrating sphere
    //     collects the scattered light and reads the diffuse density every curve in
    //     the database is expressed in; a condenser or a point source loses it
    //     outside its acceptance angle and reads a HIGHER density, steepening the
    //     whole tone scale. That is why a silver negative printed on a condenser
    //     enlarger is contrastier than the same negative on a diffusion enlarger at
    //     the same paper grade.
    //
    //     HERE, at the boundary between the developed negative and everything that
    //     reads it, because BOTH readers in this chain are affected - an optical
    //     printer with a condenser and a scanner with a directed source see the same
    //     steepened density. Before stage 13 rather than after, since the print
    //     stage's own curve must act on what its optics actually see.
    //
    //     ⚠ IN PLACE ON THE STAGE-12 PLANES, no scratch buffer: the operation is
    //     pointwise, so stage 13 reads the corrected values from the same pointers.
    //     ⚠ AND IT IS INERT AT THE DEFAULT. scannerSpecular is 0, the factor is
    //     exactly 1.0, the stage returns before touching a pixel, and every render
    //     made before this stage existed is reproduced. It is also inert at ANY
    //     setting for the colour stocks, whose dye images carry Q = 1.0.
    //
    //     The anchor solve above sees the same factor, and it has to: a lab that
    //     switches to a condenser head RE-TIMES the print. Leave the solve blind
    //     and the control shifts mid grey by more than it changes contrast.
    // -----------------------------------------------------------------------
    {
        const AlgoType s12Dmin[3] =
        {
            static_cast<AlgoType>(profile.curves.r.dmin),
            static_cast<AlgoType>(profile.curves.g.dmin),
            static_cast<AlgoType>(profile.curves.b.dmin)
        };

        ALGO_PROF_MARK("12b  Callier");
        AlgoStage12b_Callier(s12R, s12G, s12B,
                             sizeX, sizeY, pitch, s12Dmin, profile,
                             static_cast<HighPrecType>(algoCtrl.scannerSpecular));
    }

    // -----------------------------------------------------------------------
    // 13. DUPLICATION GENERATIONS, THEN THE PRINT          S12 -> S13
    //
    //     Nobody ever projected a camera negative. A release print is three or four
    //     generations away, and each intermediate adds its own grain and its own MTF
    //     loss. Generations come in PAIRS so polarity returns to negative before the
    //     final print, and duplicating stock runs at gamma 1.0 so contrast does not
    //     compound while grain and softness do.
    //
    //     Within a generation the order is blur, then curve, then THIS generation's
    //     own grain - because that grain is created in this emulsion and so is not
    //     blurred by this emulsion's optics. Adding it before the blur makes a long
    //     dupe chain come out cleaner than a short one, which is backwards.
    //
    //     Reports which curve set produced the output, because stage 14 needs those
    //     endpoints and they differ between the print and reversal paths.
    // -----------------------------------------------------------------------
    film::RGBCurves finalCurves = profile.curves;

    ALGO_PROF_MARK("13   duplication + print");
    AlgoStage13_Duplication(s12R, s12G, s12B,
                            s13R, s13G, s13B,
                            logER, logEG, logEB,
                            scrDbar, scrDbarBlur, scrBlurA, scrGrainR,
                            sizeX, sizeY, pitch,
                            profile, algoCtrl,
                            pPrint, pDupe,
                            scanSigmaPx, pxPerMm,
                            frameIndex, ALGO_SALT_DUPE,
                            finalCurves);

    // -----------------------------------------------------------------------
    // 14. PRINT GRAIN, THEN TRANSMITTANCE                  S13 -> S14
    //
    //     Print grain is created AFTER the print curve, so unlike negative grain it
    //     is not compressed by the shoulder - a difference in highlight grain
    //     behaviour that a single-stage grain model cannot produce at all.
    //
    //     Then density to display-linear transmittance, normalised against the final
    //     curve's own endpoints. That normalisation is what the anchor solves aimed
    //     at, so the two must use the same expression.
    //
    //     LEAVES THE DENSITY DOMAIN. Everything after this is transmittance.
    // -----------------------------------------------------------------------
    ALGO_PROF_MARK("14   print grain + transmit");
    AlgoStage14_Transmittance(s13R, s13G, s13B,
                              s14R, s14G, s14B,
                              scrDbar, scrDbarBlur, scrBlurA, scrGrainR,
                              sizeX, sizeY, pitch,
                              profile, algoCtrl, pPrint, finalCurves,
                              profile.isReversal(),
                              scanSigmaPx, pxPerMm,
                              frameIndex, ALGO_SALT_PRINT_GRAIN);

    // -----------------------------------------------------------------------
    // 14b. RESEAU RECONSTRUCTION, THEN RESIDUAL BASE TINT  S14 -> S14b
    //
    //      Projection sends light back through the same filter grid in register, and
    //      only here does a single monochrome record become colour again. At the very
    //      end because on a real additive print the grid physically sits in the light
    //      path AT VIEWING TIME, downstream of everything.
    //
    //      The grid is recomputed from the same helper stage 7 used, so no mask
    //      travels between the two stages and none can go stale.
    // -----------------------------------------------------------------------
    ALGO_PROF_MARK("14b  reseau reconstruct");
    AlgoStage14b_ReseauReconstruct(s14R, s14G, s14B,
                                   s14bR, s14bG, s14bB,
                                   scrDbar, scrDbarBlur, scrField, scrFieldLo,
                                   scrBlurA,
                                   sizeX, sizeY, pitch,
                                   profile, algoCtrl, pxPerMm);

    // -----------------------------------------------------------------------
    // 14c. SILVER IMAGE TONE                               S14b -> S14c
    //
    //      Developed silver is not spectrally neutral, which is why a
    //      black-and-white print is almost never actually grey. Weighted by output
    //      level, because the effect is strongest where there is least silver.
    //
    //      AFTER the anchor solves on purpose: base_tint is compensated by the
    //      printer-light solve, which is why base_tint cannot tint a monochrome stock
    //      at all. This stage is downstream of that solve and therefore survives it.
    // -----------------------------------------------------------------------
    ALGO_PROF_MARK("14c  silver tone");
    AlgoStage14c_SilverTone(s14bR, s14bG, s14bB,
                            s14cR, s14cG, s14cB,
                            sizeX, sizeY, pitch, profile);

    // -----------------------------------------------------------------------
    // 15. GATE WEAVE                                       S14c -> S15
    //
    //     Successive frames do not sit in exactly the same place in the gate. A
    //     sub-pixel whole-frame translation, and one of the strongest cues that
    //     something was shot and projected on film - stronger than grain, because the
    //     eye tracks motion better than it judges texture.
    //
    //     !! STATUS CORRECTED 2026-08-28: this stage IS implemented and active.
    //     It was labelled a stub here long after it stopped being one. Skipping
    //     is decided INSIDE the stage, by AlgoControls::filmDamageEnabled and
    //     then by its own class levels -- the call site is unconditional. It is here rather than with the
    //     negative-side stages because weave happens at PROJECTION, to the finished
    //     positive, and its time base is the projection rate.
    // -----------------------------------------------------------------------
    ALGO_PROF_MARK("15   gate weave");
    AlgoStage15_GateWeave(s14cR, s14cG, s14cB,
                          s15R, s15G, s15B,
                          scrBlurA, scrBlurB,
                          sizeX, sizeY, pitch,
                          profile, algoCtrl,
                          negWidthMm, negHeightMm, pxPerMm,
                          frameIndex, frameRate, ALGO_SALT_WEAVE);

    // -----------------------------------------------------------------------
    // 16. GATE-SIDE DEFECTS                                S15 -> S16
    //
    //     Dirt on the gate, a hair in the light path, a projector scratch, a splice
    //     in the release print. Applied to the finished POSITIVE, so it passes through
    //     no curve and is duplicated by no generation - the exact opposite of stage 9b
    //     on every count, including polarity, which is why they are two stages either
    //     side of the print.
    //
    //     !! STATUS CORRECTED 2026-08-28: this stage IS implemented and active.
    //     It was labelled a stub here long after it stopped being one. Skipping
    //     is decided INSIDE the stage, by AlgoControls::filmDamageEnabled and
    //     then by its own class levels -- the call site is unconditional.
    // -----------------------------------------------------------------------
    ALGO_PROF_MARK("16   gate defects");
    AlgoStage16_GateDefects(s15R, s15G, s15B,
                            s16R, s16G, s16B,
                            sizeX, sizeY, pitch,
                            profile, algoCtrl,
                            negWidthMm, negHeightMm, pxPerMm,
                            frameIndex, frameRate, ALGO_SALT_GATE_DEFECTS);

    // -----------------------------------------------------------------------
    // 17. THE SINGLE FINAL CLAMP                           S16 -> S17 -> Dst
    //
    //     Every earlier stage left its output unclamped at the top, deliberately: the
    //     characteristic curve's shoulder needs real highlight information above the
    //     nominal white point in order to roll it off, and clamping early is what
    //     makes highlights look digital. The range is imposed exactly once, here.
    //
    //     The floors at zero in earlier stages are a different matter - a negative
    //     exposure and a negative optical density are physically meaningless, and both
    //     the logarithm at stage 8 and the exponentiation at stage 14 require
    //     non-negative input.
    //
    //     Also the one place AlgoType narrows back to ImgType, matching the single
    //     widening at stage 2.
    // -----------------------------------------------------------------------
    ALGO_PROF_MARK("17   final clamp");
    AlgoStage17_FinalClamp(s16R, s16G, s16B,
                           s17R, s17G, s17B,
                           oR, oG, oB,
                           sizeX, sizeY, pitch);

    // Close the last segment and write the table. Nothing after this point, so the
    // report is the final act of the frame and cannot be attributed to a stage.
    ALGO_PROF_END();

    return;
}
