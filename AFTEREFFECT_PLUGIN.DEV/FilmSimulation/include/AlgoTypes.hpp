#pragma once

// ---------------------------------------------------------------------------
//  AlgoTypes.hpp  --  AVX2 BUILD
//
//  STANDALONE AND SELF-CONTAINED. This file defines every numeric type and
//  constant the vector engine needs. It includes no other AlgoTypes header and
//  selects nothing at build time.
//
//  ⚠ THE SCALAR AND AVX2 PROJECTS EACH SHIP THEIR OWN AlgoTypes.hpp.
//    scalar project      Algorithm/Scalar/include/AlgoTypes.hpp  AlgoType = double
//    AVX2 project        Algorithm/AVX2/include/AlgoTypes.hpp AlgoType = float
//
//  They are two separate files with the same name, one per project, picked up
//  by each project's own include path. Never put both on one include path, and
//  never include one from the other.
//
//  ---------------------------------------------------------------------------
//  WHY float
//  ---------------------------------------------------------------------------
//
//  Eight lanes of a 256-bit register per step instead of four. That is the
//  whole reason the vector path exists, and the vector translation units assert
//  sizeof(AlgoType) == 4 for themselves - so a build that reaches them with
//  double fails loudly at the first one rather than producing a slow,
//  silently-different result.
//
//  ---------------------------------------------------------------------------
//  THE EXCEPTION, AND IT CUTS BOTH WAYS
//  ---------------------------------------------------------------------------
//
//  Specific computations in the vector sources are written in double BECAUSE
//  THEY NEED TO BE, and they are load-bearing:
//
//    - the frame-setup solvers in Algo_08_Sim.cpp - a sixty-step bisection
//      whose bracket shrinks below float resolution long before it finishes;
//    - the blackbody radiance in Algo_03_Sim.cpp, whose intermediates span
//      roughly sixty decades;
//    - the coating-field web offset in Algo_04_Sim.cpp, which grows without
//      bound along a clip, so single precision loses its low bits;
//    - the frame-mean veil accumulation behind veiling flare, which sets the
//      black floor of the whole frame.
//
//  DO NOT convert those to float to make this file's rule look tidy. Equally,
//  DO NOT introduce new, unnecessary double arithmetic into the AVX2 hot paths
//  to make results match the scalar build - that gives up the throughput this
//  path exists for. Any new double here needs a numerical reason recorded at
//  the site. HighPrecType below is the type to reach for.
//
//  ---------------------------------------------------------------------------
//  ⚠ THE ONLY BUILD-SYSTEM REQUIREMENT: THE INSTRUCTION SET
//  ---------------------------------------------------------------------------
//
//    MSVC        /arch:AVX2   on the AVX2 configuration.
//    GCC/Clang   -mavx2 -mfma
//
//  That enables the instructions. WHICH type this project computes in is
//  decided by this file being the AlgoTypes.hpp on the include path, not by any
//  define you have to remember to set.
//
//  ⚠ THE SHARED TRANSLATION UNITS ARE THE REASON THIS MATTERS. The vector
//  project compiles its own copies of AlgorithmMain.cpp, AlgoDefectField.cpp
//  and AlgoSpectralSensitivity.cpp, and those size planes by sizeof(AlgoType)
//  and take AlgoType* parameters. Because they are compiled inside THIS project
//  they see THIS header, so one alias cannot end up with two layouts in a single
//  link. The benign version of that mistake is a ten-symbol link failure; the
//  malign version is a silent ABI mismatch.
//
//  PRECISION AND QUALITY ARE ORTHOGONAL. Precision is set here, by which
//  project you are building. QUALITY is set by QualityPolicy /
//  AlgoControls::simMode. A quality preset must NEVER change precision.
//
//  AVX2/FMA only. NO AVX-512 without explicit approval.
// ---------------------------------------------------------------------------

// Project-wide primitives, included unconditionally as required by the project
// coding standard.
#include "Common.hpp"
#include "CompileTimeUtils.hpp"

#include <cstddef>      // std::size_t
#include <cstdint>      // fixed-width integer types
#include <type_traits>  // std::is_floating_point


// ---------------------------------------------------------------------------
//  ⚠ THIS HEADER DECLARES THE AVX2 CONTRACT ITSELF. NOTHING NEEDS TO BE SET IN
//  THE BUILD SYSTEM.
//
//  Which of the two AlgoTypes.hpp you get is decided by WHICH PROJECT'S include
//  DIRECTORY is on the search path - AVX2/include or Scalar/include - and by
//  nothing else. There is no detection, no auto-configuration and no
//  preprocessor switch for you to set or forget.
//
//  A handful of shared headers still need to know which path they are in:
//  AlgoCurveLut.hpp exposes its vector evaluator only under ALGO_TARGET_AVX2.
//  So this file DEFINES that symbol rather than demanding the build system
//  supply it. Defining it here means the include path and the symbol can never
//  disagree, which is the failure a project-wide define invites: set it on the
//  wrong configuration and the sources compile at one width while the shared
//  translation units compile at the other, and the link may even succeed.
//
//  It is defined only if absent, so a build that already sets it project-wide
//  keeps working unchanged and sees no redefinition warning.
// ---------------------------------------------------------------------------
#if !defined(ALGO_TARGET_AVX2)
#  define ALGO_TARGET_AVX2 1
#endif


/// Arithmetic type of every algorithmic computation inside the engine.
/// Vector path: float. See the block comment above for the double exceptions
/// that are retained inside the vector sources and must stay.
using AlgoType = float;

static_assert(sizeof(AlgoType) == 4,
              "PRECISION RULE: the AVX2 path computes in float and AlgoType "
              "must be 32-bit.");

// ---------------------------------------------------------------------------
//  Storage type of image samples at the engine boundary.
//
//  float in BOTH paths. Storing image data in double would double the bandwidth
//  for accuracy the host's own 8/10/16-bit output cannot carry.
// ---------------------------------------------------------------------------
using ImgType = float;


// ---------------------------------------------------------------------------
//  HighPrecType - setup-time arithmetic that must NOT follow AlgoType.
//
//  Fixed at double in BOTH paths, on purpose. A few once-per-frame computations
//  have a dynamic range single precision cannot hold, and they would produce
//  WRONG numbers rather than merely less accurate ones if they tracked AlgoType
//  down to float:
//
//    - blackbody radiance for colour balance: the fifth power of a wavelength
//      in metres is of order 1e-32 and the exponential's argument reaches ~53,
//      so intermediates span roughly sixty decades;
//    - frame-mean accumulation: a flat single-precision accumulator over two
//      million samples loses the low bits of the running total.
//
//  These are once or a few times per frame, so double costs nothing measurable.
//  The independence from AlgoType is exactly what lets the AVX2 path be float
//  throughout without giving up accuracy where accuracy is genuinely required.
// ---------------------------------------------------------------------------
using HighPrecType = double;

static_assert(sizeof(HighPrecType) == 8,
              "HighPrecType must be double in BOTH paths, independent of "
              "AlgoType");


// ---------------------------------------------------------------------------
//  Guards. These fire at EVERY include site.
//
//  Both types must be floating point: every stage assumes fractional values,
//  negative intermediates and a usable dynamic range.
//
//  AlgoType must be at least as wide as ImgType. Computing in a narrower type
//  than the storage would discard precision the input already carries.
// ---------------------------------------------------------------------------
static_assert(std::is_floating_point<ImgType>::value,
              "ImgType must be a floating point type");

static_assert(std::is_floating_point<AlgoType>::value,
              "AlgoType must be a floating point type");

static_assert(sizeof(AlgoType) >= sizeof(ImgType),
              "AlgoType must be at least as wide as ImgType: computing in a "
              "narrower type than the storage discards input precision");


// ---------------------------------------------------------------------------
//  ALIGNMENT
//
//  32 bytes: the AVX2 register width (256 bits / 8 = 32) and thus the alignment
//  required by the aligned load and store forms. Every plane base address AND
//  every row start is aligned to it from the outset. The scalar path does not
//  need it, but retro-fitting alignment later means re-deriving every stride in
//  the engine and is a reliable source of subtle bugs.
//
//  The element count spanning one quantum differs by type, which is why the two
//  are derived separately rather than assumed equal:
//
//      ImgType  = float            ->  32 / 4 = 8 elements
//      AlgoType = double (scalar)  ->  32 / 8 = 4 elements
//      AlgoType = float  (AVX2)    ->  32 / 4 = 8 elements
//
//  A stride computed with the wrong one of these is the single most likely way
//  to break alignment while leaving the image looking correct.
// ---------------------------------------------------------------------------
constexpr std::size_t ALGO_ALIGN_BYTES = 32u;

/// Elements of ImgType per 32-byte quantum. float: 8.
constexpr std::size_t IMG_ALIGN_ELEMS = ALGO_ALIGN_BYTES / sizeof(ImgType);

/// Elements of AlgoType per 32-byte quantum. Scalar double: 4. AVX2 float: 8.
constexpr std::size_t ALGO_ALIGN_ELEMS = ALGO_ALIGN_BYTES / sizeof(AlgoType);


// ---------------------------------------------------------------------------
//  Convenience constants in AlgoType.
//
//  Written once here so no stage has to spell out a cast for values it needs
//  constantly. Using these instead of a literal also means a change of AlgoType
//  cannot leave a stray double literal behind to force a conversion in an inner
//  loop - which on the AVX2 path is a real cost, not a style point.
// ---------------------------------------------------------------------------
constexpr AlgoType ALGO_ZERO = static_cast<AlgoType>(0);
constexpr AlgoType ALGO_ONE  = static_cast<AlgoType>(1);
constexpr AlgoType ALGO_HALF = static_cast<AlgoType>(0.5);


// ---------------------------------------------------------------------------
//  ALGO_VECTOR_HINT -- portable "this loop has no cross-iteration dependence"
//
//  Common.hpp defines __VECTOR_ALIGNED__, but its non-MSVC branch expands to
//  __pragma(loop(ivdep)). __pragma is an MSVC extension: GCC and Clang reject it
//  outright, so any translation unit using that macro fails to build there. The
//  engine therefore uses its own, defined here so that fixing the shared header
//  stays the owner's decision.
//
//    Intel     vector always + vector aligned -- strongest hint Intel accepts.
//    MSVC      loop(ivdep) -- the only vectorisation hint MSVC has.
//    Clang     clang loop vectorize(enable). Clang does NOT understand
//              "GCC ivdep", so it MUST be tested BEFORE __GNUC__ -- Clang
//              defines __GNUC__ too, and the wrong order silently feeds it a
//              pragma it warns about and ignores.
//    GCC       GCC ivdep, available since GCC 4.9.
//    other     empty. The hint is advisory; correctness never depends on it.
//
//  _Pragma rather than #pragma because only _Pragma can appear in a macro
//  expansion. The macro must sit immediately before the `for` statement.
// ---------------------------------------------------------------------------
#if defined(__INTEL_COMPILER)
  #define ALGO_VECTOR_HINT __pragma(vector always) __pragma(vector aligned)
#elif defined(_MSC_VER)
  #define ALGO_VECTOR_HINT __pragma(loop(ivdep))
#elif defined(__clang__)
  #define ALGO_VECTOR_HINT _Pragma("clang loop vectorize(enable)")
#elif defined(__GNUC__)
  #define ALGO_VECTOR_HINT _Pragma("GCC ivdep")
#else
  #define ALGO_VECTOR_HINT
#endif

static_assert(ALGO_ALIGN_ELEMS == 8u,
              "AVX2 path: 32 bytes must span 8 AlgoType elements");
