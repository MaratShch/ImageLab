#pragma once

// ---------------------------------------------------------------------------
//  AlgoTypes.hpp
//
//  THE SINGLE PLACE where the engine's numeric representations are chosen.
//
//  Everything in the film simulation derives its arithmetic and storage types
//  from the two aliases below. Nothing else in the engine writes 'float' or
//  'double' for image data or for algorithmic computation. The purpose is to make
//  a precision experiment a one-line change: switch an alias, rebuild, and
//  measure performance, numerical agreement and image quality without touching a
//  single line of algorithm code.
//
//  THE TWO AXES ARE INDEPENDENT, AND THAT IS THE POINT
//
//    ImgType  - how image SAMPLES are STORED at the engine boundary. This is the
//               pixel format the host hands over and expects back. It governs
//               memory footprint and bandwidth, which at 4K dominate the frame
//               time far more than arithmetic does.
//
//    AlgoType - how the engine COMPUTES. It governs numerical accuracy through
//               the pipeline, and it matters most in the two accuracy-critical
//               stages: the characteristic curve, which evaluates a logarithm and
//               an exponential per sample per channel, and the interimage fixed
//               point, which re-evaluates that curve several times over.
//
//  Keeping them separate allows all four combinations to be measured. Collapsing
//  them into one alias would make it impossible to tell a bandwidth effect from a
//  precision effect, which is exactly the distinction worth measuring.
//
//  CURRENT SETTING AND THE REASON FOR IT
//
//    ImgType  = float   - what the host supplies and what the eventual AVX2 path
//                         will process eight-wide. Storing image data in double
//                         would double the bandwidth for accuracy that 16-bit
//                         output cannot carry.
//
//    AlgoType = double  - deliberately generous for the first implementation. The
//                         scalar path is validated against the reference model in
//                         double, and only once that agrees is AlgoType switched
//                         to float and the comparison repeated. Starting in float
//                         would leave any disagreement ambiguous between a coding
//                         error and a precision limit.
// ---------------------------------------------------------------------------

// Project-wide primitives, included unconditionally as required by the project
// coding standard.
#include "Common.hpp"
#include "CompileTimeUtils.hpp"

#include <cstddef>      // std::size_t
#include <cstdint>      // fixed-width integer types
#include <type_traits>  // std::is_floating_point


// ---------------------------------------------------------------------------
//  THE TWO ALIASES. Change these and nothing else.
// ---------------------------------------------------------------------------

/// Storage type of image samples at the engine boundary.
using ImgType = float;

/// Arithmetic type of every algorithmic computation inside the engine.
using AlgoType = float;//double;


// ---------------------------------------------------------------------------
//  Guards on the choices above.
//
//  Both must be floating point: every stage assumes fractional values, negative
//  intermediates and a usable dynamic range, none of which an integer type
//  provides.
//
//  AlgoType must be at least as wide as ImgType. Computing in a narrower type
//  than the one used for storage would discard precision the input already
//  carries, which is never a configuration worth measuring - it is simply wrong.
//  Catching it here turns a subtle accuracy loss into a compile error.
// ---------------------------------------------------------------------------
static_assert(std::is_floating_point<ImgType>::value,
              "ImgType must be a floating point type");

static_assert(std::is_floating_point<AlgoType>::value,
              "AlgoType must be a floating point type");

static_assert(sizeof(AlgoType) >= sizeof(ImgType),
              "AlgoType must be at least as wide as ImgType: computing in a "
              "narrower type than the storage discards input precision");


// ---------------------------------------------------------------------------
//  HighPrecType - setup-time arithmetic that must NOT follow AlgoType.
//
//  Fixed at double, on purpose, and this is not an oversight.
//
//  A small number of once-per-frame computations have a dynamic range that
//  single precision cannot hold, and they would silently produce wrong numbers
//  rather than merely less accurate ones if they tracked AlgoType down to float.
//  The clearest case is the blackbody radiance used for colour balance: the fifth
//  power of a wavelength in metres is of the order 1e-32, and the exponential's
//  argument reaches about 53, so the intermediate products span roughly sixty
//  decades. Frame-mean accumulation is the other: a flat single-precision
//  accumulator over two million samples loses the low bits of the running total
//  once it has grown large relative to each addend.
//
//  These are all once or a few times per frame, so fixing them at double costs
//  nothing measurable. The alias exists so that the intent is explicit at each
//  site rather than looking like a place where somebody forgot to use AlgoType.
// ---------------------------------------------------------------------------
using HighPrecType = double;


// ---------------------------------------------------------------------------
//  ALIGNMENT
//
//  32 bytes: the AVX2 register width (256 bits / 8 bits per byte = 32) and thus
//  the alignment required by the aligned load and store instruction forms. Every
//  plane base address AND every row start is aligned to it from the outset. The
//  scalar path does not need it, but retro-fitting alignment later means
//  re-deriving every stride in the engine and is a reliable source of subtle
//  bugs.
//
//  The element count spanning one quantum differs by type, which is why the two
//  are derived separately rather than assumed equal:
//
//      ImgType  = float   ->  32 / 4 = 8 elements
//      AlgoType = double  ->  32 / 8 = 4 elements
//
//  A stride computed with the wrong one of these is the single most likely way to
//  break alignment while leaving the image looking correct.
// ---------------------------------------------------------------------------
constexpr std::size_t ALGO_ALIGN_BYTES = 32u;

/// Elements of ImgType per 32-byte quantum. float: 8.
constexpr std::size_t IMG_ALIGN_ELEMS = ALGO_ALIGN_BYTES / sizeof(ImgType);

/// Elements of AlgoType per 32-byte quantum. double: 4.
constexpr std::size_t ALGO_ALIGN_ELEMS = ALGO_ALIGN_BYTES / sizeof(AlgoType);


// ---------------------------------------------------------------------------
//  Convenience constants in AlgoType.
//
//  Written once here so that no stage has to spell out a cast for the two values
//  it needs constantly. Using these instead of a literal also means a change of
//  AlgoType cannot leave a stray double literal behind to force a conversion in
//  an inner loop.
// ---------------------------------------------------------------------------
constexpr AlgoType ALGO_ZERO = static_cast<AlgoType>(0);
constexpr AlgoType ALGO_ONE  = static_cast<AlgoType>(1);
constexpr AlgoType ALGO_HALF = static_cast<AlgoType>(0.5);


// ---------------------------------------------------------------------------
//  ALGO_VECTOR_HINT -- portable "this loop has no cross-iteration dependence"
//
//  Common.hpp defines __VECTOR_ALIGNED__, but its non-MSVC branch expands to
//  __pragma(loop(ivdep)). __pragma is an MSVC extension: GCC and Clang reject it
//  outright with "'__pragma' was not declared in this scope", so any translation
//  unit using the macro fails to build on those compilers.
//
//  The engine therefore uses its own macro. It is defined here rather than in
//  Common.hpp so that fixing the shared header stays the owner's decision, and so
//  the engine keeps building on every compiler either way.
//
//  Per-compiler spelling, and why each is what it is:
//
//    Intel     __pragma(vector always) + (vector aligned) -- the strongest hint
//              Intel accepts, matching what Common.hpp already does there.
//    MSVC      __pragma(loop(ivdep)) -- the only vectorisation hint MSVC has.
//    Clang     _Pragma("clang loop vectorize(enable)"). Clang does NOT understand
//              "GCC ivdep", so it must be tested BEFORE __GNUC__ -- Clang defines
//              __GNUC__ as well, and getting the order wrong silently feeds it a
//              pragma it will warn about and ignore.
//    GCC       _Pragma("GCC ivdep"), available since GCC 4.9.
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
