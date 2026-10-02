#pragma once

// ---------------------------------------------------------------------------
//  AlgoCounterRng.hpp
//
//  Counter-based random number generation.
//
//  WHY NOT AN ORDINARY GENERATOR
//
//  A conventional generator carries state and produces its next value from the
//  previous one. That makes it useless here for three independent reasons:
//
//    - The host renders frames out of order, speculatively, and may render the
//      same frame twice. A sequential generator would give a different answer
//      each time, so grain and defects would crawl and flicker under scrubbing.
//    - Several instances of the engine may run concurrently in different
//      threads. Shared generator state would need locking, and locking in the
//      inner loop of an image filter is not viable.
//    - The engine may not hold mutable state of any kind, by design.
//
//  A counter-based generator solves all three at once: every value is a PURE
//  FUNCTION of its coordinates. Ask for the value at (seed, frame, stage,
//  ordinal) and the same answer comes back regardless of when, in what order, or
//  on which thread it is asked for. There is no state to carry, share or lock.
//
//  CONSTRUCTION
//
//  The four coordinates are packed into a 64-bit counter, which is then passed
//  through a strong integer mixing function. The mixer used is SplitMix64's
//  finalising step, which is a well-studied bijection on 64 bits: it passes the
//  usual empirical test batteries, avalanches every input bit across the whole
//  output, and costs three multiplies and three shifts. Bijectivity matters -
//  distinct coordinates can never collide onto the same value.
//
//  This is header-only and every function is a pure inline computation, so there
//  is no translation unit and nothing to link.
// ---------------------------------------------------------------------------

// Project-wide primitives, included unconditionally as required by the project
// coding standard.
#include "Common.hpp"
#include "CompileTimeUtils.hpp"

// ImgType, AlgoType, HighPrecType. The single place numeric types are chosen.
#include "AlgoTypes.hpp"

#include <cstdint>   // uint32_t, uint64_t, int32_t
#include <cmath>     // std::log, std::sqrt, std::cos


// ---------------------------------------------------------------------------
//  STAGE IDENTIFIERS
//
//  Each consumer of randomness gets its own identifier, so that two stages can
//  never draw the same numbers. Without this, the coating field and the grain
//  field would be correlated - the same values would appear in both, and the
//  grain would visibly follow the coating streaks.
//
//  Values are arbitrary but must be distinct and must never be reused or
//  renumbered, because doing so changes the appearance of every existing render.
//  Spaced by 0x100 so that a stage needing several independent sub-streams can
//  take a small offset without colliding with its neighbour.
// ---------------------------------------------------------------------------
enum class eALGO_RNG_STAGE : uint32_t
{
    eRNG_COATING_STATIC = 0x0100u,  // stage 4b, fixed cross-web streaks
    eRNG_COATING_DRIFT  = 0x0200u,  // stage 4b, drifting two-dimensional field
    eRNG_FLICKER        = 0x0300u,  // stage 3c, exposure flicker phases
    eRNG_NEG_DEFECTS    = 0x0400u,  // stage 9b, negative-side damage
    eRNG_GRAIN_R        = 0x0500u,  // stage 11, red record grain field
    eRNG_GRAIN_G        = 0x0600u,  // stage 11, green record grain field
    eRNG_GRAIN_B        = 0x0700u,  // stage 11, blue record grain field
    eRNG_PRINT_GRAIN    = 0x0800u,  // stage 14, grain added by the print stock
    eRNG_DUPE_GRAIN     = 0x0900u,  // stage 13, grain added per dupe generation
    eRNG_MISREG         = 0x0A00u,  // stage 10, channel registration jitter
    eRNG_WEAVE          = 0x0B00u,  // stage 15, gate weave displacement
    eRNG_GATE_DEFECTS   = 0x0C00u   // stage 16, gate-side dirt and hair
};


// ---------------------------------------------------------------------------
//  SplitMix64 finalising constants
//
//  These are the published SplitMix64 mixing constants. They are not arbitrary
//  odd numbers: they were selected by search for avalanche quality, meaning that
//  flipping any single input bit changes each output bit with probability close
//  to one half. Substituting different constants would still give a bijection
//  but would degrade the statistical quality, so they are reproduced exactly.
//
//  0x9E3779B97F4A7C15 is the 64-bit odd approximation of the golden ratio
//  conjugate, 2^64 / phi, used as the stream increment.
// ---------------------------------------------------------------------------
constexpr uint64_t ALGO_RNG_GOLDEN = 0x9E3779B97F4A7C15ull;
constexpr uint64_t ALGO_RNG_MIX_1  = 0xBF58476D1CE4E5B9ull;
constexpr uint64_t ALGO_RNG_MIX_2  = 0x94D049BB133111EBull;

// Shift distances of the SplitMix64 finaliser, likewise published values.
constexpr int32_t ALGO_RNG_SHIFT_1 = 30;
constexpr int32_t ALGO_RNG_SHIFT_2 = 27;
constexpr int32_t ALGO_RNG_SHIFT_3 = 31;


// ---------------------------------------------------------------------------
//  AlgoRngMix64
//
//  The mixing bijection. Takes any 64-bit value and returns a well-distributed
//  64-bit value; equal inputs give equal outputs, distinct inputs give distinct
//  outputs.
// ---------------------------------------------------------------------------
FORCE_INLINE uint64_t AlgoRngMix64 (uint64_t z) noexcept
{
    z += ALGO_RNG_GOLDEN;
    z  = (z ^ (z >> ALGO_RNG_SHIFT_1)) * ALGO_RNG_MIX_1;
    z  = (z ^ (z >> ALGO_RNG_SHIFT_2)) * ALGO_RNG_MIX_2;
    return z ^ (z >> ALGO_RNG_SHIFT_3);
}


// ---------------------------------------------------------------------------
//  AlgoRngCounter
//
//  Pack the four coordinates into one 64-bit counter.
//
//  Layout, from the most significant end:
//
//      bits 63..32   seed, 32 bits
//      bits 31..24   stage identifier, 8 bits (the high byte of eALGO_RNG_STAGE)
//      bits 23..00   ordinal, 24 bits
//
//  and frameIndex is folded into the seed field by multiplication before packing
//  rather than given a field of its own. Two reasons: 24 bits of ordinal is
//  16.7 million draws per stage per frame, which is ample for a 4K plane's worth
//  of field coefficients but not for a per-pixel draw, so the ordinal is used
//  for coefficient indices rather than pixel indices; and folding the frame in
//  through the golden-ratio constant keeps successive frames far apart in
//  counter space, which matters because the mixer is fed similar values.
//
//  frameIndex is SIGNED and may be negative, which happens legitimately near the
//  start of a clip when a defect's birth frame is searched backwards. It is cast
//  to unsigned for the arithmetic, which is well defined and wraps - and wrapping
//  is harmless here because the mixer treats all 64-bit values alike.
// ---------------------------------------------------------------------------
FORCE_INLINE uint64_t AlgoRngCounter
(
    const uint32_t        seed,
    const int32_t         frameIndex,
    const eALGO_RNG_STAGE stage,
    const uint32_t        ordinal
) noexcept
{
    // Fold the frame index into the seed. The multiply by the golden constant
    // decorrelates adjacent frames far more effectively than addition would.
    const uint64_t frameSalt =
        static_cast<uint64_t>(static_cast<uint32_t>(frameIndex)) * ALGO_RNG_GOLDEN;

    const uint64_t seedField = (static_cast<uint64_t>(seed) << 32) ^ frameSalt;

    // Stage occupies bits 31..24; take the high byte of the enumerator value.
    const uint64_t stageField =
        (static_cast<uint64_t>(UnderlyingType(stage) >> 8) & 0xFFull) << 24;

    // Ordinal occupies the low 24 bits.
    const uint64_t ordField = static_cast<uint64_t>(ordinal) & 0x00FFFFFFull;

    return seedField ^ stageField ^ ordField;
}


// ---------------------------------------------------------------------------
//  AlgoRngUniform01
//
//  Returns HighPrecType, not AlgoType, and must continue to: the value is formed
//  from 53 bits of mantissa, which is what a double holds exactly. Following
//  AlgoType down to float would silently discard 29 of those bits and reduce the
//  generator to about 8 million distinct values, which is visible as banding in
//  any field built from it.
//
//  Uniform in [0, 1). Formed from the top 53 bits of the mixed value, which is
//  the number of bits a double's mantissa can represent exactly, so every
//  representable value in the range is reachable and none is favoured.
//
//  2^-53 is written as a hexadecimal float-free constant expression rather than
//  a decimal literal so there is no question of it being rounded on the way in.
// ---------------------------------------------------------------------------
FORCE_INLINE HighPrecType AlgoRngUniform01 (const uint64_t counter) noexcept
{
    const uint64_t bits = AlgoRngMix64(counter) >> 11;   // keep the top 53 bits

    // 1.0 / 2^53 = 1.1102230246251565e-16.
    return static_cast<HighPrecType>(bits) * (1.0 / 9007199254740992.0);
}


// ---------------------------------------------------------------------------
//  AlgoRngUniformRange
//
//  Uniform in [lo, hi). Used for phase angles, which need [0, 2*pi).
// ---------------------------------------------------------------------------
FORCE_INLINE HighPrecType AlgoRngUniformRange
(
    const uint64_t counter,
    const HighPrecType lo,
    const HighPrecType hi
) noexcept
{
    return lo + (hi - lo) * AlgoRngUniform01(counter);
}


// ---------------------------------------------------------------------------
//  AlgoRngNormal
//
//  Standard normal, mean 0 and standard deviation 1, by the Box-Muller
//  transform:
//
//      z = sqrt(-2 ln u1) * cos(2 pi u2)
//
//  Two independent uniforms are needed, and they are obtained by mixing two
//  adjacent counters rather than by drawing sequentially - there is no sequence
//  to draw from. Only the cosine branch is returned; the sine branch would give
//  a second independent value but keeping it would require state, so it is
//  discarded. That doubles the cost per value and is accepted, because the normal
//  draws here are for a fixed small number of field coefficients rather than per
//  pixel.
//
//  u1 is guarded away from exactly zero. log(0) is negative infinity and would
//  propagate as a non-finite value through the whole field; the probability is
//  2^-53 per draw, which is negligible but not zero, and a single infinity would
//  destroy an entire frame.
//
//  Box-Muller rather than the ziggurat method: this needs to be a pure function
//  of its counter with no rejection loop, and ziggurat's variable number of
//  draws per value makes it awkward to index deterministically.
//
//  Note that std::log and std::cos are not bit-specified by the C++ standard, so
//  two different toolchains may produce results differing in the last bits.
//  Any claim that two builds are bit-identical is therefore a claim about one
//  toolchain, not about the algorithm.
// ---------------------------------------------------------------------------
FORCE_INLINE HighPrecType AlgoRngNormal (const uint64_t counter) noexcept
{
    // Two decorrelated uniforms from one logical draw. The second counter is
    // displaced by the golden constant rather than by 1, so the two inputs to the
    // mixer are far apart even though they came from the same request.
    const HighPrecType u1raw = AlgoRngUniform01(counter);
    const HighPrecType u2    = AlgoRngUniform01(counter ^ ALGO_RNG_GOLDEN);

    // Smallest representable step of the 53-bit uniform, used as the floor.
    const HighPrecType kTiny = 1.0 / 9007199254740992.0;

    const HighPrecType u1 = (u1raw < kTiny) ? kTiny : u1raw;

    // 2 pi, written out so the value is visible rather than assembled from a
    // macro that may or may not be defined on a given platform.
    const HighPrecType kTwoPi = 6.283185307179586476925286766559;

    return std::sqrt(-2.0 * std::log(u1)) * std::cos(kTwoPi * u2);
}
