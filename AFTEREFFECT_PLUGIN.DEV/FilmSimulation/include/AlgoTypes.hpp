#pragma once

// ===========================================================================
//  Shared numeric policy for the film simulation core.
//  C++14, MSVC 2015 SP3 / GCC 13 compatible. No C++17 or later features.
// ===========================================================================

#include <cstdint>
#include <cstddef>

#include "Common.hpp"       // RESTRICT

// ---------------------------------------------------------------------------
// Working precision.
//
// Start at double, validate against the Python reference to 6 decimals, THEN
// switch to float and re-validate. Scalar float alone buys only 1.0-1.3x --
// the real gain arrives with AVX2, where float doubles the lane count from 4
// to 8. Switching early just loses the reference match with nothing to show.
//
// Note the test harness scales its tolerance with this typedef: double is
// expected to match the float64 reference essentially bit-exactly, while float
// legitimately differs by a few ULP (epsilon is 1.19e-7).
// ---------------------------------------------------------------------------
using AlgoType = double;
// using AlgoType = float;      // stage 2 of the plan

// ---------------------------------------------------------------------------
// 32-byte alignment: AVX2 wants it, and it costs nothing to require it now.
// Requiring it later means revisiting every allocation.
//
// DELETE THIS if Common.hpp already carries an alignment constant -- no point
// having two.
// ---------------------------------------------------------------------------
constexpr std::size_t ALGO_ALIGN = 32u;

// 18 % grey. The pipeline's single reference point -- every anchor solve exists
// to put this value back where it started.
constexpr double ALGO_MID_GREY = 0.18;
