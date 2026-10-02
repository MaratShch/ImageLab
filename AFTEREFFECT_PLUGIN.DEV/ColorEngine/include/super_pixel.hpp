#pragma once

#include <cstdint>
#include <cmath>
#include <type_traits>

// Aggregated ("super") pixel: weighted-mean linear RGB of the pixels that
// survive the illuminant-evidence exclusion rules.
// Templated on field type; recommended TOut = double (single value, feeds the
// double-precision CCT/Duv solve, so no reason to store it in float).
template <typename T>
struct SuperPixel
{
    T r;
    T g;
    T b;
};


template <typename T>
struct CctDuv
{
    T cct;
    T duv;
};