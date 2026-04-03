#pragma once

#include <memory>
#include "Common.hpp"
#include "AefxDevPatch.hpp"
#include "PaintSparseMatrix.hpp"


A_long bw_image2cocircularity_graph_AVX2_flat
(
    const float* RESTRICT eigX,
    const float* RESTRICT eigY,
    A_long* RESTRICT pI,
    A_long* RESTRICT pJ,
    size_t max_edges,      // Matches the perfectly aligned size_t in your MemHandler
    A_long width,
    A_long height,
    float coCirc,
    float coCone,
    A_long radius
);