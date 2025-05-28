#include "FastAriphmetics.hpp"
#include "cct_interface.hpp"

using namespace AlgoCCT;

// Placeholder structure for spline coefficients.
// The actual content of this struct would depend on the chosen spline implementation.
struct SplineCoeffs
{
    std::vector<float> x_knots;
    std::vector<float> y_values;
    bool empty = true;
};


bool CctHandleF32::refine (const float& u0, const float& v0, float& cct, float& duv, const std::vector<CCT_LUT_Entry<float>>& lut)
{
    const size_t lutSize = lut.size();

    // build vector of distances
    std::vector<float> vec_distance(lutSize);
    for (size_t i = 0u; i < lutSize; i++)
    {
        const float du = u0 - lut[i].u;
        const float dv = v0 - lut[i].v;
        vec_distance[i] = Distance (du, dv);
    }

    auto min_it = std::min_element(vec_distance.begin(), vec_distance.end());
    size_t idx_min = static_cast<size_t>(std::distance(vec_distance.begin(), min_it));

    int32_t range_start_idx = static_cast<int32_t>(idx_min) - 2;
    range_start_idx = std::max(0, range_start_idx);

    int32_t range_end_idx = static_cast<int32_t>(idx_min) + 2;
    range_end_idx = std::min(static_cast<int32_t>(lut.size() - 1), range_end_idx);

    if (range_start_idx > range_end_idx)
    {
        range_start_idx = 0;
        range_end_idx = static_cast<int32_t>(lut.size() - 1);
    }

    const size_t expected_seg_size = static_cast<size_t>(range_end_idx - range_start_idx + 1);
    std::vector<float> u_seg, v_seg, cct_seg;
    u_seg.reserve  (expected_seg_size);
    v_seg.reserve  (expected_seg_size);
    cct_seg.reserve(expected_seg_size);

    for (int32_t i = range_start_idx; i <= range_end_idx; ++i)
    {
        // static_cast<size_t>(i) is safe because range_start_idx/range_end_idx are bounded by lut.size()
        u_seg.push_back  (lut[i].u);
        v_seg.push_back  (lut[i].v);
        cct_seg.push_back(lut[i].cct);
    }


    return true;
}