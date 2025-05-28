#include "cct_interface.hpp"
#include "FastAriphmetics.hpp"


using namespace AlgoCCT;


SplineCoeffs CctHandleF32::spline_impl(const std::vector<float>& x_in, const std::vector<float>& y_in)
{
    SplineCoeffs coeffs;

    if (x_in.size() == y_in.size() || x_in.size() >= static_cast<size_t>(2))
    {
        coeffs.empty = false;
        coeffs.x_knots = x_in;
        coeffs.y_values = y_in;
    }
    // This function needs a full implementation of cubic spline interpolation.....

    return coeffs;
}


// Evaluates a piecewise polynomial (spline) at given points
float CctHandleF32::ppval_impl (const SplineCoeffs& fit_coeffs, float t_eval)
{
    // This function needs a full implementation of spline evaluation....

    const auto& x = fit_coeffs.x_knots;
    const auto& y = fit_coeffs.y_values;

    if (t_eval <= x.front())
        return y.front();
    
    if (t_eval >= x.back())
        return y.back();

    auto it = std::upper_bound(x.begin(), x.end(), t_eval);
    if (it == x.begin())
        return y.front();

    size_t idx1 = static_cast<size_t>(std::distance(x.begin(), it) - 1);
    size_t idx2 = idx1 + 1;

    if (idx2 >= x.size())
        return y.back();

    const float x1 = x[idx1], y1 = y[idx1];
    const float x2 = x[idx2], y2 = y[idx2];

    if (std::abs(x1 - x2) < std::numeric_limits<float>::epsilon())
        return y1;

    // Linear interpolation:
    const float val = y1 + (y2 - y1) * (t_eval - x1) / (x2 - x1);
    
    return val;
}

// Finds a local minimum of a bounded scalar function
float CctHandleF32::fminbnd_impl (const std::function<float(float)>& objective_func, float lower_bound, float upper_bound, float tolerance)
{
    int num_steps = 10000;
    float min_val = std::numeric_limits<float>::infinity();
    float t_star = lower_bound;
    if (upper_bound <= lower_bound) return lower_bound; // Or handle error
    float step_size = (upper_bound - lower_bound) / static_cast<float>(num_steps);

    // This function needs a full implementation of a bounded minimization algorithm: Golden Section Search, Brent's Method ...

    return t_star;
}



bool CctHandleF32::refine (const float& u0, const float& v0, float& cct, float& duv, const std::vector<CCT_LUT_Entry<float>>& lut)
{
    bool bRet = false;
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
    if (expected_seg_size >= static_cast<size_t>(2))
    {
        std::vector<float> u_seg, v_seg, cct_seg;
        u_seg.reserve(expected_seg_size);
        v_seg.reserve(expected_seg_size);
        cct_seg.reserve(expected_seg_size);

        for (int32_t i = range_start_idx; i <= range_end_idx; ++i)
        {
            u_seg.push_back  (lut[i].u);
            v_seg.push_back  (lut[i].v);
            cct_seg.push_back(lut[i].cct);
        } // for (int32_t i = range_start_idx; i <= range_end_idx; ++i)

        const int32_t num_seg_points = static_cast<int32_t>(u_seg.size());
        std::vector<float> t_norm = linspace (0.0f, 1.0f, num_seg_points);

        SplineCoeffs u_fit = spline_impl  (t_norm, u_seg);
        SplineCoeffs v_fit = spline_impl  (t_norm, v_seg);
        SplineCoeffs cct_fit = spline_impl(t_norm, cct_seg);

        auto objective = [&](float tval) -> float
        {
            float u_t = ppval_impl(u_fit, tval);
            float v_t = ppval_impl(v_fit, tval);
            float du = u_t - u0;
            float dv = v_t - v0;
            return Distance (du, dv);
        };

        constexpr float fTolerance = 1e-7f;
        float t_star = fminbnd_impl (objective, 0.0f, 1.0f, fTolerance);

        float u_proj = ppval_impl (u_fit, t_star);
        float v_proj = ppval_impl (v_fit, t_star);
        const float refinedCCT = ppval_impl (cct_fit, t_star);

        constexpr float delta = 1e-5f;

        float u_at_t_star_plus_delta, u_at_t_star_minus_delta;
        float v_at_t_star_plus_delta, v_at_t_star_minus_delta;
        float effective_denominator_h;

        if (t_star <= delta)
        {
            u_at_t_star_minus_delta = ppval_impl(u_fit, t_star);
            u_at_t_star_plus_delta  = ppval_impl(u_fit, std::min(1.0f, t_star + delta));
            v_at_t_star_minus_delta = ppval_impl(v_fit, t_star);
            v_at_t_star_plus_delta  = ppval_impl(v_fit, std::min(1.0f, t_star + delta));
            effective_denominator_h = delta;
        }
        else if (t_star >= 1.0f - delta)
        {
            u_at_t_star_minus_delta = ppval_impl(u_fit, std::max(0.0f, t_star - delta));
            u_at_t_star_plus_delta  = ppval_impl(u_fit, t_star);
            v_at_t_star_minus_delta = ppval_impl(v_fit, std::max(0.0f, t_star - delta));
            v_at_t_star_plus_delta  = ppval_impl(v_fit, t_star);
            effective_denominator_h = delta;
        }
        else
        {
            u_at_t_star_minus_delta = ppval_impl(u_fit, t_star - delta);
            u_at_t_star_plus_delta  = ppval_impl(u_fit, t_star + delta);
            v_at_t_star_minus_delta = ppval_impl(v_fit, t_star - delta);
            v_at_t_star_plus_delta  = ppval_impl(v_fit, t_star + delta);
            effective_denominator_h = 2.0f * delta;
        }

        float u_tangent = (std::abs(effective_denominator_h) > std::numeric_limits<float>::epsilon()) ?
            (u_at_t_star_plus_delta - u_at_t_star_minus_delta) / effective_denominator_h : 0.0f;
        float v_tangent = (std::abs(effective_denominator_h) > std::numeric_limits<float>::epsilon()) ?
            (v_at_t_star_plus_delta - v_at_t_star_minus_delta) / effective_denominator_h : 0.0f;

        float norm_t = FastCompute::VectorNorm (u_tangent, v_tangent);
        if (norm_t > std::numeric_limits<float>::epsilon())
        {
            u_tangent /= norm_t;
            v_tangent /= norm_t;
        }
        else {
            u_tangent = 1.0f;
            v_tangent = 0.0f;
        }

        float vec_u = u0 - u_proj;
        float vec_v = v0 - v_proj;

        float dot_val = FastCompute::Dot (vec_u, vec_v, u_tangent, v_tangent);
        float orth_vec_u = vec_u - dot_val * u_tangent;
        float orth_vec_v = vec_v - dot_val * v_tangent;

        float norm_orthogonal_vec = FastCompute::VectorNorm (orth_vec_u, orth_vec_v);
        float det_val = FastCompute::Determinant2x2 (u_tangent, v_tangent, vec_u, vec_v);

        const float refinedDuv = static_cast<float>(FastCompute::Sign(det_val)) * norm_orthogonal_vec;

        cct = refinedCCT;
        duv = refinedDuv;

        bRet = true;
    } // if (expected_seg_size > static_cast<size_t>(2))

    return bRet;
}
