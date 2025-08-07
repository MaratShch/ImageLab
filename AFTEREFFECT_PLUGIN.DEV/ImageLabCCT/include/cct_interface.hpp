#ifndef __IMAGELAB2_CCT_INTERAFCE_LIBRARY_MODULE__
#define __IMAGELAB2_CCT_INTERAFCE_LIBRARY_MODULE__

#include "ClassRestrictions.hpp"
#include "CctLut.hpp"
#include "CctLimits.hpp"
#include <utility>
#include <atomic>
#include <vector>
#include <tuple>
#include <functional>

namespace AlgoCCT
{

    // Placeholder structure for spline coefficients.
    struct SplineCoeffs
    {
        std::vector<float> x_knots;
        std::vector<float> y_values;
        bool empty = true;
    };


    class CctHandleF32 final
    {
        public:
            CctHandleF32();
            ~CctHandleF32();

            CLASS_NON_COPYABLE(CctHandleF32);
            CLASS_NON_MOVABLE (CctHandleF32);

            void Deinitialize(void);

            std::pair<float /* CCT */, float /* Duv */> ComputeCct (const std::pair<float, float>& uv, eCOLOR_OBSERVER observer);

            std::pair<float /* CCT */, float /* Duv */> getPlanckianUV (float cct, float Duv, eCOLOR_OBSERVER observer);
            std::pair<float /* CCT */, float /* Duv */> getPlanckianUV (const std::pair<float, float>& cct_Duv, eCOLOR_OBSERVER observer);

#ifdef _DEBUG
            std::vector<CCT_LUT_Entry<float>>& getLut_CIE_1931(void) { return m_Lut1; }
            std::vector<CCT_LUT_Entry<float>>& getLut_CIE_1964(void) { return m_Lut2; }
#else
            std::vector<CCT_LUT_Entry<float>> getLut_CIE_1931(void) { return{}; }
            std::vector<CCT_LUT_Entry<float>> getLut_CIE_1964(void) { return{}; }
#endif

            float getCctMin (void) const noexcept { return CCT_Limits::cctMin; }
            float getCctMax (void) const noexcept { return CCT_Limits::cctMax; }

        private:


            bool InitializeLut1(void);
            bool InitializeLut2(void);

            bool cct_compute (const float& u, const float& v, float& cct, float& duv, const std::vector<CCT_LUT_Entry<float>>& lut);
            bool refine (const float& u, const float& v, float& cct, float& duv, const std::vector<CCT_LUT_Entry<float>>& lut);
            
            // Helper functions for refine computations
            SplineCoeffs spline_impl(const std::vector<float>& x_in, const std::vector<float>& y_in);
            float ppval_impl (const SplineCoeffs& fit_coeffs, float t_eval);
            float fminbnd_impl (const std::function<float(float)>& objective_func, float lower_bound, float upper_bound, float tolerance);
            float Distance (const float& v1, const float& v2) noexcept { return v1 * v1 + v2 * v2; }

            std::vector<float> linspace (float start, float end, int32_t num_points)
            {
                std::vector<float> vec;
                if (num_points <= 0)
                    return vec;
                if (num_points == 1)
                {
                    vec.push_back(start);
                    return vec;
                }
                const float step = (end - start) / static_cast<float>(num_points - 1);
                for (int32_t i = 0; i < num_points; ++i)
                    vec.push_back(start + static_cast<float>(i) * step);
                return vec;
            }

            static std::atomic<bool> g_flagL1;
            static std::atomic<bool> g_flagL2;

            // CCT Lut for CIE_1931 2 degrees observer
            std::vector<CCT_LUT_Entry<float>> m_Lut1;

            // CCT Lut for CIE_1964 10 degrees observer
            std::vector<CCT_LUT_Entry<float>> m_Lut2;

    }; // class CctHandleF32

}; // namespace AlgoCCT

#endif // __IMAGELAB2_CCT_INTERAFCE_LIBRARY_MODULE__