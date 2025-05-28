#ifndef __IMAGELAB2_CCT_INTERAFCE_LIBRARY_MODULE__
#define __IMAGELAB2_CCT_INTERAFCE_LIBRARY_MODULE__

#include "ClassRestrictions.hpp"
#include "CctLut.hpp"
#include <utility>
#include <atomic>
#include <vector>


namespace AlgoCCT
{

    class CctHandleF32 final
    {
        public:
            CctHandleF32();
            ~CctHandleF32();

            CLASS_NON_COPYABLE(CctHandleF32);
            CLASS_NON_MOVABLE (CctHandleF32);

            void Deinitialize(void);

            std::pair<float /* CCT */, float /* Duv */> ComputeCct (const std::pair<float, float>& uv, eCOLOR_OBSERVER observer);

#ifdef _DEBUG
            std::vector<CCT_LUT_Entry<float>>& getLut_CIE_1931(void) { return m_Lut1; }
            std::vector<CCT_LUT_Entry<float>>& getLut_CIE_1964(void) { return m_Lut2; }
#else
            std::vector<CCT_LUT_Entry<float>> getLut_CIE_1931(void) { return{}; }
            std::vector<CCT_LUT_Entry<float>> getLut_CIE_1964(void) { return{}; }
#endif

        private:

            bool InitializeLut1(void);
            bool InitializeLut2(void);

            bool cct_compute (const float& u, const float& v, float& cct, float& duv, const std::vector<CCT_LUT_Entry<float>>& lut);
            bool refine (const float& u, const float& v, float& cct, float& duv, const std::vector<CCT_LUT_Entry<float>>& lut);

            // Helper functions for refine computations
            float Distance (const float& v1, const float& v2) noexcept { return v1 * v1 + v2 * v2; }

            static std::atomic<bool> g_flagL1;
            static std::atomic<bool> g_flagL2;

            // CCT Lut for CIE_1931 2 degrees observer
            std::vector<CCT_LUT_Entry<float>> m_Lut1;

            // CCT Lut for CIE_1964 10 degrees observer
            std::vector<CCT_LUT_Entry<float>> m_Lut2;

    }; // class CctHandleF32

}; // namespace AlgoCCT

#endif // __IMAGELAB2_CCT_INTERAFCE_LIBRARY_MODULE__