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
            CLASS_NON_MOVABLE(CctHandleF32);

            bool Initialize(void);
            void Deinitialize(void);

            bool DeferredInit(void) { return false; };

            std::pair<float /* CCT */, float /* Duv */> ComputeCct (const std::pair<float, float>& uv, eCOLOR_OBSERVER observer) const;

#ifdef _DEBUG
            std::vector<CCT_LUT_Entry<float>>& getLut_CIE_1931(void) { return m_Lut1; }
            std::vector<CCT_LUT_Entry<float>>& getLut_CIE_1964(void) { return m_Lut2; }
#else
            std::vector<CCT_LUT_Entry<float>> getLut_CIE_1931(void) { return{}; }
            std::vector<CCT_LUT_Entry<float>> getLut_CIE_1964(void) { return{}; }
#endif

        private:
            // flag
            std::atomic<bool> m_isValid;

            // CCT Lut for CIE_1931 2 degrees observer
            std::vector<CCT_LUT_Entry<float>> m_Lut1;

            // CCT Lut for CIE_1964 10 degrees observer
            std::vector<CCT_LUT_Entry<float>> m_Lut2;

    };


};

#endif // __IMAGELAB2_CCT_INTERAFCE_LIBRARY_MODULE__