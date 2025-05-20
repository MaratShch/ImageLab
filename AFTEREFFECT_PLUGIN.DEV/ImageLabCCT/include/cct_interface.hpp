#ifndef __IMAGELAB2_CCT_INTERAFCE_LIBRARY_MODULE__
#define __IMAGELAB2_CCT_INTERAFCE_LIBRARY_MODULE__

#include "ClassRestrictions.hpp"
#include "CctLut.hpp"
#include <utility>

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

            std::pair<float /* CCT */, float /* Duv */> ComputeCct (std::pair<float, float>& uv, int32_t observer);

        private:
            // CCT Lut for CIE_1931 2 degrees observer
            std::vector<CCT_LUT_Entry<float>> m_Lut1;

            // CCT Lut for CIE_1964 10 degrees observer
            std::vector<CCT_LUT_Entry<float>> m_Lut2;

    };


};

#endif // __IMAGELAB2_CCT_INTERAFCE_LIBRARY_MODULE__