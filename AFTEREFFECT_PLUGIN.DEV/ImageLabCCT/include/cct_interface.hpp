#ifndef __IMAGELAB2_CCT_INTERAFCE_LIBRARY_MODULE__
#define __IMAGELAB2_CCT_INTERAFCE_LIBRARY_MODULE__

#include "ClassRestrictions.hpp"
#include "CctLut.hpp"

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

        private:
            std::vector<CCT_LUT_Entry<float>> m_Lut;

    };


};

#endif // __IMAGELAB2_CCT_INTERAFCE_LIBRARY_MODULE__