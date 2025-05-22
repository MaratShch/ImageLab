#include "cct_interface.hpp"
#include "CctLimits.hpp"


using namespace AlgoCCT;

CctHandleF32::CctHandleF32()
{
    m_Lut1.clear();
    m_Lut2.clear();

    m_isValid = false;
    return;
}

CctHandleF32::~CctHandleF32()
{
    Deinitialize();
    return;
}

bool CctHandleF32::Initialize(void)
{
    m_Lut1 = initLut (CCT_Limits::waveLenMin, CCT_Limits::waveLenMax, CCT_Limits::waveLenStep, CCT_Limits::cctMin, CCT_Limits::cctMax, CCT_Limits::cctStep, observer_CIE_1931);
    m_Lut2 = initLut (CCT_Limits::waveLenMin, CCT_Limits::waveLenMax, CCT_Limits::waveLenStep, CCT_Limits::cctMin, CCT_Limits::cctMax, CCT_Limits::cctStep, observer_CIE_1964);

    m_isValid = true;

    return true;
}


void CctHandleF32::Deinitialize(void)
{
    m_isValid = false;

    if (m_Lut1.size())
        m_Lut1.clear();
    if (m_Lut2.size())
        m_Lut2.clear();
    return;
}


std::pair<float, float> CctHandleF32::ComputeCct (const std::pair<float, float>& uv, eCOLOR_OBSERVER observer) const
{
    float Cct = 0.f;
    float Duv = 0.f;

    const float u = uv.first;
    const float v = uv.second;

    if (observer_CIE_1931 == observer)
    {
        // we working with LUT1
    }
    else if (observer_CIE_1964 == observer)
    {
        // we working with LUT2
    }
        // invalid or non-supported observer
    return std::make_pair (Cct, Duv);
}
