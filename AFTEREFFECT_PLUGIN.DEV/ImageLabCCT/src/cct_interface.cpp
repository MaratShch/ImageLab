#include "cct_interface.hpp"
#include "CctLimits.hpp"


using namespace AlgoCCT;

CctHandleF32::CctHandleF32()
{
    return;
}

CctHandleF32::~CctHandleF32()
{
    return;
}

bool CctHandleF32::Initialize(void)
{
    m_Lut = initLut (CCT_Limits::waveLenMin, CCT_Limits::waveLenMax, CCT_Limits::waveLenStep, CCT_Limits::cctMin, CCT_Limits::cctMax, CCT_Limits::cctStep);

    return true;
}


void CctHandleF32::Deinitialize(void)
{
    m_Lut.clear();
    return;
}
