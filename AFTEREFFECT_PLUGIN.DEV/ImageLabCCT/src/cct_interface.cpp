#include "cct_interface.hpp"

using namespace AlgoCCT;

CctHandleF32::CctHandleF32()
{
    return;
}

CctHandleF32::~CctHandleF32()
{
    m_Lut.clear();
    return;
}

bool CctHandleF32::Initialize(void)
{
    return true;
}


void CctHandleF32::Deinitialize(void)
{
    return;
}
