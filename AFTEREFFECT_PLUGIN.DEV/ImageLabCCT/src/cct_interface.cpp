#include "cct_interface.hpp"
#include "CctLimits.hpp"
#include <cmath>

using namespace AlgoCCT;

std::atomic<bool> CctHandleF32::g_flagL1 = false;
std::atomic<bool> CctHandleF32::g_flagL2 = false;


CctHandleF32::CctHandleF32()
{
    g_flagL1 = false;
    g_flagL2 = false;

    m_Lut1.clear();
    m_Lut2.clear();
    return;
}

CctHandleF32::~CctHandleF32()
{
    Deinitialize();
    return;
}

bool CctHandleF32::InitializeLut1(void)
{
    m_Lut1 = initLut(CCT_Limits::waveLenMin, CCT_Limits::waveLenMax, CCT_Limits::waveLenStep, CCT_Limits::cctMin, CCT_Limits::cctMax, CCT_Limits::cctStep, observer_CIE_1931);
    g_flagL1 = true;
    return true;
}

bool CctHandleF32::InitializeLut2(void)
{
    m_Lut2 = initLut(CCT_Limits::waveLenMin, CCT_Limits::waveLenMax, CCT_Limits::waveLenStep, CCT_Limits::cctMin, CCT_Limits::cctMax, CCT_Limits::cctStep, observer_CIE_1964);
    g_flagL2 = true;
    return true;
}


void CctHandleF32::Deinitialize(void)
{
    g_flagL1 = g_flagL2 = false;
    if (m_Lut1.size())
        m_Lut1.clear();
    if (m_Lut2.size())
        m_Lut2.clear();
    return;
}


bool CctHandleF32::cct_compute(const float& u, const float& v, float& cct, float& duv, const std::vector<CCT_LUT_Entry<float>>& lut)
{
    cct = 0.f;
    duv = 0.f;

    auto euclideanDistance = [&](const float u1, const float v1, const float u2, const float v2) -> float
    {
        const float du_val = u1 - u2;
        const float dv_val = v1 - v2;
        return std::sqrtf(du_val * du_val + dv_val * dv_val);
    };

    int32_t low = 0;
    int32_t high = static_cast<int32_t>(lut.size()) - 1;
    int32_t best_idx = 0;

    while (low <= high)
    {
        int32_t mid_idx = low + (high - low) / 2;

        const float dist = euclideanDistance(u, v, lut[mid_idx].u, lut[mid_idx].v);
        float prevDist = std::numeric_limits<float>::infinity();
        float nextDist = std::numeric_limits<float>::infinity();

        if (mid_idx > 0)
            prevDist = euclideanDistance(u, v, lut[mid_idx - 1].u, lut[mid_idx - 1].v);
        if (mid_idx < static_cast<int32_t>(lut.size()) - 1)
            nextDist = euclideanDistance(u, v, lut[mid_idx + 1].u, lut[mid_idx + 1].v);

        if (dist <= prevDist && dist <= nextDist)
        {
            best_idx = mid_idx;
            break;
        }

        if (mid_idx > 0 && prevDist < dist)
            high = mid_idx - 1;
        else
            low = mid_idx + 1;
    }

    if (low > high && best_idx == 0)
    { // If loop terminated without break, and best_idx wasn't updated
      // This means the minimum is likely at an edge or near where low/high crossed
        int32_t idx1 = std::max(0, high);
        int32_t idx2 = std::min(static_cast<int32_t>(lut.size()) - 1, low);

        // Ensure idx1 is valid if high became -1
        if (idx1 < 0) idx1 = 0;
        if (idx1 >= static_cast<int32_t>(lut.size())) 
            idx1 = static_cast<int32_t>(lut.size()) - 1;

        // Ensure idx2 is valid
        if (idx2 < 0) idx2 = 0;
        if (idx2 >= static_cast<int32_t>(lut.size())) 
            idx2 = static_cast<int32_t>(lut.size()) - 1;

        float d1 = euclideanDistance(u, v, lut[idx1].u, lut[idx1].v);
        float d2 = std::numeric_limits<float>::infinity();
        if (idx1 != idx2)
            d2 = euclideanDistance(u, v, lut[idx2].u, lut[idx2].v);

        best_idx = (d1 <= d2) ? idx1 : idx2;
    }

    cct = lut[best_idx].cct;
    duv = lut[best_idx].Duv;

    return refine (u, v, cct, duv, lut);
}


std::pair<float, float> CctHandleF32::ComputeCct (const std::pair<float, float>& uv, eCOLOR_OBSERVER observer)
{
    bool bReady = true;
    float Cct = 0.f;
    float Duv = 0.f;

    const float u = uv.first;
    const float v = uv.second;

    if (observer_CIE_1931 == observer)
    {
        if (false == g_flagL1)
            bReady = InitializeLut1();

        // we working with LUT1
        cct_compute(u, v, Cct, Duv, m_Lut1);
    }
    else if (observer_CIE_1964 == observer)
    {
        if (false == g_flagL2)
            bReady = InitializeLut2();

        // we working with LUT2
        cct_compute(u, v, Cct, Duv, m_Lut2);
    }

    return std::make_pair (Cct, Duv);
}


const std::pair<float, float> CctHandleF32::getPlanckianUV (float cct, eCOLOR_OBSERVER observer)
{
    std::vector<CCT_LUT_Entry<float>>& lut = (observer_CIE_1931 == observer ? m_Lut1 : m_Lut2);
    const std::size_t lutSize = lut.size();

    std::pair<float, float> out{};
    float u, v;

    // Clamp to min/max of LUT range
    if (cct <= lut[0].cct)
    {
        u = lut[0].u;
        v = lut[0].v;
        out = std::make_pair(u, v) ;
    }
    else if (cct >= lut[lutSize - 1].cct)
    {
        u = lut[lutSize - 1].u;
        v = lut[lutSize - 1].v;
        out = std::make_pair(u, v);
    }
    else
    {
        // Search for bracketing interval
        for (size_t i = 0u; i < lutSize - 1; ++i)
        {
            const CCT_LUT_Entry<float>& p0 = lut[i];
            const CCT_LUT_Entry<float>& p1 = lut[i + 1];

            if (cct >= p0.cct && cct <= p1.cct)
            {
                const float t = (cct - p0.cct) / (p1.cct - p0.cct);
                u = p0.u + t * (p1.u - p0.u);
                v = p0.v + t * (p1.v - p0.v);
                out = std::make_pair(u, v);
                break;
            }
        }
    }

    return out;
}
