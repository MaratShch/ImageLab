#pragma once

#define FAST_COMPUTE_EXTRA_PRECISION

#include "ClassRestrictions.hpp"
#include "FastAriphmetics.hpp"
#include <type_traits>
#include <atomic>
#include <mutex>
#include <windows.h>

using GaussianT = float;

class GaussianHpFilter final
{
public:
    CLASS_NON_COPYABLE(GaussianHpFilter);
    CLASS_NON_MOVABLE(GaussianHpFilter);

    GaussianHpFilter()
    {
        m_pFilter = m_pCenter = nullptr;
        m_sideSize = 0ull;
        cutF = static_cast<GaussianT>(0);
        tableId.store(0u);
        return;
    }

    ~GaussianHpFilter(void)
    {
        FreeMemory();
        cutF = static_cast<GaussianT>(0);
        m_sideSize = 0ull;
        m_pFilter = m_pCenter = nullptr;
        tableId.store(0u);
    }

    GaussianT* getFilter (uint32_t tId, std::size_t sizeM, GaussianT cutFreq)
    {
        const std::lock_guard<std::mutex> lock(protect);
        if (tId != tableId || nullptr == m_pFilter)
        {
            // validate parameters
            if (m_sideSize != sizeM || cutF != cutFreq)
            {
                // re-compute table 
            }
            // update table ID
            uint32_t prev = tableId.exchange(tId);

        } // if (tId != tableId || nullptr == m_pFilter)

        return m_pCenter;
    }

protected:
private:
    GaussianT* m_pFilter;
    GaussianT* m_pCenter;
    std::size_t m_sideSize; // this is will be square type filter
    GaussianT cutF;

    std::atomic<uint32_t> tableId;
    std::mutex protect;

    bool AllocMemory (std::size_t sizeM)
    {
        const std::size_t oddSize = sizeM | 0x1ull;
        const SIZE_T totalSize = static_cast<SIZE_T>(oddSize * oddSize * sizeof(GaussianT));
        constexpr DWORD allocType = MEM_RESERVE | MEM_COMMIT | MEM_TOP_DOWN;
         
        LPVOID p = VirtualAlloc (NULL, totalSize, allocType, PAGE_READWRITE);
        if (NULL != p)
        {
            m_pFilter = reinterpret_cast<GaussianT*>(p);
            m_pCenter = m_pFilter + (sizeM * (sizeM / 2) + (sizeM / 2));
            m_sideSize = sizeM;
        }
    }

    void FreeMemory (void)
    {
        if (nullptr != m_pFilter)
        {
            VirtualFree (reinterpret_cast<LPVOID>(m_pFilter), 0, MEM_RELEASE);
            m_pFilter = m_pCenter = nullptr;
            m_sideSize = 0ull;
        }
    }

    bool Recompute(std::size_t sizeM, GaussianT cutFreq)
    {
        return true;
    }


}; // class GaussianHpFilter