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
        if (tId != tableId)
        {
            // validate parameters
            if (m_sideSize != sizeM || cutF != cutFreq)
            {
                // re-compute table 
            }
            // update table ID
            uint32_t prev = tableId.exchange(tId);
        } // if (tId != tableId)

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

    bool AllocMemory(void)
    {
        std::size_t totalSize = m_sideSize * m_sideSize * sizeof(GaussianT);
    }

    void FreeMemory (void)
    {
        if (nullptr != m_pFilter)
        {

        }
    }

    bool Recompute(std::size_t sizeM, GaussianT cutFreq)
    {
        return true;
    }


}; // class GaussianHpFilter