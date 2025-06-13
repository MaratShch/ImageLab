#pragma once

#define FAST_COMPUTE_EXTRA_PRECISION

#include "ClassRestrictions.hpp"
#include "StylizationEnums.hpp"
#include <type_traits>
#include <atomic>
#include <mutex>
#include <windows.h>


class GaussianHpFilter final
{
public:
    CLASS_NON_COPYABLE(GaussianHpFilter);
    CLASS_NON_MOVABLE(GaussianHpFilter);

    GaussianHpFilter(void)
    {
        m_pFilter = m_pCenter = nullptr;
        m_sizeM = m_sizeN = 0ull;
        m_cutF = static_cast<GaussianT>(0);
        tableId.store(0xFFFFFFFFu);
        return;
    }

    ~GaussianHpFilter(void)
    {
        FreeMemory();
        m_cutF = static_cast<GaussianT>(0);
        m_sizeM = m_sizeN = 0ull;
        m_pFilter = m_pCenter = nullptr;
        tableId.store(0xFFFFFFFFu);
    }

    GaussianT* getFilter (SequenceIdT tId, std::size_t sizeM, std::size_t sizeN, GaussianT cutFreq)
    {
        bool bMemValid = true;

        const std::lock_guard<std::mutex> lock(protect);
        if (tId != tableId)
        {
            // validate parameters
            if (nullptr == m_pFilter)
                bMemValid = AllocMemory(sizeM, sizeN);

            if (m_sizeM != sizeM || m_sizeN != sizeN)
            {
                bMemValid = false;
                FreeMemory();
                bMemValid = AllocMemory(sizeM, sizeN);
            }

            if (true == bMemValid)
            {
                m_cutF = cutFreq;

                // recompute filter
                Recompute();

                // update table ID
                const SequenceIdT prevId = tableId.exchange(tId);
            }

        } // if (tId != tableId)

        return m_pCenter;
    }

protected:
private:
    GaussianT* m_pFilter;
    GaussianT* m_pCenter;
    std::size_t m_sizeM; // column size
    std::size_t m_sizeN; // line size
    GaussianT m_cutF;

    std::atomic<SequenceIdT> tableId;
    std::mutex protect;

    bool AllocMemory (std::size_t sizeM, std::size_t sizeN)
    {
        const std::size_t oddSizeM = sizeM | 0x1ull; // column size
        const std::size_t oddSizeN = sizeN | 0x1ull; // line size
        const SIZE_T totalSize = static_cast<SIZE_T>(oddSizeM * oddSizeN * sizeof(GaussianT));
        constexpr DWORD allocType = MEM_RESERVE | MEM_COMMIT | MEM_TOP_DOWN;
        bool ret;

        LPVOID p = VirtualAlloc (NULL, totalSize, allocType, PAGE_READWRITE);
        if (NULL != p)
        {
#ifdef _DEBUG
            memset(p, 0, static_cast<std::size_t>(totalSize));
#endif
            m_pFilter = reinterpret_cast<GaussianT*>(p);
            m_pCenter = m_pFilter + (oddSizeM * (oddSizeN >> 1) + (oddSizeN >> 1));
            m_sizeM = sizeM;
            m_sizeN = sizeN;
            ret = true;
        }
        else
        {
            m_pFilter = m_pCenter = nullptr;
            m_sizeM = m_sizeN = 0ull;
            m_cutF = static_cast<GaussianT>(0);
            ret = false;
        }

        return ret;
    }

    void FreeMemory (void)
    {
        if (nullptr != m_pFilter)
        {
            VirtualFree (reinterpret_cast<LPVOID>(m_pFilter), 0, MEM_RELEASE);
            m_pFilter = m_pCenter = nullptr;
            m_sizeM = m_sizeN = 0ull;
            m_cutF = static_cast<GaussianT>(0);
        }
    }

    void Recompute (void)
    {
        const int32_t halfM = static_cast<int32_t>(m_sizeM >> 1);
        const int32_t halfN = static_cast<int32_t>(m_sizeN >> 1);
        std::size_t idx = 0ull;
        const GaussianT cutFreqSq = m_cutF * m_cutF * static_cast<GaussianT>(2);
        constexpr GaussianT one{ 1 };

        for (int32_t j = -halfM; j <= halfM; j++)
        {
            for (int32_t i = -halfN; i <= halfN; i++)
            {
                const GaussianT d_squared = j * j + i * i;
                const GaussianT exponent = -d_squared / cutFreqSq;
                m_pFilter[idx] = one - std::exp(exponent);
                idx++;
            }
        }
        return;
    }


}; // class GaussianHpFilter