#pragma once 

#include <windows.h>
#include <atomic>
#include "ClassRestrictions.hpp"

class Semaphore
{
private:
	HANDLE   m_hSemaphore;
	std::atomic<uint32_t> m_atomicCount;
	const uint32_t m_maxCnt;

public:
	CLASS_NON_COPYABLE(Semaphore);
	CLASS_NON_MOVABLE(Semaphore);

	explicit Semaphore (uint32_t initial_count) noexcept :
	m_maxCnt(initial_count)
	{
		m_atomicCount = initial_count;
		m_hSemaphore = CreateSemaphore (NULL, m_atomicCount, m_atomicCount, NULL);
	};

	~Semaphore() noexcept
	{
		while (m_maxCnt - m_atomicCount)
		{
			ReleaseSemaphore (m_hSemaphore, 1, nullptr);
			m_atomicCount++;
		}
		CloseHandle(m_hSemaphore);
		m_hSemaphore = INVALID_HANDLE_VALUE;
	}

	bool Wait (int32_t timeWait = -1) noexcept
	{
		bool bRet = false;
		const DWORD dwWait = WaitForSingleObject(m_hSemaphore, (timeWait < 0 ? INFINITE : timeWait));
		if (WAIT_OBJECT_0 == dwWait)
		{
			m_atomicCount--;
			bRet = true;
		}
		return bRet;
	}

	bool Wait (void) noexcept
	{
		return Wait(-1);
	}

	bool Release (void) noexcept
	{
		bool bRet = false;
		LONG prevCnt = 0l;
		if (FALSE != ReleaseSemaphore(m_hSemaphore, 1, nullptr))
		{
			m_atomicCount++;
			bRet = true;
		}
		return bRet;
	}

};