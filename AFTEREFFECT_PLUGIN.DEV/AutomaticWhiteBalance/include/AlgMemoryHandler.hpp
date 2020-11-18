#pragma once

#include <windows.h>
#include <thread>
#include <mutex>
#include <atomic>
#include "ClassRestrictions.hpp"

class CAlgMemHandler
{
public:
	CLASS_NON_COPYABLE(CAlgMemHandler);
	CLASS_NON_MOVABLE (CAlgMemHandler);

	CAlgMemHandler()  noexcept;
	~CAlgMemHandler() noexcept;
	
	bool MemInit(const size_t& size) noexcept;

	inline void* __restrict GetMemory(const uint32_t& idx)
	{
		const uint32_t i = idx & 0x1u;
		m_protect[i].lock();
		return m_p[i]; 
	} 

	inline void ReleaseMemory(const uint32_t& idx)
	{
		const uint32_t i = idx & 0x1u;
		m_protect[i].unlock();
	}


private:
	std::atomic<size_t> m_memBytesSize;
	std::recursive_mutex m_protect[2];
	void* __restrict m_p[2];

	void MemFree(void) noexcept;

};


CAlgMemHandler* getMemoryHandler(void);