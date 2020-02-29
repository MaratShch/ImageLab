#pragma once

#include <functional>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include "CCommon.hpp"

// #define _JOB_PROFILING

class CLASS_EXPORT CAsyncJob
{
public:
	CAsyncJob(void);

	CAsyncJob(CAsyncJob const& a)
	{
		m_Algorithm = a.m_Algorithm;
		memcpy(m_AlgParams, a.m_AlgParams, sizeof(m_AlgParams));
		m_site = a.m_site;
		m_completeNotify = a.m_completeNotify;
		m_validJob = a.m_validJob;
	}

	CAsyncJob& operator = (CAsyncJob const& a)
	{
		m_Algorithm = a.m_Algorithm;
		memcpy(m_AlgParams, a.m_AlgParams, sizeof(m_AlgParams));
		m_completeNotify = a.m_completeNotify;
		m_validJob = a.m_validJob;
	}

	~CAsyncJob(void);

	/* non movable cass */
	CLASS_NON_MOVABLE(CAsyncJob);

	inline void assignBuffers(void* RESTRICT in, void* RESTRICT out) {m_inBuffer = in; m_outBuffer = out; m_tmpBuffer = nullptr;}
	inline void assignTmpBuffer(void* RESTRICT p) {m_tmpBuffer = p;}
	inline void assignSite(const void* RESTRICT pSite) { m_site = pSite; }
	inline void assignCache(void* RESTRICT pCache, const size_t& size) { m_Cache = pCache; m_CacheSize = size; }

	inline void assignAlgorithm(std::function<int32_t(CAsyncJob* pAlg)>& f, const void* RESTRICT pSite = nullptr, const bool& notify = true)
	{
		m_Algorithm = f;
		m_site = pSite;
		m_completeNotify = notify;
		m_validJob = true;
	}

	inline void putParamChar  (const uint32_t idx, const char&& ch)       { m_AlgParams[idx].ch = ch; }
	inline void putParamSize  (const uint32_t idx, const size_t&& size)   { m_AlgParams[idx].size = size; }
	inline void putParamInt8  (const uint32_t idx, const int8_t&& i8)     { m_AlgParams[idx].i8 = i8; }
	inline void putParamUInt8 (const uint32_t idx, const uint8_t&& ui8)   { m_AlgParams[idx].ui8 = ui8; }
	inline void putParamInt16 (const uint32_t idx, const int16_t&& i16)   { m_AlgParams[idx].i16 = i16; }
	inline void putParamUInt16(const uint32_t idx, const uint16_t&& ui16) { m_AlgParams[idx].ui16 = ui16; }
	inline void putParamInt32 (const uint32_t idx, const int32_t&& i32)   { m_AlgParams[idx].i32 = i32; }
	inline void putParamUInt32(const uint32_t idx, const uint32_t&& ui32) { m_AlgParams[idx].ui32 = ui32; }
	inline void putParamInt64 (const uint32_t idx, const int64_t&& i64)   { m_AlgParams[idx].i64 = i64; }
	inline void putParamUInt64(const uint32_t idx, const uint64_t&& ui64) { m_AlgParams[idx].ui64 = ui64; }
	inline void putParamFloat (const uint32_t idx, const float&& f32)     { m_AlgParams[idx].f32 = f32; }
	inline void putParamDouble(const uint32_t idx, const double&& f64)    { m_AlgParams[idx].f64 = f64; }
	inline void putParamPtr   (const uint32_t idx, void*&& p)             { m_AlgParams[idx].p = p; }
	inline void putParamResPtr(const uint32_t idx, void*&& __restrict pR) { m_AlgParams[idx].pR = pR; }

	inline const bool isCompleteNotify (void) {return m_completeNotify;}
	inline void releaseJob(void)
	{ 
		m_validJob = false; 
		m_Algorithm = nullptr; 
		m_site = nullptr;  
		m_Cache = m_inBuffer = m_outBuffer = m_tmpBuffer = nullptr; 
		m_CacheSize = 0u;
	}

#ifdef _JOB_PROFILING
	const uint64_t getJobDuration(void) {return m_timeProfStop - m_timeProfStart;}
#else
	const uint64_t getJobDuration(void) { return 0ull; }
#endif

#ifndef _DEBUG
	inline void Execute(void)
	{ 
#ifdef _JOB_PROFILING
		m_timeProfStart = RDTSC;
#endif
		m_opResult = ((nullptr != m_Algorithm && true == m_validJob) ? m_Algorithm(this) : -1);
#ifdef _JOB_PROFILING
		m_timeProfStop = RDTSC;
#endif
}
#else
	void Execute(void);
#endif

	inline int32_t getOperationResult(void) { return m_opResult; }


	void* operator new (size_t size)
	{
		return allocate_aligned_mem(size, CACHE_LINE);
	}

	void operator delete (void* p)
	{
		if (nullptr != p)
		{
			free_aligned_mem(p);
			p = nullptr;
		}
		return;
	}

protected:
private:
	/* algorithm parameteres */
	algParams m_AlgParams[maxAlgParams];

	/* algorithm function */
	std::function<int32_t(CAsyncJob* pAlg)> m_Algorithm;

	/* algorithm memory storages */
	void* RESTRICT m_inBuffer;
	void* RESTRICT m_outBuffer;
	void* RESTRICT m_tmpBuffer;
	void* RESTRICT m_Cache;
	size_t m_CacheSize;

	/* sweet pie site */
	const void* RESTRICT m_site;

#ifdef _JOB_PROFILING
	uint64_t m_timeProfStart;
	uint64_t m_timeProfStop;
#endif /* _JOB_PROFILING */

	std::atomic<int32_t> m_opResult;

	bool m_completeNotify;
	bool m_validJob;

};
