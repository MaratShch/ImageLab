#include "CAsyncJob.hpp"

CAsyncJob::CAsyncJob(void)
{
	m_completeNotify = true;
	m_validJob = false;
	m_opResult = 0;
	m_inBuffer = m_outBuffer = m_tmpBuffer = nullptr;
	m_site = nullptr;
	m_Algorithm = nullptr;
#ifdef _JOB_PROFILING
	m_timeProfStart = m_timeProfStop = 0ull;
#endif
	memset(m_AlgParams, 0, sizeof(m_AlgParams));
	return;
}

CAsyncJob::~CAsyncJob(void)
{
	m_completeNotify = true;
	m_validJob = false;
	m_opResult = 0;
	m_inBuffer = m_outBuffer = m_tmpBuffer = nullptr;
	m_site = nullptr;
	m_Algorithm = nullptr;

	memset(m_AlgParams, 0, sizeof(m_AlgParams));
	return;
}

#ifdef _DEBUG
void CAsyncJob::Execute(void)
{
#ifdef _JOB_PROFILING
	m_timeProfStart = RDTSC;
#endif
	
	if (nullptr != m_Algorithm && true == m_validJob)
	{
		if (nullptr != m_inBuffer && nullptr != m_outBuffer)
		{
			m_opResult = m_Algorithm(this);
		}
		else
		{
			std::cout << "Invalid memory reached. inBuffer = " << m_inBuffer << " outBuffer = " << m_outBuffer << std::endl;
			m_opResult = -2;
		}
	}
	else
	{
		std::cout << "Invalid Job reached" << std::endl;
		m_opResult = -1;
	}
#ifdef _JOB_PROFILING
	m_timeProfStop = RDTSC;
#endif

}
#endif