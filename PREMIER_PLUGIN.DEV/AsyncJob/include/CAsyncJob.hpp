#pragma once

constexpr size_t workerQueueSize = 16u; /* 16 jobs per one worker */



class CAsyncJob
{
public:
	inline const bool isCompleteNotify (void) {return m_completeNotify;}
	inline void releaseJob(void) { m_validJob = false; }
protected:
private:
	bool m_completeNotify = true;
	bool m_validJob = false;
};

