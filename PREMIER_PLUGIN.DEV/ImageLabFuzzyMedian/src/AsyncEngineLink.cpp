#include <windows.h>
#include "CThreadPool.hpp"

#ifdef _DEBUG
#pragma comment(lib, "../Debug/AsyncJob.lib")
#else
#pragma comment(lib, "../Release/AsyncJob.lib")
#endif

extern CLASS_EXPORT CThreadPool& GetThreadPool(void);
CThreadPool& parallelEngine = GetThreadPool();

const uint32_t totalAsyncJobs = CThreadPool::getCpuCoresNumber();